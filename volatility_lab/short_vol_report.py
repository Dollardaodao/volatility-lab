"""Readable pre-cost and post-cost comparisons for delta-hedged straddles."""
import json
import os
import numpy as np
import pandas as pd
from .theta_sdk import ROOT

LABELS={'garch_t':'GARCH','ewma':'EWMA','rolling21':'21日滚动方差',
        'har_proxy_ridge':'HAR代理回归','ml_hist_gradient_boosting':'ML梯度提升树'}


def make_report(out,plan,complete=True):
    frame=pd.read_csv(out/'all_results.csv',low_memory=False)
    decisions=pd.read_csv(out/'decisions.csv',low_memory=False)
    completed=frame[frame.status=='completed'].copy()
    keys=['asset','origin','target_dte']
    counts=completed.groupby(keys).size()
    eligible=counts[counts==len(plan['models'])*2*2].index
    paired=completed[pd.MultiIndex.from_frame(completed[keys]).isin(eligible)].copy()
    paired.to_csv(out/'paired_results.csv',index=False)
    summaries=[];cost_rows=[]
    for (direction,model),f in paired.groupby(['direction','model']):
        actual=f[f.cost_scenario=='observed_costs'];ideal=f[f.cost_scenario=='no_execution_cost']
        active=actual[actual.side!=0];iactive=ideal[ideal.side!=0]
        scorable=active[active.vol_bet_status=='scorable']
        summaries.append({'direction':direction,'model':model,'completed_candidates':len(actual),
            'trades':len(active),'market_pnl_before_costs':active.market_pnl.sum(),
            'no_execution_cost_net_pnl':ideal.net_pnl.sum(),'observed_net_pnl':actual.net_pnl.sum(),
            'execution_cost':active.execution_cost.sum(),'abandonment_writeoff':active.abandonment_writeoff.sum(),
            'net_financing':active.net_financing.sum(),'no_cost_net_financing':iactive.net_financing.sum(),
            'execution_drag_including_interest_effect':ideal.net_pnl.sum()-actual.net_pnl.sum(),
            'vol_bet_correct':int(scorable.vol_bet_correct.astype(bool).sum()),'vol_bet_scorable':len(scorable),
            'vol_bet_undefined':len(active)-len(scorable),'observed_net_wins':int((active.net_pnl>0).sum()),
            'no_cost_net_wins':int((iactive.net_pnl>0).sum()),'median_holding_days':active.holding_days.median(),
            'early_exercise_risk_closes':int((active.exit_reason=='short_exercise_risk_close').sum()),
            'holding_lt_one_day':int((active.holding_days<1).sum()),
            'worst_observed_trade':active.net_pnl.min(),'largest_individual_drawdown':active.max_drawdown_usd.max(),
            'mean_observed_notional_return_pct':actual.return_on_notional.mean()*100,
            'mean_no_cost_notional_return_pct':ideal.return_on_notional.mean()*100})
        cost_rows.append({'direction':direction,'model':model,'trades':len(active),
            'executed_option_spread':active.option_spread_cost.sum()-active.abandonment_writeoff.sum(),
            'stock_spread':active.stock_spread_cost.sum(),'extra_stock_slippage':active.slippage_cost.sum(),
            'option_commissions':active.option_fees.sum(),'stock_commissions_and_sell_surcharge':active.stock_fees.sum(),
            'total_execution_cost':active.execution_cost.sum(),'abandoned_rights_writeoff':active.abandonment_writeoff.sum(),
            'stock_borrow':active.borrow_cost.sum(),'cash_interest_minus_benchmark':(active.cash_interest-active.benchmark_interest).sum(),
            'net_financing_after_borrow':active.net_financing.sum()})
    summary=pd.DataFrame(summaries);summary.to_csv(out/'strategy_summary.csv',index=False)
    costs=pd.DataFrame(cost_rows);costs.to_csv(out/'cost_breakdown.csv',index=False)
    actual=paired[(paired.cost_scenario=='observed_costs')&(paired.side!=0)].copy()
    ideal=paired[(paired.cost_scenario=='no_execution_cost')&(paired.side!=0)][keys+['model','direction','net_pnl']].rename(columns={'net_pnl':'no_execution_cost_net_pnl'})
    actual=actual.merge(ideal,on=keys+['model','direction'],validate='one_to_one')
    actual.to_csv(out/'active_trade_details.csv',index=False)
    frame[frame.status!='completed'].to_csv(out/'failed_scenarios.csv',index=False)
    actual[actual.vol_bet_status!='scorable'].to_csv(out/'undefined_vol_bets.csv',index=False)
    for dimension in ('asset','target_dte','origin'):
        actual.groupby(['direction','model',dimension]).agg(trades=('net_pnl','size'),
            market_pnl=('market_pnl','sum'),no_execution_cost_net_pnl=('no_execution_cost_net_pnl','sum'),
            observed_net_pnl=('net_pnl','sum'),execution_cost=('execution_cost','sum'),
            net_wins=('net_pnl',lambda x:int((x>0).sum()))).reset_index().to_csv(out/f'by_{dimension}.csv',index=False)
    funnel=decisions.groupby(['direction','model']).agg(candidates=('side','size'),
        signals_after_filters=('side',lambda x:int((x!=0).sum())),
        known_exercise_risk_rejections=('rejected_known_exercise_risk',lambda x:int(x.sum()))).reset_index()
    funnel.to_csv(out/'signal_funnel.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    english={'garch_t':'GARCH','ewma':'EWMA','rolling21':'Rolling 21d','har_proxy_ridge':'HAR-proxy Ridge','ml_hist_gradient_boosting':'ML Gradient Boosting'}
    for ax,direction in zip(axes,['short','long']):
        s=summary[summary.direction==direction].set_index('model').loc[plan['models']]
        x=np.arange(len(s));ax.bar(x-.19,s.mean_no_cost_notional_return_pct,width=.38,label='No execution costs; financing kept',color='#367b68')
        ax.bar(x+.19,s.mean_observed_notional_return_pct,width=.38,label='Observed bid/ask + modeled fees',color='#bd554d')
        ax.set_xticks(x,[english[n] for n in s.index],rotation=24,ha='right',fontsize=8)
        ax.axhline(0,color='grey',lw=.7);ax.set_title(f'{direction.title()} straddles · 60-minute hedges')
        ax.set_ylabel('Mean net P&L / entry stock notional (%)');ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'before_after_costs.png',dpi=180);plt.close(fig)
    def table(f):return f.to_html(index=False,border=0,float_format=lambda x:f'{x:,.2f}')
    def main_table(direction):
        s=summary[summary.direction==direction].set_index('model').loc[plan['models']].reset_index()
        return pd.DataFrame({'模型':s.model.map(LABELS),'入场笔数':s.trades,
            '成本前市场损益':s.market_pnl_before_costs,'无执行成本净损益':s.no_execution_cost_net_pnl,
            '现实净损益':s.observed_net_pnl,
            '押对波动率次数':[f'{int(a)}/{int(b)}' for a,b in zip(s.vol_bet_correct,s.vol_bet_scorable)],
            '现实盈利次数':[f'{int(a)}/{int(b)}' for a,b in zip(s.observed_net_wins,s.trades)]})
    short=summary[summary.direction=='short']
    best=short.sort_values('mean_observed_notional_return_pct',ascending=False).iloc[0]
    verdict='按每笔一组跨式的美元加总，本轮五个卖方模型扣成本后均未取得正净损益。' if (short.observed_net_pnl<=0).all() else '本轮有卖跨式模型在所列执行及风险退出假设下获得正净损益；这不等同于真实指派及经纪商保证金下的已验证盈利。'
    qlike=pd.read_csv(ROOT/'artifacts'/'ml_vol_study'/'forecast_scores.csv')[['model','observations','qlike','mae_vol_points']]
    qlike['observations']=qlike.observations.astype(int)
    qlike['model']=qlike.model.map(LABELS)
    costs_display=costs.copy();costs_display['model']=costs_display.model.map(LABELS)
    risk=summary[['direction','model','trades','vol_bet_undefined','median_holding_days','holding_lt_one_day',
                  'early_exercise_risk_closes','worst_observed_trade','largest_individual_drawdown']].copy()
    risk['model']=risk.model.map(LABELS)
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>卖跨式：成本前后与波动率押注</title>
<style>body{{background:#faf9f6;color:#263238;font:16px/1.7 system-ui;margin:0}}main{{max-width:1300px;margin:auto;padding:35px 25px}}h1{{font-size:32px}}h2{{margin-top:36px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:8px 11px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}a{{color:#23685c}}.note{{background:#e9efeb;padding:18px}}</style><main>
<h1>卖跨式 vs 买跨式：先看无执行成本，再看现实结果</h1>
<p class="note">{'完整研究' if complete else '部分运行结果'}：20只股票、8个原始历史起点、7/30/90天目标期限、每笔一张call加一张put、60分钟整股Delta对冲。共同完整候选{len(eligible)}个。{verdict}</p>
<h2>先看卖跨式（美元，各模型独立选交易）</h2>{table(main_table('short'))}
<p><strong>三列损益的含义：</strong>成本前市场损益＝期权中间价变化＋股票对冲收益＋股票现金股息，不含执行费用、融资或借股。无执行成本净损益＝真正把同一批交易和订单按中间价重放，佣金、卖出附加费和额外滑点为0，仍保留融资与借股。现实净损益＝历史bid/ask成交、所列佣金与滑点、融资及借股全部计入。两次重放的订单、入场和平仓时间均核对相同；无成本结果不是简单把手续费加回。</p><p>两个“净损益”均已减去同本金闲置现金利息，属于相对现金基准的超额损益；该基准扣除不是交易手续费。现实版本是历史报价下的成交及费用假设，不是已发生的实盘成交。</p>
<p>卖出规则：预测波动率低于两腿bid-IV较低者3个波动率百分点，并通过原始成本门槛；原来已知会触发提前行权风险退出的候选在入场前跳过。卖出按bid，买回按ask。历史有明确0 bid但正ask且正常报价和足够挂单量时，空头可以按真实ask买回，不需要虚构双边市场。</p>
<h2>对应买跨式（同一共同候选，原四模型现实损益保持不变）</h2>{table(main_table('long'))}
<p>买入规则保持预测高于两腿ask-IV较高者3个百分点，且通过同一成本门槛。无执行成本版本保留这批成本感知入场信号，没有重新降低门槛来制造更多交易。买方零bid权利仍按原保守规则零回收并承诺不行权，日志不记为成交。相对中间价的剩余权利放弃折价另列为abandonment_writeoff，不是券商收取的手续费；该金额在理想版本也保留，不虚构mid卖出。</p>
<img src="before_after_costs.png"><p>图表以每个候选损益/入场100股标的名义金额的平均值比较，包含空仓0。美元表为独立交易账户加总，不能当作共同资金组合的收益率。二者均不是年化收益。每笔固定一组跨式时，不同股票价格及拆股前后的美元敞口差异很大；因此美元加总与等标的名义金额平均值可能不同号，不能混为同一个资金配置方案。</p>
<h2>到底押对波动率几次？</h2><p>卖跨式：实际入场至实际平仓的RV小于现实入场时两腿bid-IV的较低者；买跨式：同一窗口RV大于两腿ask-IV的较高者。RV使用15分钟及隔夜对数收益平方和，按实际持有年数年化。IV来自相同美式模型、相同真实入场时刻。两成本版本沿用现实入场报价作为判断参照，因此不会因理想成交假设改变押对次数。</p>
<p>这是普通RV与IV的简明检验，不是gamma加权损益恒等式，也不是GARCH点预测误差。提前平仓、IV变化、gamma随价格变化、离散对冲和费用都会令押对次数与盈利次数不同。发生入场bid/ask无法反演IV、零持有时间等情况时，现金损益照样保留，只把押注诊断记为不可计算；分母仅使用可计算交易，未定义数量在下面公开。</p>
<h2>持有期、提前退出与尾部损失</h2>{table(risk)}
<p><strong>卖跨式的限制：</strong>沿用既有主动提前行权风险退出规则，以当时观察到的时间价值、滞后利率和过去分红代理判断，并在下一有效快照平仓。这没有模拟真实美式提前指派、股票交割或真实券商强平。短期持有及提前退出比例必须与收益一起看，不能把早退结果当作持有至原到期日的完整波动率押注。每笔3倍入场100股名义金额的研究本金规则不是交易所保证金。表中最大单笔回撤也不是实际多笔并发组合回撤。</p>
<h2>交易成本究竟是什么？</h2>
<table><tr><th>项目</th><th>来源与本轮处理</th></tr><tr><td>期权与股票买卖价差</td><td>真实历史快照；买ask、卖bid，相对mid的差额在账目中只计一次。</td></tr><tr><td>期权佣金</td><td>研究假设每张、每腿、每次1美元；一组跨式正常开平仓共4次合约交易，即4美元。</td></tr><tr><td>股票佣金</td><td>研究假设每股0.0035美元，每单最低0.35美元。</td></tr><tr><td>额外股票滑点</td><td>研究假设每次股票成交额的1bp，即0.01%，在bid/ask之外另加。</td></tr><tr><td>股票卖出附加费</td><td>研究假设卖出金额的0.1bp，即0.001%。</td></tr><tr><td>融资与借股</td><td>另列；闲置现金SOFR减50bp最低0，借款SOFR加150bp，借股年费率2%，减同本金闲置现金基准。</td></tr></table>
<p>费率和滑点是研究假设，不是用户账户实际费率。例：一腿bid=4.90、ask=5.10、mid=5.00，卖出后报价不变再买回，损失(5.10−4.90)×100＝20美元，即使券商免佣也存在。两腿同样价差共40美元，再加开平仓合约佣金4美元，尚未计入股票对冲成本。时间价值衰减已包含在期权报价变化中，属于市场损益。</p>
{table(costs_display)}
<p>现实账目的精确等式为：净损益＝成本前市场损益−total_execution_cost−abandoned_rights_writeoff＋net_financing_after_borrow。理想版本比现实版本多赚的金额还包含执行费用导致的后续利息差异，因此不应直接称为佣金。</p>
<h2>QLIKE是什么？</h2><p>QLIKE是方差预测误差评分，不是收益率、押对概率或交易胜率。本项目采用非负归一化形式：</p>
<p><code>QLIKE = R/F − ln(R/F) − 1</code>，F为预测方差，R为之后观察到的方差代理。F与R必须使用相同区间和单位；波动率需要先平方。预测与观察值相等时为0，越低越好。例：预测20%、实际40%，得分约1.614；预测40%、实际20%，约0.636，低估相同倍数的波动会受到更大惩罚。</p>
<p>之前0.318表示该模型在月度预测样本上的平均损失函数值，不是31.8%胜率。之前评分的目标是日线平方收益方差代理，每个模型共同2840个标签；本表押对率使用实际交易窗口的15分钟RV和可成交IV，属于另一种检验。</p>{table(qlike)}
<h2>样本、失败与复现</h2>{table(funnel)}
<p>候选仍为480个。所有模型、方向和成本版本仅在共同完整数据上比较，失败保存在原始结果，不能把失败视为0收益。本轮不重新训练或选择最赚钱参数；GARCH/EWMA及两ML模型复用已保存的当时预测，新增21日滚动方差遵守相同过去信息与精确到期汇总。原来多头现实结果通过现金结果和原始订单复用保持不变。它是已查看历史上的探索性比较，仍只有8个市场起点，重叠期限和股票相关，未建立实盘盈利证据。报价是15分钟重采样快照，原始tick新鲜度无法据此验证；股票挂单量仍按原研究每round-lot为100股的单位假设约束模拟成交。</p>
<p><a href="strategy_summary.csv">成本前后汇总</a> · <a href="cost_breakdown.csv">成本逐项</a> · <a href="active_trade_details.csv">所有入场逐笔结果及押注判断</a> · <a href="all_results.csv">全部场景</a> · <a href="failed_scenarios.csv">失败清单</a> · <a href="undefined_vol_bets.csv">不可计算押注</a> · <a href="../../short_vol_plan.json">固定方案</a> · <a href="run_configuration.json">来源哈希</a></p>
<p>复现：<code>.venv/bin/python -m volatility_lab.short_vol_study --workers 4</code>。原始研究文件保留不变，本轮新增成交、现金账目在trades目录，逐起点完整结果在origins目录。</p>
<p>参考：<a href="https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf">Patton：方差代理和预测损失函数</a> · <a href="https://www.optionseducation.org/strategies/all-strategies/short-straddle">OIC：卖跨式和提前指派</a>。</p></main></html>'''
    (out/'report.html').write_text(text)
    print(summary[['direction','model','trades','no_execution_cost_net_pnl','observed_net_pnl','vol_bet_correct','vol_bet_scorable']].to_string(index=False),flush=True)
