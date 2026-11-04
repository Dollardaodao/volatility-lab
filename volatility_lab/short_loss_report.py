"""Presentation for loss attribution, exit/hedge diagnostics and vol-gap gates."""
import json
import os
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .stock_replay import exercise_risk

LABELS={'garch_t':'GARCH','ewma':'EWMA','rolling21':'21日滚动方差',
        'har_proxy_ridge':'HAR代理回归','ml_hist_gradient_boosting':'ML梯度提升树'}


def audit_exit_triggers(out,payloads):
    records=[]
    for payload in payloads:
        case=payload['case']
        if case['exit_reason']!='short_exercise_risk_close':continue
        panel=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/(case['case_id']+'_priced.csv'),index_col=0,parse_dates=True)
        panel.index=pd.to_datetime(panel.index,utc=True)
        fills=pd.read_csv(out/'cases'/case['case_id']/'baseline_observed_costs_fills.csv')
        close=fills[(fills.instrument=='call')&(fills.quantity>0)].iloc[-1]
        time=pd.Timestamp(close.decision_time);row=panel.loc[time];strike=case['strike']
        records.append({'case_id':case['case_id'],'trigger_time':str(time),
            'trigger_time_new_york':str(time.tz_convert('America/New_York')),'actual_exit':case['exit'],
            'trigger_matches_original_rule':exercise_risk(row,strike),
            'stock_valid':bool(row.stock_passes_snapshot_filters),'option_pair_valid':bool(row.options_valid),
            'spot':row.spot,'call_mid':row.call_mid,'put_mid':row.put_mid,
            'call_extrinsic_mid':row.call_mid-max(row.spot-strike,0),
            'put_extrinsic_mid':row.put_mid-max(strike-row.spot,0),
            'quarterly_dividend_proxy':row.dividend_yield*row.spot/4,
            'put_extrinsic_trigger_limit':max(.02,strike*row.rate/365)})
    audit=pd.DataFrame(records);audit.to_csv(out/'exit_trigger_snapshot_audit.csv',index=False)
    return audit


def make_report(out,plan,complete=True):
    threshold=pd.read_csv(out/'threshold_summary.csv');counter=pd.read_csv(out/'counterfactual_results.csv',low_memory=False)
    cases=pd.read_csv(out/'worst_unique_trades.csv',low_memory=False)
    forecasts=pd.read_csv(out/'matched_forecast_diagnostics.csv')
    quality=pd.read_csv(out/'quote_quality_audit.csv')
    payloads=[json.loads(f.read_text()) for f in (out/'cases').glob('*/diagnostics.json')]
    payloads=sorted(payloads,key=lambda p:p['case']['ideal_net_pnl'])
    table=lambda f:f.to_html(index=False,border=0,float_format=lambda x:f'{x:,.3f}')
    summary=cases[['case_id','model','decision_spot','strike','capital','entry_execution_iv','holding_realized_vol',
        'gross_option_pnl','hedge_gross_pnl','dividends','market_pnl','ideal_net_pnl','net_pnl','exit_reason']].copy()
    summary['entry_execution_iv']*=100;summary['holding_realized_vol']*=100
    summary['entry_stock_notional']=summary.capital/3
    summary['no_cost_loss_pct_of_stock_notional']=summary.ideal_net_pnl/summary.entry_stock_notional*100
    summary['model']=summary.model.map(LABELS)
    summary.to_csv(out/'worst_loss_cash_bridge.csv',index=False)
    effects=[];case_sections=[]
    for payload in payloads:
        case=payload['case'];case_id=case['case_id'];folder=out/'cases'/case_id
        c=counter[(counter.case_id==case_id)&(counter.status=='completed')]
        pivot=c.pivot(index=['exit_policy','hedge_minutes'],columns='cost_scenario',values='net_pnl').reset_index()
        pivot['change_vs_baseline_no_cost']=pivot.no_execution_cost-case['ideal_net_pnl']
        def value(policy,frequency,cost):
            f=c[(c.exit_policy==policy)&(c.hedge_minutes==frequency)&(c.cost_scenario==cost)]
            return float(f.net_pnl.iloc[0]) if len(f) else np.nan
        base=value('fixed_original_actual_exit',60,'no_execution_cost')
        scheduled=c[(c.exit_policy=='scheduled_preexpiry_without_proactive_exercise_close')&(c.hedge_minutes==60)&(c.cost_scenario=='observed_costs')]
        scheduled_meta=scheduled.iloc[0] if len(scheduled) else None
        effects.append({'case_id':case_id,'baseline_no_cost':base,
            'fixed_exit_15m_no_cost':value('fixed_original_actual_exit',15,'no_execution_cost'),
            'fixed_exit_15m_change':value('fixed_original_actual_exit',15,'no_execution_cost')-base,
            'scheduled_exit_60m_no_cost':value('scheduled_preexpiry_without_proactive_exercise_close',60,'no_execution_cost'),
            'scheduled_exit_60m_change':value('scheduled_preexpiry_without_proactive_exercise_close',60,'no_execution_cost')-base,
            'baseline_observed_costs':case['net_pnl'],
            'scheduled_requested_exit_reached':bool(scheduled_meta.exit_at_requested_time) if scheduled_meta is not None else False,
            'scheduled_variant_actual_exit':scheduled_meta['exit'] if scheduled_meta is not None else '',
            'scheduled_variant_exit_reason':scheduled_meta.exit_reason if scheduled_meta is not None else ''})
        f=forecasts[forecasts.case_id==case_id].copy();f['model']=f.model.map(LABELS)
        for col in ['forecast_vol_to_expiry','forecast_vol_for_actual_holding','signal_bid_iv','entry_bid_iv','actual_holding_rv','rv_entry_to_expiry']:
            f[col]*=100
        f=f[['model','forecast_vol_to_expiry','forecast_vol_for_actual_holding','signal_bid_iv','entry_bid_iv',
             'actual_holding_rv','decision_gap_vol_points','actual_entry_gap_vol_points']]
        intervals=pd.read_csv(folder/'largest_loss_intervals.csv')
        intervals['stock_return_pct']=intervals.stock_return*100
        display_intervals=intervals[['time_new_york','overnight','stock_return_pct','shares_before_interval',
            'delta_residual_before_interval','option_mark_pnl','hedge_pnl','market_pnl']]
        fixed=pivot[pivot.exit_policy=='fixed_original_actual_exit'].drop(columns='exit_policy')
        exits=pivot[pivot.hedge_minutes==60].drop(columns='hedge_minutes')
        exit_meta=c[(c.hedge_minutes==60)&(c.cost_scenario=='observed_costs')][['exit_policy','exit','exit_reason','exit_at_requested_time']]
        exits=exits.merge(exit_meta,on='exit_policy',validate='one_to_one')
        models=', '.join(LABELS[n] for n in case['selected_by_models'])
        case_sections.append(f'''<h2>{case_id}：同时被 {models} 选中</h2>
<p>原退出：{case['exit']}（{case['exit_reason']}）；原计划退出：{case['planned_exit']}。入场100股名义金额约${case['capital']/3:,.2f}，无执行成本亏损占该名义金额{case['ideal_net_pnl']/(case['capital']/3)*100:.3f}%。</p>
<p>预测与RV（波动率列单位为%，gap列为波动率百分点）：</p>{table(f)}
<p>固定原实际退出时间，只改变对冲最短间隔：</p>{table(fixed)}
<p>固定60分钟对冲，对比原实际退出与关闭主动行权风险早退后的版本。后者仍保留研究资本约束，也没有模拟途中真实美式指派；exit_at_requested_time为False时并未持有到原计划退出：</p>{table(exits)}
<details><summary>最差区间的期权、股票与残余Delta损益</summary>{table(display_intervals)}<p>overnight表示跨交易日。这里的期权区间变化包含时间、IV和非线性效应，不能全部叫gamma；残余Delta也只是原定价模型下的估计。无有效期权对价时沿用过去标记，区间跳变可能包含报价更新，因此不单凭某一根柱断言发生了交易损失。全窗口金额与现金账目核对一致。</p></details>
<p><a href="cases/{case_id}/fill_cost_audit.csv">这笔所有订单的逐项费用</a> · <a href="cases/{case_id}/interval_attribution.csv">全部持仓区间</a> · <a href="cases/{case_id}/forecast_horizon_audit.csv">预测时间窗口</a></p>''')
    effects=pd.DataFrame(effects);effects.to_csv(out/'hedge_exit_effects.csv',index=False)
    def threshold_table(direction):
        f=threshold[threshold.direction==direction][['model','threshold_vol_points','trades','no_execution_cost_net_pnl',
            'observed_net_pnl','vol_bet_correct','vol_bet_scorable','mean_active_observed_notional_pct',
            'mean_candidate_observed_notional_pct']].copy()
        f['model']=f.model.map(LABELS)
        return f.sort_values(['model','threshold_vol_points'])
    periods=pd.read_csv(out/'threshold_by_period.csv')
    periods=periods[(periods.direction=='short')&(periods.model.isin(['garch_t','ml_hist_gradient_boosting']))]
    periods=periods[['model','threshold_vol_points','period','trades','no_execution_cost_net_pnl','observed_net_pnl']].copy()
    periods['model']=periods.model.map(LABELS)
    first=payloads[0];audit=pd.read_csv(out/'cases'/first['case']['case_id']/'fill_cost_audit.csv')
    trigger_table=table(audit_exit_triggers(out,payloads))
    examples=pd.concat([audit[audit.instrument!='stock'],audit[(audit.instrument=='stock')&(audit.quantity>0)].head(1),
                         audit[(audit.instrument=='stock')&(audit.quantity<0)].head(1)]).sort_values('time_utc')
    examples=examples[['time_new_york','instrument','quantity','bid','ask','mid','fill_price','spread_cost',
                       'added_slippage','commission','sell_surcharge','total_fee','cash_flow']]
    examples.to_csv(out/'actual_order_examples.csv',index=False)
    buy=audit[(audit.instrument=='stock')&(audit.quantity>0)].iloc[0]
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    english={'garch_t':'GARCH','ewma':'EWMA','rolling21':'Rolling 21d','har_proxy_ridge':'HAR-proxy Ridge','ml_hist_gradient_boosting':'ML Gradient Boosting'}
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for model,f in threshold[threshold.direction=='short'].groupby('model'):
        for ax,col in zip(axes,['mean_candidate_observed_notional_pct','mean_active_observed_notional_pct','trades']):
            ax.plot(f.threshold_vol_points,f[col],marker='o',label=english[model])
    for ax in axes:ax.set_xlabel('Entry gap (absolute volatility points)');ax.set_xticks(plan['thresholds_vol_points']);ax.spines[['top','right']].set_visible(False)
    axes[0].set(title='Mean after-cost result incl. flat candidates',ylabel='P&L / entry stock notional (%)')
    axes[1].set(title='Mean after-cost result among actual trades',ylabel='P&L / entry stock notional (%)')
    axes[2].set(title='Entries retained',ylabel='Number of cohort entries')
    axes[0].axhline(0,color='grey',lw=.7);axes[1].axhline(0,color='grey',lw=.7);axes[2].legend(fontsize=7)
    fig.savefig(out/'threshold_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,min(2,len(payloads)),figsize=(13,4.8),layout='constrained',squeeze=False)
    for ax,payload in zip(axes[0],payloads[:2]):
        case_id=payload['case']['case_id']
        f=counter[(counter.case_id==case_id)&(counter.status=='completed')]
        values=[];labels=[]
        for frequency in [15,60,240,390]:
            row=f[(f.exit_policy=='fixed_original_actual_exit')&(f.hedge_minutes==frequency)&(f.cost_scenario=='no_execution_cost')]
            values.append(float(row.net_pnl.iloc[0]));labels.append(f'{frequency}m\nfixed exit')
        row=f[(f.exit_policy=='scheduled_preexpiry_without_proactive_exercise_close')&(f.hedge_minutes==60)&(f.cost_scenario=='no_execution_cost')]
        values.append(float(row.net_pnl.iloc[0]));labels.append('60m\nno risk exit')
        values.append(payload['case']['net_pnl']);labels.append('60m\nobserved costs')
        ax.bar(range(len(values)),values,color=['#367b68']*4+['#537eaa','#bd554d'])
        for i,v in enumerate(values):ax.annotate(f'{v:,.0f}',(i,v),xytext=(0,-13 if v<0 else 5),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(len(values)),labels,fontsize=8)
        ax.set_ylim(min(values)*1.2,max(0,max(values))*1.2+300)
        ax.axhline(0,color='grey',lw=.7);ax.set_title(case_id);ax.set_ylabel('Final excess P&L (USD)')
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'worst_trade_counterfactuals.png',dpi=170);plt.close(fig)
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>空波动亏损拆解、对冲和阈值</title>
<style>body{{background:#faf9f6;color:#263238;font:16px/1.7 system-ui;margin:0}}main{{max-width:1400px;margin:auto;padding:35px 25px}}h1{{font-size:32px}}h2{{margin-top:36px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}a{{color:#23685c}}.note{{background:#e9efeb;padding:18px}}</style><main>
<h1>为什么免执行成本的卖跨式仍亏？</h1><p class="note">{'完整诊断' if complete else '部分诊断'}：按不扣执行成本的美元亏损选取最差8个不同股票/起点/期权合约，重复入场的模型列在同一笔下。阈值敏感性覆盖所有五模型、买卖两个方向、全部478个共同完整候选。所有原始结果保留不变。</p>
<h2>先把亏损本身拆开</h2>{table(summary)}
<p>gross_option_pnl是期权中间价变化，hedge_gross_pnl是股票对冲价格变化，dividends是实际股票现金股息；三者相加得到market_pnl。无执行成本的ideal_net_pnl仍包含融资及借股并减闲置现金基准；现实net_pnl再计入bid/ask、佣金、附加费、额外滑点。它们是不同层面的会计结果。股票对冲亏钱不能直接称为对冲失误：短gamma通常随上涨买入更多股票、下跌卖出股票，波动实现得过大时，期权所赚时间价值可能不足以支付这部分损失。</p>
<p>尤其是拆股前AMZN：当时每组跨式对应约33万美元标的名义金额。相同“一组期权”在不同股票、拆股前后对应的风险金额不同，美元大亏不能直接等同于相同本金下的百分比大亏。上表另列no_cost_loss_pct_of_stock_notional；本报告没有把比率归一化冒充可实际成交的零碎合约组合。</p>
<img src="worst_trade_counterfactuals.png"><p>图中前四根柱为固定原退出时刻、免执行成本的对冲频率对照；第五根为取消主动行权风险退出的60分钟对照（没有实际指派模型）；最后一根为原60分钟含执行成本结果。均使用已闭合的最终现金损益，不把中间沿用报价造成的标记尖峰当作已实现亏损。</p>
<h2>固定入场后的诊断：对冲频率和提前退出</h2>{table(effects)}
<p>fixed_exit_15m_change表示固定原实际退出时间，改为15分钟对冲后无执行成本损益的变化；scheduled_exit_60m_change表示保持60分钟对冲，取消主动提前行权风险退出、保留原研究资本约束后的变化。正数仅表示这笔历史路径下该对照更好，负数表示更差。NVDA 2024年1月90天目标的对照先触发研究资本约束，未持有到原计划退出，不能当作完整等待结果。原实际退出时间是事后固定的条件诊断，不是可预知的实盘退出策略。禁用主动早退也没有模拟真实美式指派，不能据此推荐裸露等待到期。</p>
<h2>把entry门槛从3提高到5、8、10点</h2><p>这里是<strong>绝对波动率百分点</strong>：例如bid-IV为30%，5点门槛要求模型预测低于25%，不是30%乘5%。保留相同的成本门槛、09:45信号、10:00成交、风险过滤、60分钟对冲和退出规则；更高门槛只会删掉原有交易，不会用之后收益决定是否入场。</p>
<img src="threshold_comparison.png"><h2>卖方完整阈值结果</h2>{table(threshold_table('short'))}
<p>mean_active_observed_notional_pct是实际入场交易的平均损益/入场100股标的名义金额；mean_candidate_observed_notional_pct包含空仓0。提高门槛后总亏损变小，可能主要来自交易数减少，必须同时看实际入场平均结果。零交易的0损益不是盈利证据。阈值研究是在看过3点结果后按用户要求进行的历史敏感性，5/8/10全部报告，不把其中最好一格当成独立验证的最优参数。</p>
<details><summary>买方对应阈值结果</summary>{table(threshold_table('long'))}</details>
<h2>分时期检查（2025已被看过，不是新盲测）</h2>{table(periods)}
<h2>我实际怎么计算每一笔交易成本</h2>
<p>期权：卖出取当时bid，买回取ask；每张、每腿、每次佣金1美元。股票：买价＝ask×1.0001，卖价＝bid×0.9999；佣金＝max(0.35美元，成交股数×0.0035美元)；股票卖出另加成交金额×0.00001的附加费。股票1bp额外滑点、佣金和卖出附加费均为研究假设；bid/ask来自历史快照。它们不是用户账户实际费率，也不是ThetaData订阅费。</p>
<p><code>订单现金流＝−带符号成交数量×合约乘数×成交价−手续费</code>。期权乘数100，股票乘数1。账本直接使用这个现金流；价差和滑点另列只是相对mid的损益拆解，不再额外扣第二次。融资、借股和同本金现金基准分别记录，不混入执行佣金；时间价值损耗包含在期权价格变化中。</p>
<p>真实订单例（{first['case']['case_id']}）：买{int(buy.quantity)}股，bid={buy.bid:.6f}，ask={buy.ask:.6f}，mid={buy.mid:.6f}。模拟成交价={buy.ask:.6f}×1.0001={buy.fill_price:.6f}；价差成本${buy.spread_cost:.6f}，额外滑点${buy.added_slippage:.6f}，佣金${buy.commission:.6f}，合计相对mid成本${buy.spread_cost+buy.added_slippage+buy.total_fee:.6f}。现金支出${-buy.cash_flow:.6f}。</p>
{table(examples)}<p>该笔整段持仓的所有订单均按同样公式重新计算，并核对期权价差、股票价差、滑点、两类佣金以及最终现金损益：<a href="cases/{first['case']['case_id']}/fill_cost_audit.csv">全部逐单核算</a>。</p>
<h2>报价和退出触发器检查</h2>{table(quality)}
<p>RV_valid_stock_snapshots只使用通过原股票报价过滤的快照，并将相邻有效快照的收益连接，供污染敏感性检查。期权无有效双边报价时，仅中间标记可沿用过去信息；实际成交仍需符合原退出报价规则。退出时的期权中间价时间价值可能接近0甚至为负，主动早退触发器并不等于实际发生提前指派，也可能受重采样和不同步报价影响。因此下列反事实用于诊断规则敏感性，不能精确因果认定某个模型或退出规则独自造成多少美元损失。</p>
<p>下面按真实平仓订单日志的decision_time检查触发快照，区别于实际成交时刻。部分中间价时间价值为负；宽价差下的中间价也不能直接当作可成交价格。AMZN 30天目标的退出信号出现在前一日收盘，实际平仓在次日9:45，途中真实指派仍未模拟。</p>{trigger_table}
{''.join(case_sections)}
<h2>原始结果与复现</h2><p><a href="threshold_summary.csv">全部阈值汇总</a> · <a href="threshold_selected_trades.csv">各阈值留下的交易</a> · <a href="counterfactual_results.csv">全部对冲/退出反事实</a> · <a href="matched_forecast_diagnostics.csv">持有期匹配的预测</a> · <a href="fill_audit_totals.csv">逐单现金核对汇总</a> · <a href="../../short_loss_diagnostics_plan.json">固定诊断方案</a> · <a href="run_configuration.json">源文件与数据哈希</a></p>
<p>复现：<code>.venv/bin/python -m volatility_lab.short_loss_diagnostics --workers 4</code>。不购买新数据。旧股票、ML和卖跨式结果保留，费用实例可追溯至其原始报价和成交日志。</p></main></html>'''
    (out/'report.html').write_text(text)
