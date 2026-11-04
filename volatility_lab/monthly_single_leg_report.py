"""Reviewable monthly single-option results; report all opportunity outcomes."""
import json
import os
import pandas as pd
from .theta_sdk import ROOT


def aggregate(frame,keys):
    rows=[]
    for names,g in frame.groupby(keys,dropna=False):
        if not isinstance(names,tuple):names=(names,)
        complete=g[g.status=='completed'];scored=complete[complete.vol_bet_status=='scorable']
        row={**dict(zip(keys,names)),'selected_opportunities':len(g),'completed_trades':len(complete),
             'entry_rejected':int((g.status=='entry_rejected').sum()),'unresolved':int((g.status=='replay_failed').sum()),
             'net_pnl':complete.net_pnl.sum(),'profitable_trades':int((complete.net_pnl>0).sum()),
             'capital_reserve_closes':int(complete.observed_exit_reason.eq('research_capital_reserve_close').sum()),
             'off_schedule_exits':int(complete.exit_at_planned_time.eq(False).sum()),
             'market_pnl':complete.market_pnl.sum(),'execution_cost':complete.execution_cost.sum(),
             'abandonment_writeoff':complete.abandonment_writeoff.sum(),'net_financing':complete.net_financing.sum(),
             'stock_orders':int(complete.stock_orders.sum()),'stock_turnover_shares':int(complete.stock_turnover_shares.sum()),
             'mean_abs_delta_shares_equal_trade':complete.mean_abs_delta_shares.mean(),
             'mean_trade_return_on_capital_pct':complete.return_on_capital.mean()*100,
             'vol_bets_scorable':len(scored),'vol_bets_correct_15m':int(scored.vol_bet_correct_15m.sum()),
             'vol_bets_correct_1m':int(scored.vol_bet_correct_1m.sum()),
             'option_spread_cost':complete.option_spread_cost.sum()-complete.abandonment_writeoff.sum(),
             'stock_spread_cost':complete.stock_spread_cost.sum(),'slippage_cost':complete.slippage_cost.sum(),
             'option_fees':complete.option_fees.sum(),'stock_fees':complete.stock_fees.sum()}
        rows.append(row)
    return pd.DataFrame(rows)


def make_report(out):
    f=pd.read_csv(out/'all_results.csv',low_memory=False);decisions=pd.read_csv(out/'decisions.csv',low_memory=False)
    summary=aggregate(f,['hedge_minutes','cost_scenario']);summary.to_csv(out/'summary.csv',index=False)
    table_summary=summary.pivot(index='hedge_minutes',columns='cost_scenario',values='net_pnl').reset_index()
    observed=summary[summary.cost_scenario=='observed_costs']
    displaycols=['hedge_minutes','completed_trades','entry_rejected','unresolved','profitable_trades','capital_reserve_closes','off_schedule_exits','execution_cost',
                 'stock_orders','mean_abs_delta_shares_equal_trade','vol_bets_scorable','vol_bets_correct_15m','vol_bets_correct_1m']
    table_summary=table_summary.merge(observed[displaycols],on='hedge_minutes',validate='one_to_one').sort_values('hedge_minutes',ascending=False)
    ideal_wins=summary[summary.cost_scenario=='no_execution_cost'][['hedge_minutes','profitable_trades']].rename(columns={'profitable_trades':'profitable_trades_no_execution_cost'})
    table_summary=table_summary.merge(ideal_wins,on='hedge_minutes',validate='one_to_one')
    table_summary.to_csv(out/'frequency_comparison.csv',index=False)
    counts=decisions.groupby('status').size().rename('opportunities').reset_index();counts.to_csv(out/'opportunity_counts.csv',index=False)
    selected=decisions[decisions.status=='selected'];kinds=selected.groupby(['target_dte','kind']).size().rename('selected_contracts').reset_index()
    kinds.to_csv(out/'selected_contract_counts.csv',index=False)
    grouped={}
    for dimension in ['asset','origin','target_dte','kind']:
        grouped[dimension]=aggregate(f,[dimension,'hedge_minutes','cost_scenario'])
        grouped[dimension].to_csv(out/f'by_{dimension}.csv',index=False)
    def hourly_groups(dimension):
        g=grouped[dimension];g=g[g.hedge_minutes==60]
        result=g.pivot(index=dimension,columns='cost_scenario',values='net_pnl').reset_index()
        actual=g[g.cost_scenario=='observed_costs'][[dimension,'completed_trades','profitable_trades','vol_bets_correct_15m']]
        return result.merge(actual,on=dimension,validate='one_to_one')
    stocks=hourly_groups('asset');tenors=hourly_groups('target_dte')
    annual=f.assign(entry_year=f.origin.str[:4])
    years=aggregate(annual,['entry_year','hedge_minutes','cost_scenario']);years.to_csv(out/'by_entry_year.csv',index=False)
    yearly=years[years.hedge_minutes==60].pivot(index='entry_year',columns='cost_scenario',values='net_pnl').reset_index()
    costs=observed[['hedge_minutes','option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees',
                    'execution_cost','abandonment_writeoff','net_financing']]
    costs.to_csv(out/'cost_breakdown.csv',index=False)
    complete=f[f.status=='completed'];tradekeys=['case_id','asset','origin','target_dte','kind','strike','expiry_date']
    pertrade=complete.pivot(index=tradekeys,columns=['hedge_minutes','cost_scenario'],values='net_pnl')
    pertrade.columns=[f'{m}m_{s}' for m,s in pertrade.columns];pertrade=pertrade.reset_index();pertrade.to_csv(out/'trade_comparison.csv',index=False)
    bet=complete[(complete.hedge_minutes==60)&(complete.cost_scenario=='no_execution_cost')]
    betcols=tradekeys+['forecast_vol','decision_ask_iv','vol_gap','entry_ask_iv','holding_days','holding_rv_15m','holding_rv_1m',
                      'vol_bet_status','vol_bet_correct_15m','vol_bet_correct_1m']
    bet[betcols].to_csv(out/'vol_bets.csv',index=False)
    failures=f[f.status!='completed'];failures.to_csv(out/'execution_failures.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for scenario,color in [('no_execution_cost','#1b6f60'),('observed_costs','#a4582c')]:
        g=summary[summary.cost_scenario==scenario].sort_values('hedge_minutes')
        ax=axes[0] if scenario=='no_execution_cost' else axes[1]
        ax.plot(g.hedge_minutes,g.net_pnl,marker='o',color=color);ax.axhline(0,color='grey',lw=.7)
        ax.set(title=scenario.replace('_',' '),ylabel='Sum of independent trade P&L (USD)')
    g=observed.sort_values('hedge_minutes')
    axes[2].plot(g.hedge_minutes,g.mean_abs_delta_shares_equal_trade,marker='o',color='#1b6f60')
    axes[2].set(title='Residual model Delta on the same minute grid',ylabel='Mean absolute Delta (shares), equal trade weight')
    for ax in axes:
        ax.set_xscale('log');ax.set_xticks([1,5,15,60],[1,5,15,60]);ax.set_xlabel('Hedge interval (minutes)');ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'monthly_single_option.png',dpi=165);plt.close(fig)
    table=lambda data:data.to_html(index=False,border=0,float_format=lambda v:f'{v:,.3f}')
    sourceplan=json.loads((ROOT/'monthly_single_leg_plan.json').read_text())
    quality=pd.DataFrame([{'case_id':p.parent.name,**json.loads(p.read_text())} for p in (out/'trades').glob('*/quality.json')])
    quality.to_csv(out/'quote_quality.csv',index=False)
    status_note='所有已选交易均已完成核算。' if not (f.status=='replay_failed').any() else '存在已选持仓尚无法完成退出或核算；下列加总只含已完成部分，不能当作完整策略盈利结论。'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20只股票月度单腿波动率交易</title>
<style>body{{background:#faf9f6;color:#263238;font:16px/1.7 system-ui;margin:0}}main{{max-width:1280px;margin:auto;padding:32px 24px}}h1{{font-size:30px}}h2{{margin-top:34px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}a{{color:#23685c}}.note{{background:#e9efeb;padding:18px}}</style><main>
<h1>每月重新找机会：买预测优势最大的一张Call或Put</h1>
<p class="note">本实验重新扫描20只股票、2022–2025年48个月首个交易日，以及7／30／90天目标期限，共{len(decisions):,}个机会。使用GARCH预测与真实决策时点报价；每只股票／月份／目标期限最多买一张期权，Call和Put二选一，不组合跨式，不限制原来的22笔。共选中{len(selected):,}个单腿机会。{status_note}</p>
<h2>结果：同为1分钟信号延迟，只改变对冲间隔</h2>{table(table_summary)}
<p>completed_trades为实际完成的期权交易，entry_rejected为已选合约在10:00缺少可用执行或对冲条件而未入场的机会，unresolved为不能隐去的未完成回放。profitable_trades为扣执行成本后的盈利笔数。每笔一张合约，初始资金为该时点100股名义本金的三倍；美元加总不是共同资金组合收益率，重叠期限未共用资本。同月股票及重叠持有期相关，990个信号不等于990个统计独立样本。</p>
<img src="monthly_single_option.png">
<h2>如何产生交易</h2><p>09:45纽约时间，用截至前一交易日收盘可得的最近756个日收益拟合Student-t GARCH(1,1)，将未来日方差积分到实际合约到期时刻。预测没有使用当日收盘或未来已实现波动。按原到期容忍范围5／12／25天选择距离7／30／90天最近的可用到期日；分别取Call和Put中最接近平值的有效报价合约。</p>
<p>用相同美式Bjerksund–Stensland定价近似反演各合约ask-IV，按“预测年化波动率−该合约ask-IV”从大到小排序；最大差值严格大于3个百分点才买入一张，差值相同优先Call。只搜索上述接近平值候选，不声称搜索了全行权价最优交易。没有沿用双腿跨式的粗略事前成本门槛，也没有根据后来的盈亏换腿或选阈值。原理论fair price单独保留供检查。</p>
<p>10:00执行已选合约；若该腿报价不可成交，不换另一腿补做。各期限、各股票可同时持仓，每个机会独立出资。常规退出为到期前一交易日收盘前15分钟，允许该日后续有效报价平仓。用实际选择的单腿Delta调股票，整股取整，遵守显示数量限制；t时点调仓只使用t−1分钟的Delta及过去可得报价，隔夜和开盘跳空保留。</p>
<h2>所有机会去向</h2>{table(counts)}{table(kinds)}
<p>拆股跨期的调整合约目前没有可靠交割元数据，明确列为unsupported_split_in_holding_window，不用另一条期权或期限替代。这是事后识别的数据支持排除，可能影响样本代表性，不伪装成可实时预测拆股的交易规则。无报价、模型无法反演、未触发信号及执行失败均保留记录。</p>
<h2>按入场年份：每60分钟对冲</h2>{table(yearly)}<p>年度金额仍是每笔一张合约的独立出资加总。较早未拆股股票的每张名义金额更大，美元盈利可能集中在少数股票或年份；不把四年合计当作稳定的组合收益率。</p>
<h2>按股票和期限：每60分钟对冲</h2>{table(stocks)}{table(tenors)}<p>以上沿用原每60分钟对冲作为展示基线，profitable_trades为扣除执行成本后的盈利笔数；四种频率的全部分组结果保留在对应CSV。</p>
<h2>成本与现金核算</h2>{table(costs)}
<p>observed_costs：期权买ask、卖bid，每张每次假设$1佣金；股票买ask、卖bid，额外1bp滑点，每股$0.0035佣金且每单至少$0.35，股票卖出另计0.1bp。真实头寸没有变化就没有订单或佣金。价差来自历史报价，佣金、附加费及滑点沿用研究假设，并非核实过的用户实际券商费率。</p>
<p>no_execution_cost：同一已选合约按mid成交，佣金、卖出附加费和额外滑点为零；保留现金收益max(滞后SOFR−0.5%,0)、借款SOFR＋1.5%、股票借股年化2%，并减去同初始本金的闲置现金利息。为做费用归因，零执行成本对照固定沿用该频率下实际费用方案的退出时刻，包括费用可能触发的资本退出；这不是一套独立决定退出的零费策略。两组订单及市场损益逐笔核对。零bid的买入权利采用原保守不行权／零回收，writeoff单列，不假装按mid出售。</p>
<p>核算恒等式：净损益＝期权mid价格变化＋股票对冲损益＋股票股息−执行成本−权利放弃损失＋融资净收益。价差及滑点通过成交现金只扣一次，归因列不重复扣款。市场损益未扣执行成本及融资；两种费用方案现金余额不同，也会产生小幅融资差异。</p>
<h2>波动押对和交易赚钱分别统计</h2><p>以实际入场到退出的15分钟及隔夜RV大于所买单腿实际入场ask-IV，作为主“波动押对”指标；同时提供1分钟RV口径。它们不是Gamma加权实现方差，不保证单腿交易盈利；细网格RV受报价微观结构影响，不能因更换采样频率而声称模型变准。无法反演入场IV的诊断保留为未评分，现金损益照常计入。</p>
<p>逐笔命中见<a href="vol_bets.csv">vol_bets.csv</a>；逐笔频率比较见<a href="trade_comparison.csv">trade_comparison.csv</a>。上表的vol_bets_scorable、vol_bets_correct_15m和vol_bets_correct_1m分别给出可评分与两种口径的命中笔数。</p>
<h2>报价和模型局限</h2><p>只请求已选单腿的历史价格。Delta使用这条期权当时的市场mid-IV，无法反演时最多沿用过去4天的本腿IV；不使用未来或另一腿报价。缺失Delta时不生成新对冲指令，缺失比例保留。原始tick更新时间无法从分钟采样证明；一分钟采样、有信号延迟和整股交易不是数学连续对冲。</p>
<p>模型以日收益预测未来物理方差，不能解析日内季节性；美式模型用连续股息率近似，实际股票除息现金另入账。股票显示量每单位100股是沿用的研究假设。未模拟主动行权、真实成交排队、逐笔市场冲击或股票借券可得性。所有2022–2025结果属于已研究时期的探索性扩展。较原22笔同时改变了月度扫描和单腿策略，不能把整体差额只归因于取消跨式。</p>
<h2>数据、参数与审计</h2><p><a href="decisions.csv">全部2,880个机会</a> · <a href="frequency_comparison.csv">频率汇总</a> · <a href="all_results.csv">全部执行场景</a> · <a href="cost_breakdown.csv">费用拆分</a> · <a href="by_asset.csv">按股票</a> · <a href="by_origin.csv">按月份</a> · <a href="by_entry_year.csv">按入场年份</a> · <a href="by_target_dte.csv">按期限</a> · <a href="by_kind.csv">Call／Put</a> · <a href="execution_failures.csv">未完成及拒绝入场</a> · <a href="quote_quality.csv">报价质量</a> · <a href="replay_configuration.json">回放参数与哈希</a> · <a href="selection_configuration.json">选择参数与哈希</a> · <a href="verification.json">核对结果</a> · <a href="../../monthly_single_leg_plan.json">固定实验方案</a></p>
<p>trades目录逐笔保留单腿priced分钟数据、成交记录及现金账。现有订阅下取数，未购买或升级。复现：先运行<code>monthly_single_leg_download</code>与<code>monthly_single_leg_acquire --workers 1</code>，再运行<code>monthly_single_leg_study --workers 3</code>；审计运行<code>monthly_single_leg_validate</code>。</p></main></html>'''
    (out/'report.html').write_text(body);print(table_summary.to_string(index=False),flush=True)


if __name__=='__main__':make_report(ROOT/'artifacts/monthly_single_leg')
