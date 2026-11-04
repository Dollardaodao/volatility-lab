"""Timing-aware comparison of longer stock hedges for the fixed option book."""
import json
import os
import pandas as pd
from .theta_sdk import ROOT


def aggregate(frame,keys):
    rows=[]
    for names,g in frame.groupby(keys,dropna=False):
        if not isinstance(names,tuple):names=(names,)
        scored=g[g.vol_bet_status=='scorable']
        rows.append({**dict(zip(keys,names)),'trades':len(g),'net_pnl':g.net_pnl.sum(),
            'profitable_trades':int((g.net_pnl>0).sum()),'market_pnl':g.market_pnl.sum(),
            'execution_cost':g.execution_cost.sum(),'abandonment_writeoff':g.abandonment_writeoff.sum(),
            'net_financing':g.net_financing.sum(),'stock_orders':int(g.stock_orders.sum()),
            'stock_turnover_shares':int(g.stock_turnover_shares.sum()),'mean_abs_delta_shares':g.mean_abs_delta_shares.mean(),
            'mean_abs_delta_dollars':g.mean_abs_delta_dollars.mean(),
            'delta_dollar_scorable_trades':int(g.mean_abs_delta_dollars.notna().sum()),
            'trades_with_some_missing_delta':int(g.missing_delta_fraction.gt(0).sum()),
            'mean_missing_delta_pct':g.missing_delta_fraction.mean()*100,
            'mean_trade_return_on_capital_pct':g.return_on_capital.mean()*100,
            'p95_individual_trade_drawdown_pct':(g.max_drawdown_usd/g.capital).quantile(.95)*100,
            'changed_exits':int((~g.exit_at_original_time.astype(bool)).sum()),
            'vol_bets_scorable':len(scored),'vol_bets_correct_15m':int(scored.vol_bet_correct_15m.sum()),
            'option_spread_cost':g.option_spread_cost.sum()-g.abandonment_writeoff.sum(),
            'stock_spread_cost':g.stock_spread_cost.sum(),'slippage_cost':g.slippage_cost.sum(),
            'option_fees':g.option_fees.sum(),'stock_fees':g.stock_fees.sum()})
    return pd.DataFrame(rows)


def make_report(out,plan):
    f=pd.read_csv(out/'all_results.csv',low_memory=False);assert f.status.eq('completed').all()
    summary=aggregate(f,['protocol_id','cost_scenario']);summary.to_csv(out/'summary.csv',index=False)
    comparison=summary.pivot(index='protocol_id',columns='cost_scenario',values='net_pnl').reset_index()
    observed=summary[summary.cost_scenario=='observed_costs']
    fields=['protocol_id','trades','profitable_trades','execution_cost','stock_orders','stock_turnover_shares',
            'mean_abs_delta_shares','mean_abs_delta_dollars','delta_dollar_scorable_trades','trades_with_some_missing_delta',
            'mean_missing_delta_pct','p95_individual_trade_drawdown_pct','changed_exits','vol_bets_correct_15m']
    comparison=comparison.merge(observed[fields],on='protocol_id',validate='one_to_one')
    ideal=summary[summary.cost_scenario=='no_execution_cost'][['protocol_id','profitable_trades']].rename(columns={'profitable_trades':'profitable_trades_before_execution'})
    comparison=comparison.merge(ideal,on='protocol_id',validate='one_to_one')
    order={p['id']:i for i,p in enumerate(plan['protocols'])};labels={p['id']:p['label'] for p in plan['protocols']}
    comparison['protocol_label']=comparison.protocol_id.map(labels);comparison['display_order']=comparison.protocol_id.map(order)
    comparison=comparison.sort_values('display_order');comparison.to_csv(out/'comparison.csv',index=False)
    for dimension in ['asset','origin','target_dte','kind']:
        aggregate(f,[dimension,'protocol_id','cost_scenario']).to_csv(out/f'by_{dimension}.csv',index=False)
    years=f.assign(entry_year=f.origin.str[:4]);annual=aggregate(years,['entry_year','protocol_id','cost_scenario'])
    annual.to_csv(out/'by_entry_year.csv',index=False)
    cols=['protocol_id','option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees',
          'execution_cost','abandonment_writeoff','net_financing']
    observed[cols].to_csv(out/'cost_breakdown.csv',index=False)
    keys=['case_id','asset','origin','target_dte','kind']
    trade=f.pivot(index=keys,columns=['protocol_id','cost_scenario'],values='net_pnl')
    trade.columns=[a+'_'+b for a,b in trade.columns];trade=trade.reset_index();trade.to_csv(out/'trade_comparison.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    english={'elapsed_60m':'60 calendar min','elapsed_120m':'2 calendar hours','elapsed_720m':'12 calendar hours',
             'daily_1000':'Daily 10:00 NY','daily_close':'Daily close minus 15m','trading_720m':'12 trading hours'}
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.6),sharey=True,layout='constrained')
    y=range(len(comparison));names=[english[n] for n in comparison.protocol_id]
    for ax,column,title in zip(axes,['no_execution_cost','observed_costs','mean_abs_delta_shares'],
            ['P&L before execution costs','P&L after execution costs','Residual model Delta on 1m grid']):
        values=comparison[column]
        displayed=values/1000 if column!='mean_abs_delta_shares' else values
        ax.barh(y,displayed,color=['#176a5b' if v>=0 else '#a45131' for v in values])
        ax.set_yticks(list(y),names);ax.set_title(title,fontsize=10)
        ax.set_xlabel('USD thousands' if column!='mean_abs_delta_shares' else 'Mean absolute Delta in shares')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.axvline(0,color='grey',lw=.6);ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.savefig(out/'longer_hedges.png',dpi=170);plt.close(fig)
    table=lambda data:data.to_html(index=False,border=0,float_format=lambda v:f'{v:,.3f}')
    presented=comparison[['protocol_label','no_execution_cost','observed_costs','profitable_trades_before_execution',
                          'profitable_trades','stock_orders','execution_cost','mean_abs_delta_shares','changed_exits']]
    risk=comparison[['protocol_label','mean_abs_delta_shares','mean_abs_delta_dollars','delta_dollar_scorable_trades',
                     'trades_with_some_missing_delta','mean_missing_delta_pct','p95_individual_trade_drawdown_pct']]
    yearly=annual[annual.cost_scenario=='observed_costs'].pivot(index='entry_year',columns='protocol_id',values='net_pnl').reset_index()
    times=pd.DataFrame([{'protocol_id':p['id'],'protocol_label':p['label'],'clock':p['schedule_kind'],
        'frequency_minutes':p['frequency'],'daily_time':p.get('daily_time','close minus 15m' if p['schedule_kind']=='daily_close' else '')} for p in plan['protocols']])
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>990笔单腿交易的低频对冲比较</title>
<style>body{{background:#faf9f6;color:#263238;font:16px/1.7 system-ui;margin:0}}main{{max-width:1320px;margin:auto;padding:32px 24px}}h1{{font-size:30px}}h2{{margin-top:34px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}a{{color:#23685c}}.note{{background:#e9efeb;padding:18px}}</style><main>
<h1>把对冲间隔延长到2小时、12小时和每日一次</h1>
<p class="note">固定原20只股票、2022–2025月度扫描产生的990笔做多单腿Call或Put。沿用同一1分钟市场报价、同一1分钟前Delta、原合约和退出目标、原费用及融资。没有重新筛选股票、阈值或期权，也没有新数据购买。原60分钟现金与订单逐笔复现后，再比较新增日程。</p>
<h2>完整结果</h2>{table(presented)}
<p>每行990笔，一张期权合约；美元为独立出资交易的加总，不是共享资本或净额对冲组合收益率。profitable_trades_before_execution与profitable_trades分别是不扣与扣执行成本的盈利笔数。no_execution_cost保留融资、股票借股、闲置现金利息基准和弃权损失。changed_exits标记因资本约束等原因偏离原退出的笔数；非零时不能把全部差额都归给频率。</p>
<img src="longer_hedges.png">
<h2>日程具体怎样执行</h2>{table(times)}
<p>所有版本先在入场后的首个合格分钟建立股票对冲。每日版本把这一天计作已对冲，不在入场当日额外做第二次日程调仓；之后每日最多一次。若计划时刻当前或上一分钟股票报价不合格、过去Delta不可用，就等待之后首个合格分钟。即使目标股票数量没变化、没有订单，该次日程仍算已处理。</p>
<p><b>2个自然小时：</b>距离上一次日程尝试满120分钟后，在正常时段首个合格快照调整；夜间计时但不夜间成交。因此隔夜后通常在09:31附近重新调仓，随后按间隔继续。</p>
<p><b>12个自然小时：</b>满720个自然分钟后调整，但美股正常时段只有约6.5小时。因此通常是入场先对冲，然后每个后续交易日开盘后首个合格分钟一次；不能据此宣称夜间也做了对冲。</p>
<p><b>每日纽约10点：</b>入场后的后续交易日，在10:00或之后首个合格分钟调整。它与12个自然小时的频次通常相近，但开盘附近与10点的执行价格、价差和Delta不同。</p>
<p><b>每日收盘前15分钟：</b>后续交易日在交易所日历规定的收盘前15分钟调整，半日市按提前收盘时间计算。最后平仓日如果此时已经关闭期权，就直接平掉股票，不再做一笔多余的Delta再平仓。</p>
<p><b>累计12个交易小时：</b>只累计同一交易日内开市时间，隔夜、周末和假日不计时，半日市按实际开市长度计时。满720个开市分钟后才调整，通常比每日一次更稀疏。行情无效会阻止成交，但不会停止交易时钟。</p>
<p>所有版本在期权平仓时强制清掉剩余股票，股票清仓订单计入stock_orders和实际成本。日程调仓只用t−1分钟的Delta，清仓由事先的退出规则触发；没有用当前快照Delta假设立刻成交。</p>
<h2>成本与风险一起看</h2>{table(risk)}{table(observed[cols])}
<p>mean_abs_delta_shares在同一持仓1分钟网格上逐笔求绝对残余模型Delta平均，再等权汇总；缺失Delta分钟不参与该均值，trades_with_some_missing_delta及mean_missing_delta_pct披露缺失范围。mean_abs_delta_dollars把残余股数乘以当时股票价后求平均，仅汇总整个持有路径模型Delta都可评分的交易，数量为delta_dollar_scorable_trades。这些指标不能代表缺失Delta时段的真实风险。p95_individual_trade_drawdown_pct为单笔相对闲置现金基准最大回撤占该笔初始资本的95分位，不是组合回撤。低频可能减少交易成本，但同时留下更大的方向暴露；若出现利润，不能只归因于预测波动率准确。</p>
<p>期权按ask买、bid卖，每张每次$1；股票按历史bid/ask并加1bp滑点，每股佣金$0.0035且每单至少$0.35，卖出附加费0.1bp。融资和借股沿用原假设。零执行成本对照保留同一订单、同一实际退出，按mid成交并去除执行费用；对零bid放弃的权利仍零回收。现金利息按相邻快照的实际时间间隔计提，股票股息按除息日记账，价差及费用只通过成交现金扣一次。</p>
<h2>各年份扣执行成本后的结果</h2>{table(yearly)}
<p>全部日程为看到新损益前确定的敏感性实验，仍然使用此前已研究过的时期，属于探索性历史结果。个股名义本金不同，月度持仓及期限重叠，没有组合净额对冲，不能从美元加总推导年化收益或最优未来调仓规则。</p>
<h2>数据与复现</h2><p><a href="comparison.csv">汇总</a> · <a href="cost_breakdown.csv">成本拆分</a> · <a href="by_asset.csv">按股票</a> · <a href="by_entry_year.csv">按年份</a> · <a href="by_origin.csv">按月份</a> · <a href="by_target_dte.csv">按期限</a> · <a href="trade_comparison.csv">逐笔损益</a> · <a href="hourly_reproduction.csv">旧60分钟结果复现</a> · <a href="configuration.json">方案、输入与代码哈希</a> · <a href="verification.json">核对结果</a> · <a href="../../longer_hedge_plan.json">固定日程方案</a></p>
<p>每笔每种日程的现金账和订单在trades目录；使用原始分钟priced缓存，不生成新的价格。原四频率研究、Word与PDF报告保留。复现：<code>.venv/bin/python -m volatility_lab.longer_hedge_study --workers 6</code>；核对：<code>.venv/bin/python -m volatility_lab.longer_hedge_validate</code>。</p></main></html>'''
    (out/'report.html').write_text(body);print(presented.to_string(index=False),flush=True)


if __name__=='__main__':
    make_report(ROOT/'artifacts/longer_hedge_study',json.loads((ROOT/'longer_hedge_plan.json').read_text()))
