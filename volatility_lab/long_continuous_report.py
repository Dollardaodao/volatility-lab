"""Long-volatility minute-hedge results, costs, and matched-trade diagnostics."""
import html
import json
import os
import numpy as np
import pandas as pd
from .theta_sdk import ROOT


def aggregate(frame, keys):
    rows = []
    for name, f in frame.groupby(keys, dropna=False):
        if not isinstance(name, tuple): name = (name,)
        rows.append({**dict(zip(keys, name)), 'trades': len(f), 'net_pnl': f.net_pnl.sum(),
                     'winning_trades': int((f.net_pnl > 0).sum()), 'market_pnl': f.market_pnl.sum(),
                     'execution_cost': f.execution_cost.sum(), 'abandonment_writeoff': f.abandonment_writeoff.sum(),
                     'net_financing': f.net_financing.sum(), 'stock_orders': int(f.stock_orders.sum()),
                     'stock_turnover_shares': int(f.stock_turnover_shares.sum()),
                     'mean_abs_delta_shares_equal_trade': f.mean_abs_delta_shares.mean(),
                     'mean_trade_return_on_capital_pct': f.return_on_capital.mean() * 100,
                     'option_spread_cost': f.option_spread_cost.sum() - f.abandonment_writeoff.sum(),
                     'stock_spread_cost': f.stock_spread_cost.sum(), 'slippage_cost': f.slippage_cost.sum(),
                     'option_fees': f.option_fees.sum(), 'stock_fees': f.stock_fees.sum()})
    return pd.DataFrame(rows)


def make_report(out):
    f = pd.read_csv(out / 'all_results.csv', low_memory=False)
    assert (f.status == 'completed').all()
    keys = ['data_source', 'hedge_minutes', 'signal_lag_minutes', 'cost_scenario']
    summary = aggregate(f, keys); summary.to_csv(out / 'aggregate_results.csv', index=False)
    pk = ['data_source', 'hedge_minutes', 'signal_lag_minutes']
    display = summary.pivot(index=pk, columns='cost_scenario', values='net_pnl').reset_index()
    observed = summary[summary.cost_scenario == 'observed_costs']
    extra = ['winning_trades', 'execution_cost', 'stock_orders', 'mean_abs_delta_shares_equal_trade']
    display = display.merge(observed[pk + extra], on=pk, validate='one_to_one')
    ideal = summary[summary.cost_scenario == 'no_execution_cost'][pk + ['winning_trades']].rename(columns={'winning_trades': 'winning_trades_no_execution_cost'})
    display = display.merge(ideal, on=pk, validate='one_to_one'); display.to_csv(out / 'frequency_comparison.csv', index=False)
    for dimension in ['asset', 'origin', 'target_dte']:
        aggregate(f[f.data_source == 'one_minute'], [dimension, 'hedge_minutes', 'signal_lag_minutes', 'cost_scenario']).to_csv(out / ('by_' + dimension + '.csv'), index=False)
    costrows = summary[(summary.data_source == 'one_minute') & (summary.signal_lag_minutes == 1) & (summary.cost_scenario == 'observed_costs')]
    costcols = ['hedge_minutes', 'option_spread_cost', 'stock_spread_cost', 'slippage_cost', 'option_fees', 'stock_fees', 'execution_cost', 'abandonment_writeoff', 'net_financing']
    costrows[costcols].to_csv(out / 'execution_cost_breakdown.csv', index=False)
    baseline = f[(f.data_source == 'archived_15m') & (f.cost_scenario == 'no_execution_cost')].copy()
    baseline['original_15m_vol_bet_correct'] = baseline.vol_bet_correct
    minute = f[(f.data_source == 'one_minute') & (f.hedge_minutes == 1) & (f.signal_lag_minutes == 1) & (f.cost_scenario == 'no_execution_cost')]
    bet = baseline[['case_id', 'asset', 'origin', 'target_dte', 'garch_t_forecast_vol', 'entry_execution_iv', 'holding_realized_vol', 'original_15m_vol_bet_correct']]
    bet = bet.rename(columns={'holding_realized_vol': 'holding_rv_original_15m'})
    bet = bet.merge(minute[['case_id', 'holding_realized_vol', 'vol_bet_correct']], on='case_id', validate='one_to_one').rename(columns={'holding_realized_vol': 'holding_rv_1m', 'vol_bet_correct': 'one_minute_vol_bet_correct'})
    bet.to_csv(out / 'vol_bets.csv', index=False)
    # Per-trade endpoint outcomes make winners, losers and costs reviewable.
    trades = f[(f.data_source == 'one_minute') & (f.signal_lag_minutes == 1)].pivot(
        index=['case_id', 'asset', 'origin', 'target_dte'], columns=['hedge_minutes', 'cost_scenario'], values='net_pnl')
    trades.columns = [f'{frequency}m_{scenario}' for frequency, scenario in trades.columns]
    trades = trades.reset_index(); trades.to_csv(out / 'trade_comparison.csv', index=False)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    fine = summary[summary.data_source == 'one_minute']
    for scenario, style in [('no_execution_cost', '-'), ('observed_costs', '--')]:
        for lag, color in [(1, '#14675a'), (15, '#aa6530')]:
            g = fine[(fine.cost_scenario == scenario) & (fine.signal_lag_minutes == lag)].sort_values('hedge_minutes')
            axes[0].plot(g.hedge_minutes, g.net_pnl, style, marker='o', color=color, label=f'{scenario}; lag {lag}m')
    for lag, color in [(1, '#14675a'), (15, '#aa6530')]:
        g = fine[(fine.cost_scenario == 'no_execution_cost') & (fine.signal_lag_minutes == lag)].sort_values('hedge_minutes')
        axes[1].plot(g.hedge_minutes, g.mean_abs_delta_shares_equal_trade, marker='o', color=color, label=f'Signal lag {lag}m')
    axes[0].axhline(0, color='grey', lw=.8)
    axes[0].set(title='All 22 original long straddles', ylabel='Sum of independently funded trade P&L (USD)')
    axes[1].set(title='Residual model Delta on the same 1m grid', ylabel='Mean absolute Delta (shares), equal trade weight')
    for ax in axes:
        ax.set_xscale('log'); ax.set_xticks([1, 5, 15, 60], [1, 5, 15, 60]); ax.set_xlabel('Hedge interval (minutes)')
        ax.spines[['top', 'right']].set_visible(False); ax.legend(fontsize=7)
    fig.savefig(out / 'long_minute_hedging.png', dpi=170); plt.close(fig)
    table = lambda data: data.to_html(index=False, border=0, float_format=lambda v: f'{v:,.3f}')
    main = display[(display.data_source == 'one_minute') & (display.signal_lag_minutes == 1)].sort_values('hedge_minutes', ascending=False)
    original_hits = int(bet.original_15m_vol_bet_correct.sum()); minute_hits = int(bet.one_minute_vol_bet_correct.sum())
    q = json.loads((out / 'quote_quality.json').read_text())
    qtable = pd.DataFrame([{'case_id': name, **{k: v for k, v in values.items() if not isinstance(v, dict)}} for name, values in q.items()])
    body = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原始做多波动策略：分钟级Delta对冲</title>
<style>body{{background:#faf9f6;color:#263238;font:16px/1.7 system-ui;margin:0}}main{{max-width:1260px;margin:auto;padding:32px 24px}}h1{{font-size:30px}}h2{{margin-top:34px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}a{{color:#23685c}}.note{{background:#e9efeb;padding:18px}}</style><main>
<h1>回到原始想法：预测波动率高于IV，买跨式并做Delta对冲</h1>
<p class="note">本轮使用最初六只股票GARCH策略的全部22笔已成交做多交易，覆盖2022–2025年的原始观察期及7／30／90天目标期限。每笔买入一张call和一张put，合约、入场及原实际退出固定；沿用原资金约束，逐场景检查退出是否变化。全部使用真实历史1分钟股票及期权报价，无插值、无新订阅。</p>
<h2>先看结果：同一套分钟数据、均滞后1分钟</h2>{table(main)}
<p>金额是独立出资交易损益的加总，初始名义本金随股票、时期不同；不是共同资金组合收益率。winning_trades为扣除执行成本后的盈利笔数；winning_trades_no_execution_cost为不扣执行成本的盈利笔数。残余Delta是模型估计值，在同一1分钟观察网格上逐笔求平均，再等权平均，并非真实连续时间风险。</p>
<img src="long_minute_hedging.png">
<h2>原信号和比较口径</h2><p>原入场信号：预测到合约到期的年化波动率，大于09:45时call与put的ask-IV较大值加3个百分点，并通过原每60分钟对冲假设下的粗略成本门槛。10:00入场；模型和交易名单没有根据这次盈亏重新选择。这里没有在高频方案下重新计算成本门槛，所以是固定交易的对冲诊断，而不是声称22笔都会通过每分钟交易的事前可盈利性筛选。</p>
<p>每60／15／5／1分钟对冲与信号延迟独立设置。1分钟延迟只使用t−1分钟的市场IV及美式Delta；15分钟延迟只使用t−15分钟信号。当前及信号时刻的股票报价必须通过过滤，持仓按整股取整并受显示数量限制。没有改变头寸时不虚构订单或佣金。仅NYSE正常交易时段可调仓，隔夜跳空保留。1分钟观察与滞后不等于数学连续对冲。</p>
<h2>原始15分钟基线与全部延迟方案</h2>{table(display)}
<p>archived_15m重放原15分钟资料、60分钟对冲、15分钟信号延迟，并核对原来的现金损益。resampled_15m从新分钟数据取原15分钟时刻，重算美式IV和Delta，检查数据及Greek变化。one_minute使用全部真实分钟资料。因此将旧基线直接与1分钟／1分钟方案比较，同时改变了观察网格、对冲频率和延迟；同一网格、同一延迟的行才能单独衡量频率影响。分钟借股计息积分也比原资料更密。</p>
<h2>执行成本具体是什么</h2>{table(costrows[costcols])}
<p>observed_costs：期权买ask、卖bid，每张每次假设$1佣金；股票买ask、卖bid，额外1bp滑点，佣金每股$0.0035且每单至少$0.35，股票卖出另计0.1bp。历史价差来自报价，佣金、附加费和滑点是原研究假设，并非已核实的用户实际券商费率。</p>
<p>no_execution_cost：对相同订单取mid成交，佣金、卖出附加费、额外滑点均为零。两组均保留融资、借股和同本金闲置现金利息基准；借股按年化2%，借款按滞后SOFR＋1.5%，现金收益按max(SOFR−0.5%,0)。零bid的买入期权不假装按mid出售，仍采用原保守零回收／不行权处理，其writeoff单列。价差、滑点只通过成交现金扣一次；成本列是归因，不重复扣款。两组现金余额不同会带来小幅融资差异，因此净损益差不一定恰好等于execution_cost。</p>
<h2>到底押对几次波动</h2><p>以原入场到原退出的15分钟及隔夜RV大于实际入场ask-IV较大值作为原口径，押对{original_hits}/22笔。若改用1分钟RV统计，同一批持仓为{minute_hits}/22笔。频率改变了RV测量，报价微观结构也会影响细网格RV，所以不应把测量变化当成模型或对冲使预测更准；这些命中数字也不是Gamma加权实现方差，更不等于盈利笔数。</p>{table(bet)}
<h2>每笔结果：1分钟信号延迟</h2>{table(trades)}
<h2>数据与局限</h2>{table(qtable)}
<p>这是已经研究过的22笔交易的条件诊断，原观察日每年只取1月／7月，并非每日扫描或新样本检验。缺失期权Greek沿用原规则：优先同一时点另一腿新鲜IV，否则最多沿用过去4天IV；不得使用未来报价。快照无法证明底层tick更新时刻，也未模拟排队、真实券商成交、离散现金股息下的精确美式行权或隔夜调仓。正Gamma对冲收益仍须覆盖时间价值损耗、IV重估及成本，不能靠预测IV差或提高频率保证盈利。</p>
<h2>复现与明细</h2><p><a href="frequency_comparison.csv">频率汇总</a> · <a href="trade_comparison.csv">逐笔比较</a> · <a href="execution_cost_breakdown.csv">成本拆分</a> · <a href="vol_bets.csv">波动命中明细</a> · <a href="by_asset.csv">按股票</a> · <a href="by_origin.csv">按时期</a> · <a href="by_target_dte.csv">按期限</a> · <a href="all_results.csv">全部场景</a> · <a href="quote_quality.json">报价质量</a> · <a href="run_configuration.json">参数和输入哈希</a> · <a href="verification.json">核对结果</a> · <a href="../../long_continuous_plan.json">实验方案</a></p>
<p>每笔分钟priced数据、逐笔订单与现金账在cases目录，数据源请求和原始文件哈希已保存。复现：<code>.venv/bin/python -m volatility_lab.long_continuous_study --workers 3</code>；审计：<code>.venv/bin/python -m volatility_lab.long_continuous_validate</code>。</p></main></html>'''
    (out / 'report.html').write_text(body)
    print(main.to_string(index=False), flush=True)


if __name__ == '__main__': make_report(ROOT / 'artifacts/long_continuous_study')
