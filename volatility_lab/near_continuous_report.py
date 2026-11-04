"""Presentation for actual-minute hedge timing and latency diagnostics."""
import json
import os
import numpy as np
import pandas as pd
from .theta_sdk import ROOT


def make_report(out,plan):
    frame=pd.read_csv(out/'all_results.csv',low_memory=False)
    completed=frame[frame.status=='completed'].copy()
    keys=['case_id','data_source','grid_minutes','hedge_minutes','signal_lag_minutes']
    summary=completed.pivot(index=keys,columns='cost_scenario',values='net_pnl').reset_index()
    observed=completed[completed.cost_scenario=='observed_costs'][keys+['stock_orders','stock_turnover_shares','execution_cost','market_pnl','exit_at_original_time']]
    summary=summary.merge(observed,on=keys,validate='one_to_one')
    delta_quality=[]
    for r in summary.itertuples():
        path=out/'cases'/r.case_id/f'{r.data_source}_{r.hedge_minutes}m_lag{r.signal_lag_minutes}m_no_execution_cost_ledger.csv'
        ledger=pd.read_csv(path)
        active=ledger[ledger.option_contracts_per_leg!=0]
        values=active.delta_shares.abs().dropna()
        delta_quality.append({**{k:getattr(r,k) for k in keys},
            'mean_abs_delta_shares':values.mean(),'p95_abs_delta_shares':values.quantile(.95),
            'max_abs_delta_shares':values.max(),'missing_delta_fraction':active.delta_shares.isna().mean()})
    delta_quality=pd.DataFrame(delta_quality);delta_quality.to_csv(out/'delta_quality.csv',index=False)
    summary=summary.merge(delta_quality,on=keys,validate='one_to_one');summary.to_csv(out/'summary.csv',index=False)
    quality=json.loads((out/'quote_quality.json').read_text())
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for ax,case_id in zip(axes,plan['cases']):
        f=summary[(summary.case_id==case_id)&(summary.data_source=='one_minute')]
        for lag,g in f.groupby('signal_lag_minutes'):
            g=g.sort_values('hedge_minutes')
            ax.plot(g.hedge_minutes,g.no_execution_cost,marker='o',label=f'No execution costs · lag {lag}m')
            ax.plot(g.hedge_minutes,g.observed_costs,marker='x',ls='--',label=f'Observed costs · lag {lag}m')
        ax.set_xscale('log');ax.set_xticks([1,5,15,60],[1,5,15,60]);ax.axhline(0,color='grey',lw=.7)
        ax.set(title=case_id,xlabel='Hedge interval (minutes)',ylabel='Final excess P&L (USD)')
        ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=6)
    fig.savefig(out/'minute_hedging_comparison.png',dpi=170);plt.close(fig)
    table=lambda f:f.to_html(index=False,border=0,float_format=lambda x:f'{x:,.3f}')
    sections=[]
    for case_id in plan['cases']:
        f=summary[summary.case_id==case_id]
        q=quality[case_id]
        quality_rows=pd.DataFrame([{'field':k,'value':json.dumps(v) if isinstance(v,dict) else v} for k,v in q.items()])
        sections.append(f'<h2>{case_id}</h2>{table(f)}<details><summary>真实分钟采样与旧报价核对</summary>{table(quality_rows)}</details>')
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>更接近连续对冲：真实一分钟报价</title>
<style>body{{background:#faf9f6;color:#263238;font:16px/1.7 system-ui;margin:0}}main{{max-width:1320px;margin:auto;padding:35px 25px}}h1{{font-size:32px}}h2{{margin-top:36px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}a{{color:#23685c}}.note{{background:#e9efeb;padding:18px}}</style><main>
<h1>更接近连续Delta对冲，会改变这三笔亏损吗？</h1>
<p class="note">这轮取得了真实1分钟股票和期权历史bid/ask，未用插值生成价格，也没有升级或购买订阅。固定原来的三笔空跨式、入场和实际退出，分别改变对冲频率与信号延迟；它是对既有亏损的条件诊断，不是新策略或全市场盈利检验。</p>
<h2>如何读结果</h2><p>archived_15m是原15分钟价格/Greek资料，60分钟对冲、15分钟信号滞后，用通用引擎重放并逐笔核对原结果。resampled_15m是新分钟报价只取原15分钟时刻、重新反演IV及Delta。one_minute使用全部真实分钟报价，分别测试60、15、5、1分钟对冲，并独立测试15分钟与1分钟信号延迟。</p>
<p>因此“每1分钟对冲、只滞后1分钟”与“每1分钟对冲、仍滞后15分钟”不是同一个实验。股票订单只使用严格在过去的Delta；不使用当前成交快照的Delta假设当场成交。仍采用整股取整、一张call加一张put，以及原初始本金和费用假设。没有股票持仓变化时不收取一笔虚构佣金。mean_abs_delta_shares及p95_abs_delta_shares按持仓期间观察点的残余Delta股数计算；比较同一1分钟网格时，能检查对冲误差是否下降，它们不是金额收益或完整时间积分风险。</p>
<p>no_execution_cost：成交取mid、免佣金/卖出附加费/额外滑点，但保留融资与借股；observed_costs：吃历史bid/ask，股票额外1bp滑点，沿用原费用假设。二者均减去同本金闲置现金利息。它们不是共同资金组合收益率。逐笔订单、资本退出和成本对照是否保持相同持仓均留在原始结果中。</p>
<img src="minute_hedging_comparison.png">
{''.join(sections)}
<h2>这依然不是数学意义的continuous hedging</h2><p>一分钟观察、一分钟前的信息、整股头寸和真实价差仍是离散交易；这里只在NYSE正常交易时段对冲，隔夜及开盘跳空并未被消除。原始tick的更新时间仍无法从这些重采样快照证明，分钟间可能重复同一报价。无效的股票快照不能生成对冲信号；无法使用的非正股票标记只允许沿用过去有效值，不允许未来插值。</p>
<p>连续对冲的简单无摩擦模型消除Delta方向暴露和离散调仓误差，但空头仍有负Gamma。其毛损益近似由正Gamma权重乘以“定价IV平方减实际方差”累积决定，并非对冲越频繁就保证盈利。实际波动高于所卖波动时，理想连续对冲也可能亏；提前平仓还包含IV重估风险。普通全窗口RV不是Gamma加权RV，不能单凭一个全期RV数字给出实际连续对冲的精确损益。</p>
<p>本实验固定原实际退出日期，所以没有让新分钟数据改变风险退出规则；也没有模拟期间真实美式指派。保留原研究资本约束，若exit_at_original_time为False则必须单独解释。更密集的路径还会改变借股积分和IV沿用过程，不能把所有差额都归为对冲频率。</p>
<h2>数据、引擎和复现</h2><p><a href="summary.csv">全部协议汇总</a> · <a href="all_results.csv">逐场景完整损益</a> · <a href="quote_quality.json">分钟数据及旧快照核对</a> · <a href="run_configuration.json">参数、依赖、原始报价哈希</a> · <a href="../../near_continuous_plan.json">固定实验方案</a></p>
<p>各案例分钟priced数据、逐笔订单和现金账目在cases目录；原始行情及请求参数保存在data/near_continuous/quotes。通用分钟引擎是原引擎的研究分支，原stock_replay.py和所有既有结果保留不变。复现：<code>.venv/bin/python -m volatility_lab.near_continuous_study --workers 3</code>。</p>
<p>理论参考：<a href="https://artursepp.com/2017/05/01/how-to-optimize-volatility-trading-and-delta-hedging-strategies-under-the-discrete-hedging-with-transaction-costs/">Sepp：离散对冲、波动率差与成本</a>。</p></main></html>'''
    (out/'report.html').write_text(text)
    print(summary.to_string(index=False),flush=True)
