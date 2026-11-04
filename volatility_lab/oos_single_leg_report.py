"""Compare all five models on the locked, common 2026 opportunity set."""
import html
import json
import numpy as np
import pandas as pd
from .oos_single_leg import OUT, ROOT, origins

LABELS = {'garch_t':'GARCH-t','ewma':'EWMA','rolling21':'Rolling 21d',
          'har_proxy_ridge':'HAR proxy / Ridge','ml_hist_gradient_boosting':'ML / Gradient Boosting'}


def table(frame, digits=3):
    return frame.to_html(index=False, border=0, float_format=lambda v:f'{v:,.{digits}f}', na_rep='—')


def make_report():
    plan = json.loads((ROOT/'oos_single_leg_plan.json').read_text())
    d = pd.read_csv(OUT/'decisions.csv'); trades = pd.read_csv(OUT/'unique_trade_results.csv')
    assert len(d)==2100 and not d.duplicated(['opportunity_id','model']).any()
    linked = d[d.status=='selected'].merge(trades,on='case_id',how='left',suffixes=('_decision',''),validate='many_to_many')
    assert linked.status.notna().all() and len(linked)==4*int((d.status=='selected').sum())
    linked.to_csv(OUT/'model_trade_results.csv',index=False)
    signals=d.groupby('model').agg(opportunities=('opportunity_id','size'),
        mean_forecast_vol=('forecast_vol','mean'),mean_decision_ask_iv=('decision_ask_iv','mean'),
        mean_vol_gap=('vol_gap','mean')).reset_index()
    for name in ['mean_forecast_vol','mean_decision_ask_iv','mean_vol_gap']:signals[name]*=100
    signals['signals']=signals.model.map(d[d.status=='selected'].groupby('model').size()).fillna(0).astype(int)
    signals.to_csv(OUT/'signal_summary.csv',index=False)
    sets={m:set(d[(d.model==m)&(d.status=='selected')].opportunity_id) for m in plan['models']}
    pd.DataFrame({a:{b:len(sets[a]&sets[b]) for b in sets} for a in sets}).to_csv(OUT/'signal_overlap.csv')
    summaries = []; monthly = []; stocks = []; tenors = []; kinds = []; costs = []
    for name in plan['models']:
        opportunities = d[d.model==name]; selected = opportunities[opportunities.status=='selected']
        for protocol in [p['id'] for p in plan['protocols']]:
            actual = linked[(linked.model==name)&(linked.protocol_id==protocol)&(linked.cost_scenario=='observed_costs')]
            ideal = linked[(linked.model==name)&(linked.protocol_id==protocol)&(linked.cost_scenario=='no_execution_cost')]
            completed = actual[actual.status=='completed']; free = ideal[ideal.status=='completed']
            bets = completed[completed.vol_bet_status=='scorable']
            row = {'model':name,'protocol_id':protocol,'opportunities':len(opportunities),
                'signals':len(selected),'completed':len(completed),'entry_rejected':len(actual)-len(completed),
                'calls':int((completed.kind=='call').sum()),'puts':int((completed.kind=='put').sum()),
                'before_execution_pnl':float(free.net_pnl.sum()),'after_execution_pnl':float(completed.net_pnl.sum()),
                'market_pnl':float(completed.market_pnl.sum()),
                'execution_cost':float(completed.execution_cost.sum()),'net_financing':float(completed.net_financing.sum()),
                'before_wins':int((free.net_pnl>0).sum()),'after_wins':int((completed.net_pnl>0).sum()),
                'vol_bets_scorable':len(bets),'vol_bets_correct_15m':int(bets.vol_bet_correct_15m.sum()),
                'vol_bet_hit_rate_15m':float(bets.vol_bet_correct_15m.mean()) if len(bets) else None,
                'mean_return_on_capital_before':float(free.return_on_capital.mean()) if len(free) else None,
                'mean_return_on_capital_after':float(completed.return_on_capital.mean()) if len(completed) else None,
                'all_opportunity_mean_roc_after':float(completed.return_on_capital.sum()/len(opportunities)),
                'stock_orders':int(completed.stock_orders.sum()),
                'mean_abs_delta_shares':float(completed.mean_abs_delta_shares.mean()) if len(completed) else None,
                'mean_missing_delta_fraction':float(completed.missing_delta_fraction.mean()) if len(completed) else None,
                'off_schedule_exits':int((~completed.exit_at_planned_time.astype(bool)).sum())}
            summaries.append(row)
            components = ['option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees','abandonment_writeoff','net_financing']
            cost_row={'model':name,'protocol_id':protocol,**{c:float(completed[c].sum()) for c in components}}
            cost_row['option_executed_spread_cost']=cost_row.pop('option_spread_cost')-cost_row['abandonment_writeoff']
            cost_row['total_execution_cost']=float(completed.execution_cost.sum())
            costs.append(cost_row)
            for column, target in [('origin',monthly),('asset',stocks),('target_dte',tenors)]:
                for value, eligible in opportunities.groupby(column):
                    a = completed[completed[column]==value]; b = free[free[column]==value]
                    target.append({'model':name,'protocol_id':protocol,column:value,'opportunities':len(eligible),
                        'trades':len(a),'before_execution_pnl':float(b.net_pnl.sum()),'after_execution_pnl':float(a.net_pnl.sum()),
                        'vol_bets_correct_15m':int(a.vol_bet_correct_15m.fillna(False).sum())})
            for kind, a in completed.groupby('kind'):
                b=free[free.kind==kind]
                kinds.append({'model':name,'protocol_id':protocol,'kind':kind,'trades':len(a),
                    'before_execution_pnl':float(b.net_pnl.sum()),'after_execution_pnl':float(a.net_pnl.sum())})
    summary = pd.DataFrame(summaries)
    quality=pd.DataFrame([{'case_id':p.parent.name,**json.loads(p.read_text())} for p in (OUT/'trades').glob('*/quality.json')])
    quality.to_csv(OUT/'data_quality.csv',index=False)
    exceptions=trades[(trades.cost_scenario=='observed_costs')&(~trades.exit_at_planned_time.astype(bool))]
    exceptions.to_csv(OUT/'exit_exceptions.csv',index=False)
    for filename, rows in [('summary',summary),('monthly_breakdown',monthly),('stock_breakdown',stocks),
                           ('tenor_breakdown',tenors),('kind_breakdown',kinds),('cost_breakdown',costs)]:
        pd.DataFrame(rows).to_csv(OUT/(filename+'.csv'),index=False)
    predictions = pd.read_csv(OUT/'forecast_predictions.csv')
    assert len(predictions)==2100
    def scores(frame):
        return pd.Series({'observations':len(frame),'qlike':frame.qlike.mean(),
            'mae_vol_points':100*frame.abs_vol_error.mean(),
            'rmse_vol_points':100*np.sqrt(frame.squared_vol_error.mean()),
            'mean_prediction_pct':100*frame.annualized_vol.mean(),
            'mean_realized_pct':100*frame.realized_daily_proxy_vol.mean()})
    score = predictions.groupby('model').apply(scores,include_groups=False).reset_index()
    score.to_csv(OUT/'forecast_scores.csv',index=False)
    by_horizon = predictions.groupby(['model','horizon_sessions']).apply(scores,include_groups=False).reset_index()
    by_horizon.to_csv(OUT/'forecast_scores_by_horizon.csv',index=False)
    by_month = predictions.groupby(['model','origin']).apply(scores,include_groups=False).reset_index()
    by_month.to_csv(OUT/'forecast_scores_by_month.csv',index=False)
    # Paired model forecast comparison, with each monthly origin kept intact.
    pivot = predictions.pivot(index=['asset','origin','horizon_sessions'],columns='model',values='qlike')
    baseline = pivot.garch_t
    differences = []
    for name in plan['models']:
        diff = pivot[name]-baseline
        grouped = diff.groupby(level='origin').mean()
        differences.append({'model':name,'mean_qlike_difference_vs_garch':float(diff.mean()),
            'months_better_than_garch':int((grouped<0).sum()),'months':len(grouped)})
    pd.DataFrame(differences).to_csv(OUT/'paired_forecast_comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    order = plan['models']; x = np.arange(len(order)); fig,ax = plt.subplots(1,2,figsize=(12,4.7))
    ax[0].bar(x,score.set_index('model').loc[order,'qlike'],color='#39879d')
    ax[0].set_title('2026 holdout forecast loss (lower is better)');ax[0].set_ylabel('Mean QLIKE')
    p = summary[summary.protocol_id=='daily_close'].set_index('model').loc[order]
    ax[1].bar(x-.18,p.before_execution_pnl/1000,width=.36,label='Before execution costs',color='#39879d')
    ax[1].bar(x+.18,p.after_execution_pnl/1000,width=.36,label='After execution costs',color='#dc8856')
    ax[1].axhline(0,color='black',lw=.7);ax[1].set_ylabel('One-contract cohort P&L, $ thousands')
    ax[1].set_title('Daily close hedge; different entry sets');ax[1].legend(fontsize=8)
    for a in ax:a.set_xticks(x,[LABELS[n] for n in order],rotation=25,ha='right')
    fig.tight_layout();fig.savefig(OUT/'model_comparison.png');plt.close(fig)
    display = summary.copy(); display.model=display.model.map(LABELS)
    display['vol_bet_hit_pct']=100*display.vol_bet_hit_rate_15m
    columns = ['model','signals','completed','before_execution_pnl','after_execution_pnl','before_wins','after_wins','vol_bets_correct_15m','vol_bets_scorable','vol_bet_hit_pct']
    daily = display[display.protocol_id=='daily_close'][columns]; hourly=display[display.protocol_id=='elapsed_60m'][columns]
    preview_scores=score.copy();preview_scores.model=preview_scores.model.map(LABELS)
    text = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>五模型单腿期权：2026 样本外比较</title>
    <style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1250px;margin:38px auto;padding:0 24px;color:#203040}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto;margin:20px 0}}td,th{{padding:8px 12px;border-bottom:1px solid #dde4e7;text-align:right;white-space:nowrap}}th{{background:#edf4f5}}img{{width:100%}}.note{{background:#edf4f5;padding:16px;border-left:4px solid #39879d}}code{{background:#eee;padding:2px 5px}}</style>
    <h1>五模型单腿期权：2026 年样本外比较</h1>
    <p>20 只股票 × 2026 年 1–7 月每月首个交易日 × 7/30/90 天目标期限 = 每个模型 420 个机会。保留原来单腿买入、Delta 对冲的策略。所有金额为每笔一张标准 100 股乘数合约、独立资金账户的损益加总。</p>
    <div class="note"><b>样本划分：</b>2022–2025 年已经用于探索规则，不能重新称作未见过的测试集。2026 年本轮期权链、损益此前未回测；规则于 {html.escape(plan['protocol_locked_utc'])} 锁定后开始获取。缓存日线已覆盖 2026 年，因此这是历史时间留出测试，不是前瞻实盘，也不宣称未来价格从未被任何人见过。</div>
    <h2>训练与防止未来信息</h2><p>GARCH-t、EWMA(λ=0.94)、21 日滚动方差、HAR proxy/Ridge、梯度提升树全部报告。每月重新拟合；日线特征截止前一交易日收盘。GARCH/EWMA 使用过去 756 个收益，Rolling21 使用最后 21 个收益；两个监督模型在 20 股上合并训练，以过去 756 个锚点日为窗口。标签为未来 1、2–5、6–21、22–63、64–84 交易日区块的平方日收益，只有区块结束不晚于拟合截止时刻的样本能进入训练。测试期早先月份已经完成的标签可用于后来月份训练，这属于滚动样本外评估。超参数、3 个百分点门槛、候选选择和对冲时点不随测试成绩调整。</p>
    <p>所有模型在同一组股票/月/期限机会决策，预测波动率精确积分至实际到期日。每个机会分别选最近有效 Call、Put，然后比较预测减各自 ask-IV，最多买一张。纽约 09:45 决策、10:00 入场，计划于到期前一个交易日的收盘前 15 分钟退出。相同机会中预测值相同地加在两张候选的比较式中，因此胜出合约由较低的 ask-IV 决定；模型差别主要在是否超过入场门槛。各模型选中同一合约时使用同一本逐分钟市场行情与对冲账本，避免重复计算改变结果。日线方差在收盘间隔内按自然时间分摊，尚未估计日内波动季节性。</p>
    <h2>预测评分：所有机会，不以发生交易为条件</h2><p>每个模型 420 条预测评分：20 股 × 7 月 × 5/21/63 交易日。QLIKE 使用实际方差与预测方差之比 r，损失为 r−log(r)−1，越低越好。这里的实际方差是平方日收益代理，不是交易诊断中的 15 分钟 RV。MAE 单位为波动率百分点。</p>
    {table(preview_scores.sort_values('qlike'))}<img src="model_comparison.png" alt="Out-of-sample forecast and trading comparison">
    <h2>全部机会的数据覆盖与信号</h2>{table(pd.crosstab(d.model,d.status).reset_index(),0)}
    <p>下表的预测、ask-IV 与差值均为百分数／绝对百分点；平均范围是全部可评分机会，signals 是达到 3 个百分点入场门槛的次数。</p>{table(signals,2)}
    <h2>主结果：每日收盘前 15 分钟对冲</h2>{table(daily,2)}
    <h2>对照：每 60 分钟对冲</h2>{table(hourly,2)}
    <p>before_execution_pnl 是相同实际成交时间、数量和退出的中间价账本，去掉执行价差、佣金、卖出附加费和滑点；仍保留融资、股票借券、闲置资金利息基准，以及无人报价期权放弃的损失。因此它并非所有经济成本为零。after_execution_pnl 使用真实历史 bid/ask 和下列固定费率。before_wins/after_wins 是盈利交易次数；vol_bets_correct_15m 是持有期 15 分钟加隔夜 RV 高于实际买入时 ask-IV 的次数，两者不会相同。</p>
    <h2>交易成本如何计算</h2><p>买期权付 ask，卖回收 bid；每张每次期权佣金 $1。股票买 ask、卖 bid，额外按成交价收 1bp 滑点；佣金每股 $0.0035、每单至少 $0.35，卖出另收成交额 0.1bp。现金贷出 SOFR−0.5%、借入 SOFR+1.5%，做空股票借券年费 2%；SOFR 使用滞后可知值。整股对冲，信号至少延迟一分钟，只在正常交易时段成交，遵守报价数量。初始对冲和退出清仓不受定期对冲间隔限制。</p>
    {table(pd.DataFrame(costs),2)}
    <h2>每月结果：查看是否集中在单一行情</h2>{table(pd.DataFrame(monthly),2)}
    <h2>期限拆分</h2>{table(pd.DataFrame(tenors),2)}
    <h2>资金归一与残余风险</h2>{table(display[['model','protocol_id','all_opportunity_mean_roc_after','mean_return_on_capital_before','mean_return_on_capital_after','mean_abs_delta_shares','mean_missing_delta_fraction','off_schedule_exits']],5)}
    <p>return_on_capital 列为小数收益率，资金基数是入场时 100 股名义金额的三倍；all_opportunity_mean_roc_after 将未交易机会计为零，等权平均全部 420 个机会，并非组合回报或年化。mean_abs_delta_shares 为持仓期间已知 Delta 观测的逐笔平均绝对残余股数；缺失观测不参与该均值，missing_delta_fraction 另行披露。</p>
    <p>有 {exceptions.case_id.nunique()} 个不同合约机会发生非计划退出，见下表及 <a href="exit_exceptions.csv">exit_exceptions.csv</a>。INTC 的 2026-02-02、90 天目标 Call 在 5 月 8 日触发既定研究资金储备规则，早于原计划的 5 月 14 日；这不是模拟经纪商保证金要求。EWMA、Rolling21、HAR 选中了该机会，GARCH 和梯度提升树未选中。每日和每小时对冲的触发相差一分钟，各自的中间价对照都冻结至实际退出。</p>{table(exceptions[['case_id','protocol_id','exit','exit_reason','net_pnl']],2)}
    <h2>预测期限拆分</h2>{table(by_horizon,3)}
    <h2>预测相对 GARCH 的逐月稳定性</h2>{table(pd.DataFrame(differences),4)}
    <h2>限制与可复查文件</h2><p>七个月的入场批次不足以证明长期收益；90 天交易互有重叠，股票也相关，不能把 420 个机会视为 420 次独立实验。当前 20 股名单沿用此前选择，存在股票池选择与存活偏差。独立账户的金额加总不是可直接投资组合收益，没有跨交易对冲净额；日频对冲可能留下较大方向风险。期权使用美式近似定价和市场 IV Delta；缺失 IV 允许仅向前携带最多四天，缺失 Delta 的比例与非计划退出单独披露；没有模拟实际提前行权。股票拆股跨期且未支持调整交割的机会作为数据覆盖排除披露。历史快照无法核实底层逐笔更新是否陈旧，报价并非实际成交保证。共检查 {len(quality)} 条不同合约路径，股票估值修复 {int(quality.stock_mark_repairs.sum())} 行；逐分钟质量详见 <a href="data_quality.csv">data_quality.csv</a>。</p>
    <p>详见 <a href="summary.csv">summary.csv</a>、<a href="model_trade_results.csv">逐笔五模型结果</a>、<a href="decisions.csv">全部机会与信号</a>、<a href="stock_breakdown.csv">逐股结果</a>、<a href="forecast_predictions.csv">全部预测评分</a>、<a href="configuration.json">规则及代码哈希</a>、<a href="verification.json">审计结果</a>。成交与逐分钟现金路径保存在 trades/ 下。报告不以测试结果挑选新的参数或宣布长期优胜者。</p></html>'''
    (OUT/'report.html').write_text(text)
    print(score.sort_values('qlike').to_string(index=False))
    print(summary[['model','protocol_id','signals','completed','before_execution_pnl','after_execution_pnl','vol_bets_correct_15m']].to_string(index=False))


if __name__=='__main__': make_report()
