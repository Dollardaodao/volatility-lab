"""Presentation only for the frozen monthly ML forecast experiment."""
import json
import os
import numpy as np
import pandas as pd
from .theta_sdk import ROOT

LABELS={'garch_t':'GARCH-t','ewma':'EWMA','rolling21':'Rolling 21d',
        'har_proxy_ridge':'HAR-proxy Ridge','ml_hist_gradient_boosting':'ML Gradient Boosting','always_long':'Always long'}


def make_report(out,plan,complete=True):
    predictions=pd.read_csv(out/'forecast_predictions.csv')
    keys=['asset','origin','horizon_sessions']
    coverage=predictions.groupby(keys).model.nunique()
    eligible=coverage[coverage==len(plan['models'])].index
    common=predictions[pd.MultiIndex.from_frame(predictions[keys]).isin(eligible)].copy()
    common.to_csv(out/'paired_forecast_predictions.csv',index=False)
    def forecast_stats(f):
        return pd.Series({'observations':len(f),'qlike':f.qlike.mean(),
            'mae_vol_points':f.abs_vol_error.mean()*100,'rmse_vol_points':np.sqrt(f.squared_vol_error.mean())*100,
            'mean_prediction_pct':f.annualized_vol.mean()*100,'mean_realized_pct':f.realized_daily_proxy_vol.mean()*100,
            'bias_vol_points':(f.annualized_vol-f.realized_daily_proxy_vol).mean()*100})
    scores=common.groupby('model').apply(forecast_stats,include_groups=False).reset_index()
    scores.to_csv(out/'forecast_scores.csv',index=False)
    by_horizon=common.groupby(['model','horizon_sessions']).apply(forecast_stats,include_groups=False).reset_index()
    by_horizon.to_csv(out/'forecast_scores_by_horizon.csv',index=False)
    common['year']=pd.to_datetime(common.origin).dt.year
    by_year=common.groupby(['model','year']).apply(forecast_stats,include_groups=False).reset_index()
    by_year.to_csv(out/'forecast_scores_by_year.csv',index=False)
    new=pd.read_csv(out/'new_model_trades.csv',low_memory=False)
    base=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/'all_results.csv',low_memory=False)
    base=base[(base.hedge_minutes==60)&base.policy.isin(['garch_long_flat','ewma_long_flat','always_long'])&
              base.cost_scenario.isin(['base','zero_bid_writedown'])].copy()
    base['model']=base.policy.map({'garch_long_flat':'garch_t','ewma_long_flat':'ewma','always_long':'always_long'})
    all_trades=pd.concat([base,new],ignore_index=True,sort=False)
    all_trades.to_csv(out/'all_trade_scenarios.csv',index=False)
    completed=all_trades[all_trades.status=='completed'].copy()
    trade_keys=['asset','origin','target_dte','cost_scenario']
    nmodels=5
    counts=completed.groupby(trade_keys).model.nunique();joint=counts[counts==nmodels].index
    paired=completed[pd.MultiIndex.from_frame(completed[trade_keys]).isin(joint)].copy()
    paired.to_csv(out/'paired_trade_scenarios.csv',index=False)
    rows=[]
    for (scenario,model),f in paired.groupby(['cost_scenario','model']):
        active=f[f.side!=0].copy()
        market=active.gross_option_pnl+active.hedge_gross_pnl+active.dividends
        tx=active[['option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees']].sum(axis=1)
        finance=active.cash_interest-active.benchmark_interest-active.borrow_cost
        if len(active) and (active.net_pnl-(market-tx+finance)).abs().max()>1e-5:
            raise AssertionError('ML trade accounting failed')
        rows.append({'cost_scenario':scenario,'model':model,'cohorts':len(f),'trades':len(active),
            'net_wins':int((active.net_pnl>0).sum()),'market_wins':int((market>0).sum()),
            'market_pnl_usd':market.sum(),'trading_cost_usd':tx.sum(),'net_financing_usd':finance.sum(),
            'net_pnl_usd':f.net_pnl.sum(),'mean_candidate_return_pct':f.return_on_notional.mean()*100,
            'mean_active_return_pct':active.return_on_notional.mean()*100 if len(active) else np.nan,
            'holding_vol_bet_correct':int(active.vol_bet_correct.fillna(False).astype(bool).sum()) if model in ('har_proxy_ridge','ml_hist_gradient_boosting') else np.nan})
    summary=pd.DataFrame(rows);summary.to_csv(out/'trade_summary.csv',index=False)
    zero=summary[summary.cost_scenario=='zero_bid_writedown'].copy()
    # A flat, cash-benchmark-relative baseline is exactly zero by construction.
    joint_n=zero.cohorts.max() if len(zero) else 0
    display=zero.drop(columns='cost_scenario').copy()
    display=pd.concat([display,pd.DataFrame([{'model':'No trade','cohorts':joint_n,'trades':0,'net_wins':0,
        'market_pnl_usd':0.,'trading_cost_usd':0.,'net_financing_usd':0.,'net_pnl_usd':0.,'mean_candidate_return_pct':0.}])],ignore_index=True)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6'})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for name,f in by_horizon.groupby('model'):
        axes[0].plot(f.horizon_sessions,f.qlike,marker='o',label=LABELS[name])
    axes[0].set(xlabel='Forecast horizon (sessions)',ylabel='Mean QLIKE · lower is better',title='Monthly walk-forward · daily variance proxy')
    axes[0].legend(fontsize=8)
    if len(zero):
        names=[n for n in ('garch_t','ewma','har_proxy_ridge','ml_hist_gradient_boosting','always_long') if n in set(zero.model)]
        z=zero.set_index('model').loc[names]
        axes[1].bar(range(len(z)),z.mean_candidate_return_pct,color=['#367b68' if x>=0 else '#bd554d' for x in z.mean_candidate_return_pct])
        axes[1].set_xticks(range(len(z)),[LABELS[x] for x in z.index],rotation=22,ha='right',fontsize=9)
        for i,r in enumerate(z.itertuples()):axes[1].annotate(f'{int(r.trades)} entries',(i,r.mean_candidate_return_pct),xytext=(0,-14 if r.mean_candidate_return_pct<0 else 6),textcoords='offset points',ha='center',fontsize=8)
        axes[1].axhline(0,color='grey',lw=.7)
        axes[1].set_ylim(min(-.05,float(z.mean_candidate_return_pct.min())*1.3),max(.05,float(z.mean_candidate_return_pct.max())*1.3))
    axes[1].set(ylabel='Mean P&L / entry underlying notional (%)',title='Same option candidates · 60-minute hedges')
    fig.savefig(out/'model_comparison.png',dpi=180);plt.close(fig)
    table=lambda f:f.to_html(index=False,border=0,float_format=lambda x:f'{x:,.4f}')
    valid_new=new[new.status=='completed']
    max_residual=float(valid_new.reconciliation_error.abs().max()) if len(valid_new) else np.nan
    failures=json.loads((out/'forecast_failures.json').read_text())
    months=sorted(predictions.origin.unique())
    winner=scores.sort_values('qlike').iloc[0] if len(scores) else None
    headline=f'共同样本内平均QLIKE最低的是 {LABELS[winner.model]}（{winner.qlike:.4f}）。这仅表示本轮日线方差代理预测误差较低，不代表期权策略盈利。' if winner is not None else '预测比较尚未形成共同样本。'
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ML波动率预测与期权交易比较</title>
<style>body{{background:#faf9f6;color:#263238;font:16px/1.7 system-ui;margin:0}}main{{max-width:1240px;margin:auto;padding:35px 25px}}h1{{font-size:32px}}h2{{margin-top:36px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}a{{color:#23685c}}.note{{background:#e9efeb;padding:18px}}</style><main>
<h1>用机器学习预测波动率，是否比GARCH更好？</h1>
<p class="note">{'完整研究' if complete else '运行中的部分研究'}：20只股票，{len(months)}个月度预测起点，2022–2025年；到期标签限定在2025年底之前。{headline}</p>
<h2>这次实际比较什么</h2><p>GARCH-t、EWMA、21日滚动方差作为基准；新增HAR风格Ridge回归及梯度提升树。HAR代理模型使用过去1、5、22天的平方收益均值；树模型另加入63日波动、方向收益、下行波动、高低价范围、隔夜跳空和相对成交量。模型跨固定20只股票合并训练，并包含股票身份和预测区间特征。模型参数在查看新增预测及交易结果前写入<a href="../../ml_vol_plan.json">实验计划</a>，没有做收益或超参数搜索。</p>
<p><strong>数据限制：</strong>训练目标是未来日线收盘到收盘收益平方的区间均值，不是用完整高频数据估计的每日RV，因此这里的HAR是HAR风格代理模型，不能冒称原始HAR-RV。原期权回测的15分钟及隔夜RV另用于持仓检验。日线预测准确性与可执行的期权收益分开报告。</p>
<h2>怎样避免偷看未来</h2><p>所有输入最晚取入场日前一实际交易日收盘；每月首个交易日09:45之前重新训练。训练锚点最多向前756个交易日，每个未来方差区间必须已经完整结束，才允许进入训练。例如预测63日方差不能把尚未走完的63日标签放进训练集。没有随机打乱时间切分，也没有随机早停验证。Ridge的标准化和对数转均值修正均只在训练数据上估计。</p>
<p>每个拟合审计记录每只股票的最后特征时间和最后已完成标签时间，保存在months目录。日线平方收益按未来第1天、第2–5天、第6–21天、第22–63天、第64–84天分区预测，各区间输出正方差，再按每张期权的实际剩余时间汇总。首次部分交易日仍沿用原GARCH的自然时间均匀分配近似，不能当作精确日内波动季节性模型。</p>
<h2>预测结果：所有模型使用共同标签</h2>{table(scores)}<p>共同“股票×月份×期限”标签 {len(eligible)} 个，每个模型均在同一集合上评分；总预测行 {len(predictions)}，基准拟合失败 {len(failures)} 次，全部在<a href="forecast_failures.json">失败清单</a>公开，没有用其他模型悄悄替代。QLIKE衡量方差预测误差，MAE/RMSE以波动率百分点计，越低越好。mean_prediction_pct及mean_realized_pct单位为%。</p>
<img src="model_comparison.png"><h2>按预测期限</h2>{table(by_horizon)}<h2>按年份</h2>{table(by_year)}
<h2>期权回放：模型预测改善是否能变成钱</h2>{table(display)}
<p>该表是完整共同候选的zero_bid_writedown敏感性口径：有明确零bid的多头腿按零回收，并承诺不行权，不能当作mid成交。原计划480候选，原20股票已有478个可完成共同比较；新增模型若有额外失败也会进入清单，不会补造结果。每笔一张call和一张put，60分钟对冲，预测高于两腿ask-IV较高者3个波动率百分点且通过同一个成本门槛才入场。新模型只替换预测及对应公允价值，入场、选约、Greek、对冲、费用、融资及退出规则全部沿用。GARCH和EWMA的实际到期预测逐组合与原结果核对。</p>
<p>cohorts包含空仓，trades是入场数；net_wins是扣成本盈利笔数。holding_vol_bet_correct仅对新模型统计：实际持仓期RV高于实际成交时两腿ask-IV较高者，是普通RV/IV检验，并非gamma加权收益的恒等式。该列空白不代表基准模型没有押对。mean_candidate_return_pct以入场100股标的名义金额归一化，包含空仓0；mean_active_return_pct仅实际入场，二者均非年化收益或共同资金组合收益。</p>
<h2>严格双边报价对照</h2>{table(summary[summary.cost_scenario=='base'])}
<p>严格口径要求真实可执行的双边平仓报价，未完成的组合不能按模型价格补齐。所有模型在同一完整候选集比较，因此严格口径和零bid敏感性口径分母可能不同，不能混合。新增模型最大现金核对误差为${max_residual:.8f}。</p>
<h2>结果的边界</h2><p>本研究是历史滚动预测比较。虽然每次拟合严格只用当时已完成的标签，但2022–2025年市场及原策略损益此前已被看过，因此不是全新的盲测样本。没有用本轮结果挑最赚钱的阈值或改手续费。20只股票是固定存续股票便利样本；月度63日标签互相重叠，期权仍只有原来的8个历史入场起点，股票和期限也相关。没有据此计算独立样本显著性、组合Sharpe或宣称可直接实盘盈利。更低预测误差不保证预测高于可成交IV时就存在足够覆盖成本的优势。</p>
<h2>逐笔数据与复现</h2><p><a href="forecast_predictions.csv">所有月度预测</a> · <a href="paired_forecast_predictions.csv">共同预测样本</a> · <a href="forecast_scores.csv">预测汇总</a> · <a href="all_trade_scenarios.csv">全部期权场景及失败原因</a> · <a href="paired_trade_scenarios.csv">共同完整期权样本</a> · <a href="trade_summary.csv">交易汇总</a> · <a href="run_configuration.json">参数、依赖和源数据哈希</a></p>
<p>复现：<code>.venv/bin/python -m volatility_lab.ml_study --workers 4</code>。逐月训练审计在months目录，拟合模型在models目录，新策略成交与现金账目在trades目录。原研究文件保留不变。</p>
<p>方法参考：<a href="https://doi.org/10.1093/jjfinec/nbp001">Corsi的HAR-RV原始研究</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html">梯度提升树及Poisson损失</a>。</p></main></html>'''
    (out/'report.html').write_text(text)
