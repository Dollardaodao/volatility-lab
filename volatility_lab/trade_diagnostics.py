"""Frozen-entry attribution of the original 22 long GARCH trades.

Accounting attribution is exact; forecast errors are diagnostics, not a causal
allocation of option P&L. Cost counterfactuals keep the observed quote eligibility,
past-information Greeks and original entry set. No signal is retuned here.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .stock_market import DailyInputs
from .stock_replay import StockCosts, replay
from .forecast import fit_variance, expiry_forecast

TX = ['option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees']
IDENT = ['asset','origin','target_dte']


def decompose(f):
    f=f.copy()
    f['market_pnl']=f.gross_option_pnl+f.hedge_gross_pnl+f.dividends
    f['transaction_cost']=f[TX].sum(axis=1)
    f['net_financing']=f.cash_interest-f.benchmark_interest-f.borrow_cost
    f['accounting_residual']=f.net_pnl-(f.market_pnl-f.transaction_cost+f.net_financing)
    if f.accounting_residual.abs().max()>1e-5:raise AssertionError('Attribution does not reconcile')
    return f


def main():
    old=ROOT/'artifacts'/'stock_study';out=ROOT/'artifacts'/'trade_diagnostics'
    out.mkdir(exist_ok=True)
    all_rows=pd.read_csv(old/'completed_results.csv')
    universe=all_rows[(all_rows.policy=='garch_long_flat')&(all_rows.hedge_minutes==60)&
                      (all_rows.cost_scenario=='zero_bid_writedown')].copy()
    original=decompose(universe[universe.side==1].copy())
    if len(universe)!=144 or len(original)!=22:raise AssertionError('Original study changed')
    universe['vol_gap']=universe.garch_t_forecast_vol-universe.long_execution_iv
    universe['passes_vol']=universe.vol_gap>.03
    universe['passes_cost']=universe.estimated_long_edge>0
    universe['filter_reason']=np.select([~universe.passes_vol,~universe.passes_cost],
                                        ['vol_gap_below_3pts','cost_hurdle'],default='entered')
    if not np.array_equal((universe.passes_vol&universe.passes_cost).to_numpy(),(universe.side==1).to_numpy()):
        raise AssertionError('Signal funnel inconsistent')
    universe.to_csv(out/'candidate_funnel_144.csv',index=False)
    universe.groupby(['origin','filter_reason']).size().unstack(fill_value=0).to_csv(out/'funnel_by_origin.csv')
    universe.groupby(['target_dte','filter_reason']).size().unstack(fill_value=0).to_csv(out/'funnel_by_tenor.csv')
    manifest=json.loads((ROOT/'artifacts'/'stock_study_acquisition.json').read_text())
    contracts={(c['asset'],c['origin'],c['target_dte']):c for c in manifest['cohorts'] if c['status']=='acquired'}
    daily={s:DailyInputs(ROOT/'data'/'stock_study',s) for s in original.asset.unique()}
    base=replace(StockCosts(),allow_unbid_long_abandonment=True)
    scenarios={
        'observed_costs_60m':(60,base),
        'no_commissions_60m':(60,replace(base,fee_multiplier=0)),
        'no_commissions_or_slippage_60m':(60,replace(base,fee_multiplier=0,extra_slippage_bps=0)),
        'remove_option_trading_costs_60m':(60,replace(base,option_spread_fraction=0,option_commission_per_contract=0)),
        'remove_stock_trading_costs_60m':(60,replace(base,stock_spread_fraction=0,stock_commission_per_share=0,
            minimum_stock_commission=0,stock_sell_fee_bps=0,extra_slippage_bps=0)),
        'midpoint_no_commissions_slippage_60m':(60,replace(base,option_spread_fraction=0,stock_spread_fraction=0,
            fee_multiplier=0,extra_slippage_bps=0)),
        'observed_costs_15m':(15,base),
        'observed_costs_240m':(240,base),
        'observed_costs_390m':(390,base),
    }
    config={'original_results_sha256':hashlib.sha256((old/'completed_results.csv').read_bytes()).hexdigest(),
            'replay_source_sha256':hashlib.sha256((ROOT/'volatility_lab'/'stock_replay.py').read_bytes()).hexdigest(),
            'diagnostics_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'frozen_entry_count':22,'scenarios':{k:{'frequency':v[0],'costs':v[1].__dict__} for k,v in scenarios.items()}}
    config_file=out/'run_configuration.json'
    if config_file.exists() and json.loads(config_file.read_text())!=config:
        raise ValueError('Diagnostics configuration changed; remove only diagnostics checkpoints before recomputing')
    config_file.write_text(json.dumps(config,indent=2))
    records=[];holding=[]
    for _,row in original.iterrows():
        key=tuple(row[x] for x in IDENT);contract=contracts[key]
        stem=f'{row.asset}_{row.origin}_{row.target_dte}d'
        panel=pd.read_csv(old/(stem+'_priced.csv'),index_col=0,parse_dates=True)
        panel.index=pd.to_datetime(panel.index,utc=True)
        checkpoint=out/(stem+'_counterfactuals.json')
        if checkpoint.exists():payload=json.loads(checkpoint.read_text())
        else:
            rows=[];original_orders=None
            for label,(frequency,costs) in scenarios.items():
                result,ledger,fills=replay(panel,contract,daily[row.asset],1,frequency,costs)
                if result['entry']!=row.entry or result['exit']!=row.exit:
                    raise AssertionError('Counterfactual changed holding period')
                orders=fills[['time','instrument','quantity']].astype(str).to_dict('records')
                if label=='observed_costs_60m':
                    if abs(result['net_pnl']-row.net_pnl)>1e-5:raise AssertionError('Original baseline not reproduced')
                    original_orders=orders
                elif frequency==60 and orders!=original_orders:
                    raise AssertionError('Cost counterfactual changed orders')
                rows.append({**{x:row[x] for x in IDENT},'scenario':label,'hedge_minutes':frequency,
                             **{k:v for k,v in result.items() if k!='cost_parameters'}})
            start=pd.Timestamp(row.entry);end=pd.Timestamp(row.exit)
            held=panel.loc[start:end]
            years=(end-start).total_seconds()/(365*86400)
            rv=np.sqrt(np.square(np.diff(np.log(held.spot.to_numpy()))).sum()/years)
            fit=fit_variance(daily[row.asset].prices,pd.Timestamp(contract['decision_time']),horizon=150,model='garch_t')
            decision_f=expiry_forecast(fit,pd.Timestamp(contract['decision_time']),end,'equity')['integrated_variance']
            entry_f=expiry_forecast(fit,start,end,'equity')['integrated_variance']
            payload={'rows':rows,'holding':{**{x:row[x] for x in IDENT},'holding_days':years*365,
                'realized_vol_entry_to_exit':rv,'frozen_forecast_vol_entry_to_exit':np.sqrt(entry_f/years),
                'forecast_variance_decision_to_exit':decision_f,
                'entry_ask_iv':float(max(held.iloc[0].call_iv,held.iloc[0].put_iv))}}
            # The extra field is midpoint IV at execution; entry signal ask-IV
            # is already present in original results and is not overwritten.
            payload['holding']['entry_mid_iv_max']=payload['holding'].pop('entry_ask_iv')
            checkpoint.write_text(json.dumps(payload,indent=2,default=str))
        records.extend(payload['rows']);holding.append(payload['holding'])
        print(f'Frozen trade diagnostics: {stem}',flush=True)
    original=original.merge(pd.DataFrame(holding),on=IDENT,validate='one_to_one')
    original['forecast_error_to_expiry']=original.garch_t_forecast_vol-original.realized_vol_to_expiry
    original['forecast_error_holding_window']=original.frozen_forecast_vol_entry_to_exit-original.realized_vol_entry_to_exit
    original['rv_minus_ask_iv']=original.realized_vol_to_expiry-original.long_execution_iv
    original.to_csv(out/'original_22_trade_attribution.csv',index=False)
    counter=decompose(pd.DataFrame(records));counter.to_csv(out/'fixed_22_counterfactuals.csv',index=False)
    summary=counter.groupby('scenario',sort=False).agg(trades=('net_pnl','size'),net_pnl=('net_pnl','sum'),
        market_pnl=('market_pnl','sum'),transaction_cost=('transaction_cost','sum'),net_financing=('net_financing','sum'),
        stock_fees=('stock_fees','sum'),stock_spread_cost=('stock_spread_cost','sum'),option_spread_cost=('option_spread_cost','sum'),
        slippage_cost=('slippage_cost','sum'),fills=('fills','sum'),wins=('net_pnl',lambda x:int((x>0).sum())),
        mean_drawdown=('max_drawdown_usd','mean')).reset_index()
    summary['improvement_vs_original']=summary.net_pnl-original.net_pnl.sum()
    summary.to_csv(out/'counterfactual_summary.csv',index=False)
    for col in ['asset','target_dte','origin']:
        original.groupby(col).agg(trades=('net_pnl','size'),market_pnl=('market_pnl','sum'),
            transaction_cost=('transaction_cost','sum'),net_financing=('net_financing','sum'),net_pnl=('net_pnl','sum')).to_csv(out/f'attribution_by_{col}.csv')
    scores=[]
    for label,f in [('all_144',universe),('selected_22',original)]:
        for model,col in [('garch','garch_t_forecast_vol'),('ewma','ewma_forecast_vol'),('ask_iv','long_execution_iv')]:
            actual=f.realized_vol_to_expiry;pred=f[col]
            ratio=actual.pow(2)/pred.pow(2)
            scores.append({'sample':label,'model':model,'n':len(f),'mean_forecast':pred.mean(),
                'mean_realized':actual.mean(),'bias_vol_points':(pred-actual).mean()*100,
                'mae_vol_points':(pred-actual).abs().mean()*100,'overpredictions':int((pred>actual).sum()),
                'mean_qlike':float((ratio-np.log(ratio)-1).mean())})
    pd.DataFrame(scores).to_csv(out/'forecast_diagnostics.csv',index=False)
    build_report(original,universe,counter,summary,pd.DataFrame(scores),out)
    print(summary[['scenario','net_pnl','market_pnl','transaction_cost','net_financing']].to_string(index=False),flush=True)


def build_report(trades,candidates,counter,summary,scores,out):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6'})
    t=trades
    labels=['Option marks','Hedge gains','Dividends','Option spread','Stock spread','Slippage','Option fees','Stock fees',
            'Net financing','Net P&L']
    vals=[t.gross_option_pnl.sum(),t.hedge_gross_pnl.sum(),t.dividends.sum(),
          *[-t[k].sum() for k in TX],t.net_financing.sum(),t.net_pnl.sum()]
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');running=0
    for i,v in enumerate(vals):
        bottom=running if i<len(vals)-1 else 0
        ax.bar(i,v,bottom=bottom,color='#367b68' if v>=0 else '#bd554d',width=.72)
        ax.text(i,bottom+v+(160 if v>=0 else -160),f'{v:+,.0f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
        if i<len(vals)-1:running+=v
    ax.set_ylim(min(np.cumsum(vals[:-1]))-700,500)
    ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right');ax.axhline(0,color='grey',lw=.7)
    ax.set(ylabel='USD, 22 independent one-straddle trades',title='Exact cash attribution · original 60-minute hedges')
    fig.savefig(out/'cash_waterfall.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for tenor,f in t.groupby('target_dte'):
        axes[0].scatter(f.garch_t_forecast_vol*100,f.realized_vol_to_expiry*100,label=f'{tenor}d',s=45)
    bound=max(t.garch_t_forecast_vol.max(),t.realized_vol_to_expiry.max())*110
    axes[0].plot([0,bound],[0,bound],color='grey',lw=.8);axes[0].legend()
    axes[0].set(xlabel='GARCH forecast to expiry (%)',ylabel='Observed RV to expiry (%)',title='Selected 22: forecast diagnostics')
    cf=summary[summary.scenario.str.match('observed_costs_')].copy()
    cf['frequency']=cf.scenario.str.extract(r'_(\d+)m')[0].astype(int);cf=cf.sort_values('frequency')
    axes[1].plot(cf.frequency,cf.market_pnl,marker='o',label='Market P&L before costs')
    axes[1].plot(cf.frequency,cf.net_pnl,marker='o',label='Net P&L')
    axes[1].plot(cf.frequency,-cf.transaction_cost,marker='o',label='Trading costs (negative)')
    axes[1].axhline(0,color='grey',lw=.7);axes[1].legend(fontsize=8)
    axes[1].set(xlabel='Minimum hedge interval (minutes)',ylabel='Sum USD · same 22 entries',title='Hedge-frequency counterfactual')
    fig.savefig(out/'forecast_and_hedging.png',dpi=180);plt.close(fig)
    table=lambda f:f.to_html(index=False,border=0,float_format=lambda x:f'{x:,.3f}')
    details=t[IDENT+['actual_calendar_dte','garch_t_forecast_vol','long_execution_iv','realized_vol_to_expiry',
        'gross_option_pnl','hedge_gross_pnl','dividends',*TX,'net_financing','net_pnl',
        'realized_vol_entry_to_exit','frozen_forecast_vol_entry_to_exit']].sort_values('net_pnl')
    totals=pd.DataFrame({'component':['期权中间价损益','股票对冲损益','股票股息','成本前市场损益','期权价差','股票价差','额外滑点',
        '期权佣金','股票佣金及卖出附加费','融资净额（减闲置现金基准和借股）','最终净损益'],
        'USD':[t.gross_option_pnl.sum(),t.hedge_gross_pnl.sum(),t.dividends.sum(),t.market_pnl.sum(),
               *[-t[k].sum() for k in TX],t.net_financing.sum(),t.net_pnl.sum()]})
    costs_view=summary[~summary.scenario.isin(['observed_costs_15m','observed_costs_240m','observed_costs_390m'])]
    ext=ROOT/'artifacts'/'stock_study_20'/'completed_results.csv'
    expansion='<p>20股票扩展正在独立目录计算，使用相同起点、门槛与成本。</p>'
    if ext.exists() and (ext.parent/'report.html').exists():
        f=pd.read_csv(ext,low_memory=False);f=f[(f.policy=='garch_long_flat')&(f.hedge_minutes==60)&(f.cost_scenario=='zero_bid_writedown')]
        f=decompose(f)
        f['group']=np.where(f.asset.isin(candidates.asset.unique()),'原6只','新增14只')
        comp=f.groupby('group').agg(candidates=('side','size'),trades=('side',lambda x:int((x!=0).sum())),
            total_market_usd=('market_pnl','sum'),total_trading_cost=('transaction_cost','sum'),
            total_net_financing=('net_financing','sum'),total_net_usd=('net_pnl','sum'),mean_candidate_return=('return_on_notional','mean')).reset_index()
        comp['mean_candidate_return']*=100
        stocks=f.groupby('asset').agg(candidates=('side','size'),trades=('side',lambda x:int((x!=0).sum())),
            net_usd=('net_pnl','sum'),mean_return_pct=('return_on_notional',lambda x:x.mean()*100)).reset_index()
        comp.to_csv(out/'expansion_original_vs_new.csv',index=False)
        stocks.to_csv(out/'expansion_by_stock.csv',index=False)
        planned=pd.read_csv(ext.parent/'all_results.csv',low_memory=False)
        planned=planned[(planned.policy=='garch_long_flat')&(planned.hedge_minutes==60)&(planned.cost_scenario=='zero_bid_writedown')]
        missing=planned[planned.status!='completed'][IDENT+['side','reason']]
        active=f[f.side!=0]
        expansion=f'<p>扩展至20只股票，保留8个起点、7/30/90天目标、全部信号和费用假设。股票名单在读取新增收益前固定。计划{len(planned)}个候选，{len(f)}个满足完整比较条件，入场{len(active)}笔、盈利{int((active.net_pnl>0).sum())}笔。缺失的{len(missing)}个组合列在下面，不能当作已完成；这两个GARCH本身均无入场信号。原6只、追加14只单独列出；新增股票与原股票处于相同历史阶段，不属于独立时间样本。</p><p><strong>扩展后的区别：</strong>92笔合计成本前市场损益 ${active.market_pnl.sum():,.2f}，实际交易成本 ${active.transaction_cost.sum():,.2f}，融资净额 ${active.net_financing.sum():,.2f}，最终 ${active.net_pnl.sum():,.2f}。扩展后的市场毛收益为正，但不能覆盖成本；原22笔成本前已亏的结论不能机械推广到新增股票。尚不能据此证明未来存在可交易的波动率优势。</p>{table(comp)}<p>上表candidates为完成共同样本数；mean_candidate_return单位为%，包含空仓0。新增14只原计划336组合，完成334；原6只144组合均完成。</p>{table(missing)}{table(stocks)}<p><a href="../stock_study_20/report.html">20股票完整研究及失败清单</a></p>'
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>22笔交易亏损拆解与20股票扩展</title><style>body{{background:#faf9f6;color:#263238;font:16px/1.7 system-ui;margin:0}}main{{max-width:1200px;margin:auto;padding:35px 25px}}h1{{font-size:32px}}h2{{margin-top:36px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}a{{color:#23685c}}.note{{background:#e9efeb;padding:18px}}</style><main>
<h1>22笔交易：亏在预测，还是亏在成本？</h1>
<p class="note">两者都有。原22笔60分钟对冲交易合计净损益 ${t.net_pnl.sum():,.2f}；成本前期权、对冲及股息合计 ${t.market_pnl.sum():,.2f}；实际交易成本 ${t.transaction_cost.sum():,.2f}；扣除闲置现金基准与借股后融资抵回 ${t.net_financing.sum():,.2f}。成本前市场结果已为负，不能把亏损全部归于bid/ask；也不能把全部市场损失因果归于GARCH。</p>
<p>每笔一张call加一张put，独立现金账户；净损益减去了同本金闲置现金基准。美元合计不是固定资本组合的收益率。此22笔沿用原研究的zero_bid_writedown保守口径，零bid多头放弃并承诺不行权；严格双边报价结果仍在<a href="../stock_study/report.html">原报告</a>中另列。</p>
<h2>1. 现金账目逐项核对</h2>{table(totals)}<img src="cash_waterfall.png">
<p>期权中间价损失包含时间价值衰减、价格移动、IV变化等；对冲收益与它共同组成市场损益。不能把期权损失直接称为theta，也不能把全部股票对冲收益称为gamma。以上是可核对的现金会计分解，不是依赖重估顺序的Greek因果分解。每笔残差均小于0.00001美元。</p>
<h2>2. 固定原22笔入场，真正重放成本对照</h2>{table(costs_view)}
<p>所有成本对照均保留原始入场、合约、持有期、报价有效性和对冲订单，并逐笔与原结果核对。不是简单把成本加回：现金变化会改变利息。midpoint情景仅对原本可执行的订单按mid成交，是理想化基准；零bid放弃腿仍按0回收，没有虚构mid卖出，因此可能剩余少量放弃损失。所有情景仍保留借股和融资。各对照改善额有重叠，不能相加。</p>
<h2>3. GARCH预测有没有问题？</h2>{table(scores)}
<p>在选中22笔中，GARCH高估之后到期前RV {int((t.garch_t_forecast_vol>t.realized_vol_to_expiry).sum())}/22笔；之后RV高于信号ask-IV仅{int((t.realized_vol_to_expiry>t.long_execution_iv).sum())}/22笔。选中样本均值：预测{t.garch_t_forecast_vol.mean()*100:.2f}%，ask-IV {t.long_execution_iv.mean()*100:.2f}%，之后RV {t.realized_vol_to_expiry.mean()*100:.2f}%。波动率表内0.30表示30%，bias/MAE列为波动率百分点。</p>
<p>同一预测覆盖到每张实际到期日。上表同时提供全部144候选的诊断，避免只看按高预测筛选后的22笔而宣称整个模型都不准。RV使用15分钟及隔夜对数收益平方和；不是gamma加权方差。交易在到期前退出，因此逐笔CSV额外提供实际入场至实际退出的RV以及用原拟合得到的同窗口预测；对冲频率、波动发生时的gamma、跳空和退出时IV也会影响盈亏。这些误差统计支持信号在此样本中过于乐观，不能单独精确分摊GARCH造成多少美元亏损。</p>
<h2>4. 降低对冲频率会怎样？</h2>{table(summary[summary.scenario.str.match('observed_costs_')])}<img src="forecast_and_hedging.png">
<p>频率对照统一使用原22笔，不重新筛信号；15分钟、60分钟、240分钟、390分钟是最短间隔，跨夜也计时且仍需有效快照，390分钟不保证严格每天只交易一次。减频会同时改变交易成本和裸露Delta损益，不能只看省下多少费用；均值回撤是独立交易的平均路径回撤，不是组合风险。</p>
<h2>5. 为什么只有22笔？</h2>
<p>原实验并非每天扫描：6只股票×8个固定起点×3个目标期限=144候选。{int((~candidates.passes_vol).sum())}个未达到“预测高于两腿ask-IV较高者3个百分点”；剩{int(candidates.passes_vol.sum())}个，其中1个再被成本门槛拦下，最终22个。22笔只出现在{t.origin.nunique()}个起点，且同股票不同期限可能重叠。主要限制是稀疏采样和信号条件，不是费用门槛拦掉很多交易。</p>
<p>20股票扩展仍固定8个起点，以便比较股票覆盖。若要评估真实机会频率，需要另做每周/每月滚动入场和并发持仓资本规则；增加股票不会自动解决时间样本只有8个的问题。没有为了增加交易次数降低门槛。</p>
<h2>6. 原22笔逐笔明细（按净损益排序）</h2>{table(details)}
<h2>7. 更多股票</h2>{expansion}
<p>原研究与扩展仍有15分钟报价重采样、无法验证原tick年龄、连续股息美式定价近似、部分日方差均匀分摊和历史存续股票选择等限制。扩展阈值未按收益调整，不构成可直接交易的盈利证据。</p>
<h2>数据与复现</h2><p><a href="original_22_trade_attribution.csv">原22笔全部分项CSV</a> · <a href="fixed_22_counterfactuals.csv">固定22笔所有重放</a> · <a href="counterfactual_summary.csv">对照汇总</a> · <a href="candidate_funnel_144.csv">144候选及筛除原因</a> · <a href="forecast_diagnostics.csv">预测误差</a> · <a href="run_configuration.json">配置与源文件哈希</a></p>
<p>复现：<code>.venv/bin/python -m volatility_lab.trade_diagnostics</code>。原始报价、原研究成交日志和账目均保留；API密钥不写入结果。</p></main></html>'''
    (out/'report.html').write_text(text)


if __name__=='__main__':main()
