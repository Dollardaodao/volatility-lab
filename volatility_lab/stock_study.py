"""Price, replay and report the fixed multi-stock / period / tenor study."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import html
import json
import os
import hashlib
import inspect
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .forecast import fit_variance,expiry_forecast
from .stock_market import DailyInputs,RateInputs,sessions
from .american import AmericanPricer,enrich_american
from .stock_replay import StockCosts,replay


POLICIES=("garch_long_flat","ewma_long_flat","always_long","garch_long_short")


def decision_inputs(panel,contract,daily,rates):
    decision=pd.Timestamp(contract["decision_time"]);row=panel.loc[decision]
    if not bool(row.all_valid):raise ValueError("No valid origin quotes")
    expiry=pd.Timestamp(contract["expiry_time"])
    tau=(expiry-decision).total_seconds()/(365*86400)
    values={}
    for model in ("garch_t","ewma"):
        fit=fit_variance(daily.prices,decision,horizon=150,model=model)
        f=expiry_forecast(fit,decision,expiry,"equity")
        values[model+"_forecast_vol"]=f["annualized_vol"]
        values[model+"_trained_through"]=str(f["trained_through"])
    pricers={k:AmericanPricer(contract["strike"],k) for k in ("call","put")}
    for k,p in pricers.items():
        p.configure(float(row.spot),tau,float(row.rate),float(row.dividend_yield))
        values[k+"_bid_iv"]=p.implied(float(row[k+"_bid"]))
        values[k+"_ask_iv"]=p.implied(float(row[k+"_ask"]))
        for model in ("garch_t","ewma"):
            values[k+"_"+model+"_fair"]=p.value(values[model+"_forecast_vol"])
    values["long_execution_iv"]=max(values["call_ask_iv"],values["put_ask_iv"])
    values["short_execution_iv"]=min(values["call_bid_iv"],values["put_bid_iv"])
    values["entry_mid_iv"]=(float(row.call_iv)+float(row.put_iv))/2
    for model in ("garch_t","ewma"):
        values[model+"_fair_sum"]=values["call_"+model+"_fair"]+values["put_"+model+"_fair"]
    planned=pd.Timestamp(contract["planned_exit"])
    stock=panel.loc[decision:expiry]
    realized=float(np.square(np.diff(np.log(stock.spot.to_numpy()))).sum())
    values["realized_vol_to_expiry"]=np.sqrt(realized/tau)
    realized_hold=float(np.square(np.diff(np.log(panel.loc[decision:planned].spot.to_numpy()))).sum())
    values["realized_vol_to_planned_exit"]=np.sqrt(realized_hold/((planned-decision).total_seconds()/(365*86400)))
    values["iv_carried_fraction"]=float(panel.iv_carried.mean())
    values["iv_cross_leg_fraction"]=float(panel.iv_cross_leg.mean())
    values["invalid_pair_fraction"]=float((~panel.options_valid).mean())
    values["decision_dividend_yield"]=float(row.dividend_yield)
    values["decision_rate"]=float(row.rate)
    return values


def select_side(policy,values,row,contract,frequency,buffer=.03):
    if policy=="always_long":return 1,{"estimated_long_edge":np.nan,"estimated_short_edge":np.nan}
    model="ewma" if policy=="ewma_long_flat" else "garch_t"
    vol=values[model+"_forecast_vol"]
    fair=values[model+"_fair_sum"]
    spread=(float(row.call_ask-row.call_bid)+float(row.put_ask-row.put_bid))/2
    hold=(pd.Timestamp(contract["planned_exit"])-pd.Timestamp(contract["decision_time"])).total_seconds()/(365*86400)
    ndays=len(sessions(contract["origin"],pd.Timestamp(contract["planned_exit"]).date()))
    hedge_count=ndays*(390/frequency)+2
    # A fixed, deliberately rough two-share turnover assumption is used only
    # for a pre-trade cost hurdle. Actual realized orders determine actual costs.
    hedge_cost=hedge_count*(.35+2*float(row.stock_ask-row.stock_bid)/2+2*float(row.spot)*.0001)
    commission=4.
    premium=100*float(row.call_ask+row.put_ask)
    carry=premium*max(float(row.rate)-.005,0.)*hold
    long_edge=100*(fair-float(row.call_ask+row.put_ask)-spread)-commission-hedge_cost-carry
    short_edge=100*(float(row.call_bid+row.put_bid)-fair-spread)-commission-hedge_cost
    long_ok=vol>values["long_execution_iv"]+buffer and long_edge>0
    short_ok=vol<values["short_execution_iv"]-buffer and short_edge>0
    side=1 if long_ok else (-1 if short_ok and policy=="garch_long_short" else 0)
    return side,{"estimated_long_edge":long_edge,"estimated_short_edge":short_edge}


def run_cohort(contract,daily,rates,out,plan):
    stem=f'{contract["asset"]}_{contract["origin"]}_{contract["target_dte"]}d'
    cache=out/(stem+"_priced.csv")
    if cache.exists():
        panel=pd.read_csv(cache,index_col=0,parse_dates=True)
        panel.index=pd.to_datetime(panel.index,utc=True)
    else:
        panel=pd.read_csv(contract["panel_file"],index_col=0,parse_dates=True)
        panel.index=pd.to_datetime(panel.index,utc=True)
        panel=enrich_american(panel,contract["strike"],daily,rates)
        panel.to_csv(cache)
    values=decision_inputs(panel,contract,daily,rates)
    result=[]
    decision_row=panel.loc[pd.Timestamp(contract["decision_time"])]
    for frequency in plan["hedge_minutes"]:
        for policy in POLICIES:
            side,hurdle=select_side(policy,values,decision_row,contract,frequency,plan["signal_vol_buffer"])
            base={**contract,**values,**hurdle,"policy":policy,"hedge_minutes":frequency,"side":side,"cost_scenario":"base"}
            scenarios=["base","zero_bid_writedown"]
            if policy=="garch_long_flat":scenarios.append("higher_cost")
            for scenario in scenarios:
                costs=StockCosts()
                if scenario=="higher_cost":costs=replace(costs,fee_multiplier=2,extra_slippage_bps=3)
                elif scenario=="zero_bid_writedown":costs=replace(costs,allow_unbid_long_abandonment=True)
                try:
                    replayed,ledger,fills=replay(panel,contract,daily,side,frequency,costs)
                    row={**base,**replayed,"cost_scenario":scenario,"status":"completed"}
                    filename=f"{stem}_{policy}_{frequency}_{scenario}"
                    if len(ledger):ledger.to_csv(out/(filename+"_ledger.csv"),index=False)
                    if len(fills):fills.to_csv(out/(filename+"_fills.csv"),index=False)
                    result.append(row)
                except Exception as exc:
                    result.append({**base,"cost_scenario":scenario,"status":"replay_failed","error_type":type(exc).__name__,"reason":str(exc)})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--limit",type=int)
    p.add_argument("--output-subdir",default="stock_study",help="Use a fresh artifact folder when model inputs or rules change")
    p.add_argument("--plan",type=Path,default=ROOT/"stock_study_plan.json")
    p.add_argument("--manifest",type=Path,default=ROOT/"artifacts"/"stock_study_acquisition.json")
    p.add_argument("--workers",type=int,default=1,help="Independent OS processes; QuantLib state is not shared")
    args=p.parse_args()
    if args.workers<1:p.error("workers must be positive")
    if Path(args.output_subdir).name!=args.output_subdir or args.output_subdir in (".",".."):
        p.error("output-subdir must be one folder name")
    plan=json.loads(args.plan.read_text())
    root=ROOT/"data"/"stock_study";out=ROOT/"artifacts"/args.output_subdir;out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    cohorts=[c for c in manifest["cohorts"] if c["status"]=="acquired"]
    cohorts=sorted(cohorts,key=lambda c:(c["origin"],c["asset"],c["target_dte"]))
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    finance_files=["american.py","forecast.py","stock_market.py","stock_replay.py","theta.py"]
    config={"plan":plan,"model_source_hashes":{name:digest(Path(__file__).parent/name) for name in finance_files},
            "signal_source_hash":hashlib.sha256((inspect.getsource(decision_inputs)+inspect.getsource(select_side)).encode()).hexdigest(),
            "input_hashes":{str(Path(c["panel_file"]).name):digest(c["panel_file"]) for c in cohorts},
            "daily_input_hashes":{s:digest(root/f"{s}_daily_actions.csv") for s in plan["universe"]},
            "rate_input_hash":digest(root/"sofr.json"),
            "numeric_dependencies":{name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","arch","QuantLib")}}
    config_path=out/"run_configuration.json"
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        p.error("Inputs or financial rules changed. Use --output-subdir with a fresh folder; cached outcomes were not reused.")
    config_path.write_text(json.dumps(config,indent=2))
    if args.limit:cohorts=cohorts[:args.limit]
    daily={s:DailyInputs(root,s) for s in plan["universe"]};rates=RateInputs(root/"sofr.json")
    results=[];failures=[]
    pool=ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
    pending={}
    if pool:
        for n,c in enumerate(cohorts,1):
            checkpoint=out/f'{c["asset"]}_{c["origin"]}_{c["target_dte"]}d_results.json'
            if not checkpoint.exists():pending[n]=pool.submit(run_cohort,c,daily[c["asset"]],rates,out,plan)
    for n,c in enumerate(cohorts,1):
        checkpoint=out/f'{c["asset"]}_{c["origin"]}_{c["target_dte"]}d_results.json'
        try:
            if checkpoint.exists():r=json.loads(checkpoint.read_text())
            else:
                r=pending[n].result() if pool else run_cohort(c,daily[c["asset"]],rates,out,plan)
                checkpoint.write_text(json.dumps(r,indent=2,default=str,allow_nan=True))
            results.extend(r)
            failures_here=sum(x["status"]!="completed" for x in r)
            print(f'{n}/{len(cohorts)} {c["asset"]} {c["origin"]} target={c["target_dte"]} actual={c["actual_calendar_dte"]:.1f}d; replay failures={failures_here}',flush=True)
        except Exception as exc:
            error={**c,"status":"pricing_failed","error_type":type(exc).__name__,"reason":str(exc)}
            failures.append(error);print(f'{n}/{len(cohorts)} pricing failed: {c["asset"]} {c["origin"]} {c["target_dte"]}d {type(exc).__name__}',flush=True)
        pd.DataFrame([{k:v for k,v in r.items() if k!="cost_parameters"} for r in results]).to_csv(out/"all_results.csv",index=False)
        (out/"pricing_failures.json").write_text(json.dumps(failures,indent=2,default=str))
    if pool:pool.shutdown()
    frame=pd.DataFrame([{k:v for k,v in r.items() if k!="cost_parameters"} for r in results])
    report(frame,manifest,failures,plan,out)
    print(f'Report: {out/"report.html"}',flush=True)


def report(frame,manifest,failures,plan,out):
    if frame.empty:raise ValueError("No completed pricing cohorts")
    completed=frame[frame.status=="completed"].copy()
    keys=["asset","origin","target_dte","hedge_minutes"]
    primary_names=["garch_long_flat","ewma_long_flat","always_long"]
    counts=completed[(completed.cost_scenario=="base")&completed.policy.isin(primary_names)].groupby(keys).policy.nunique()
    eligible=counts[counts==3].index
    completed["paired_comparison"]=pd.MultiIndex.from_frame(completed[keys]).isin(eligible)
    zero_counts=completed[(completed.cost_scenario=="zero_bid_writedown")&completed.policy.isin(primary_names)].groupby(keys).policy.nunique()
    zero_eligible=zero_counts[zero_counts==3].index
    zmask=completed.cost_scenario=="zero_bid_writedown"
    completed.loc[zmask,"paired_comparison"]=pd.MultiIndex.from_frame(completed.loc[zmask,keys]).isin(zero_eligible)
    completed.to_csv(out/"completed_results.csv",index=False)
    paired=completed[completed.paired_comparison].copy()
    summaries=[]
    for (policy,frequency,cost),f in paired.groupby(["policy","hedge_minutes","cost_scenario"]):
        active=f[f.side!=0]
        grouped=f.groupby("origin").return_on_notional.agg(["sum","count"])
        rng=np.random.default_rng(7317)
        indices=rng.integers(0,len(grouped),size=(3000,len(grouped)))
        boot=grouped["sum"].to_numpy()[indices].sum(axis=1)/grouped["count"].to_numpy()[indices].sum(axis=1)*100
        lo,hi=np.quantile(boot,[.025,.975])
        summaries.append({"policy":policy,"hedge_minutes":frequency,"cost_scenario":cost,"cohorts":len(f),
                          "trades":len(active),"long_trades":int((active.side>0).sum()),"short_trades":int((active.side<0).sum()),
                          "mean_pnl_usd":float(f.net_pnl.mean()),"median_pnl_usd":float(f.net_pnl.median()),
                          "mean_notional_return_pct":float(f.return_on_notional.mean()*100),
                          "mean_active_notional_return_pct":float(active.return_on_notional.mean()*100) if len(active) else np.nan,
                          "origin_bootstrap_low_pct":float(lo),"origin_bootstrap_high_pct":float(hi),
                          "abandoned_legs":int(f.abandoned_long_legs.fillna(0).sum()) if "abandoned_long_legs" in f else 0,
                          "active_win_rate":float((active.net_pnl>0).mean()) if len(active) else np.nan})
    for (frequency,cost),f in paired.groupby(["hedge_minutes","cost_scenario"]):
        n=f.drop_duplicates(["asset","origin","target_dte"]).shape[0]
        summaries.append({"policy":"no_trade","hedge_minutes":frequency,"cost_scenario":cost,"cohorts":n,
                          "trades":0,"long_trades":0,"short_trades":0,"mean_pnl_usd":0.,"median_pnl_usd":0.,
                          "mean_notional_return_pct":0.,"mean_active_notional_return_pct":np.nan,
                          "origin_bootstrap_low_pct":0.,"origin_bootstrap_high_pct":0.,"abandoned_legs":0,"active_win_rate":np.nan})
    summary=pd.DataFrame(summaries);summary.to_csv(out/"strategy_summary.csv",index=False)
    active=paired[(paired.side!=0)&(paired.cost_scenario=="zero_bid_writedown")&(paired.hedge_minutes==60)].copy()
    active["market_pnl"]=active.gross_option_pnl+active.hedge_gross_pnl+active.dividends
    active["transaction_cost"]=active.option_spread_cost+active.stock_spread_cost+active.slippage_cost+active.option_fees+active.stock_fees
    active["net_financing"]=active.cash_interest-active.benchmark_interest-active.borrow_cost
    decomposition=active.groupby("policy").agg(trades=("net_pnl","size"),mean_market_pnl=("market_pnl","mean"),mean_transaction_cost=("transaction_cost","mean"),mean_net_financing=("net_financing","mean"),mean_net_pnl=("net_pnl","mean")).reset_index()
    decomposition.to_csv(out/"active_trade_decomposition.csv",index=False)
    # Also expose the full-grid zero-bid sensitivity. It must not be silently
    # pooled with strict observed-liquidation results.
    for label,group in [("stock",["asset","policy"]),("tenor",["target_dte","policy"]),("period",["origin","policy"])]:
        extra=paired[(paired.cost_scenario=="zero_bid_writedown")&(paired.hedge_minutes==60)].groupby(group).agg(cohorts=("net_pnl","size"),trades=("side",lambda x:int((x!=0).sum())),mean_pnl=("net_pnl","mean"),mean_notional_return=("return_on_notional","mean")).reset_index()
        extra.to_csv(out/f"zero_bid_by_{label}.csv",index=False)
    by_stock=paired[(paired.cost_scenario=="base")&(paired.hedge_minutes==60)].groupby(["asset","policy"]).agg(cohorts=("net_pnl","size"),trades=("side",lambda x:int((x!=0).sum())),mean_pnl=("net_pnl","mean"),mean_notional_return=("return_on_notional","mean")).reset_index()
    by_stock.to_csv(out/"by_stock.csv",index=False)
    by_tenor=paired[(paired.cost_scenario=="base")&(paired.hedge_minutes==60)].groupby(["target_dte","policy"]).agg(cohorts=("net_pnl","size"),trades=("side",lambda x:int((x!=0).sum())),mean_pnl=("net_pnl","mean"),mean_notional_return=("return_on_notional","mean")).reset_index()
    by_tenor.to_csv(out/"by_tenor.csv",index=False)
    by_period=paired[(paired.cost_scenario=="base")&(paired.hedge_minutes==60)].groupby(["origin","policy"]).agg(cohorts=("net_pnl","size"),trades=("side",lambda x:int((x!=0).sum())),mean_pnl=("net_pnl","mean"),mean_notional_return=("return_on_notional","mean")).reset_index()
    by_period.to_csv(out/"by_period.csv",index=False)
    full_stock=pd.read_csv(out/"zero_bid_by_stock.csv")
    full_tenor=pd.read_csv(out/"zero_bid_by_tenor.csv")
    full_period=pd.read_csv(out/"zero_bid_by_period.csv")
    unique=frame.drop_duplicates(["asset","origin","target_dte"])
    forecast_scores=unique.groupby(["asset","target_dte"]).apply(lambda f:pd.Series({"cohorts":len(f),"forecast_vol_mean":f.garch_t_forecast_vol.mean(),"quoted_mid_iv_mean":f.entry_mid_iv.mean(),"realized_vol_mean":f.realized_vol_to_expiry.mean()}),include_groups=False).reset_index()
    forecast_scores.to_csv(out/"volatility_comparison.csv",index=False)
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"#faf9f6","axes.facecolor":"#faf9f6"})
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for policy in ["always_long","garch_long_flat","ewma_long_flat"]:
        a=full_tenor[full_tenor.policy==policy]
        axes[0].plot(a.target_dte,a.mean_notional_return*100,marker="o",label=policy)
        a=full_period[full_period.policy==policy]
        axes[1].plot(pd.to_datetime(a.origin),a.mean_notional_return*100,marker="o",label=policy)
    axes[0].axhline(0,color="grey",lw=.7);axes[1].axhline(0,color="grey",lw=.7)
    axes[0].set(xlabel="Target remaining calendar days",ylabel="Mean excess P&L / entry stock notional (%)",title="Full-grid zero-bid sensitivity · by tenor")
    axes[1].set(xlabel="Fixed origin",ylabel="Mean excess P&L / entry stock notional (%)",title="Full-grid zero-bid sensitivity · by origin")
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.savefig(out/"primary_comparison.png",dpi=170);plt.close(fig)
    primary=paired[(paired.policy=="garch_long_flat")&(paired.cost_scenario=="zero_bid_writedown")&(paired.hedge_minutes==60)]
    heat=primary.pivot_table(index="asset",columns="target_dte",values="return_on_notional",aggfunc="mean")*100
    heat_counts=primary.assign(active=primary.side!=0).pivot_table(index="asset",columns="target_dte",values="active",aggfunc="sum")
    if len(heat):
        fig,ax=plt.subplots(figsize=(7,max(4,len(heat)*.47)),layout="constrained")
        bound=max(float(np.nanmax(abs(heat.to_numpy()))),.1)
        im=ax.imshow(heat.to_numpy(),cmap="RdYlGn",vmin=-bound,vmax=bound,aspect="auto")
        ax.set_xticks(range(len(heat.columns)),[f"{x}d" for x in heat.columns]);ax.set_yticks(range(len(heat.index)),heat.index)
        for i in range(len(heat)):
            for j in range(len(heat.columns)):
                if np.isfinite(heat.iloc[i,j]):ax.text(j,i,f"{heat.iloc[i,j]:.2f}%\n{int(heat_counts.loc[heat.index[i],heat.columns[j]])}/8 entries",ha="center",va="center",fontsize=9)
        ax.set_title("GARCH long/flat · mean return incl. flat cohorts; entries per 8 origins",fontsize=11)
        fig.colorbar(im,ax=ax,label="Percent of entry underlying notional")
        fig.savefig(out/"stock_tenor_heatmap.png",dpi=170);plt.close(fig)
    def table(f):return f.to_html(index=False,border=0,float_format=lambda x:f"{x:.4f}",escape=True)
    def display_slice(f):
        f=f[f.policy=="garch_long_flat"].copy()
        f["mean_notional_return"]*=100
        return f.rename(columns={"mean_notional_return":"mean_return_pct","mean_pnl":"mean_pnl_usd"})
    nplan=len(plan["universe"])*len(plan["origins"])*len(plan["target_calendar_dte"])
    max_residual=float(completed.reconciliation_error.abs().max()) if len(completed) else np.nan
    main_summary=summary[(summary.policy=="garch_long_flat")&(summary.hedge_minutes==60)&(summary.cost_scenario=="zero_bid_writedown")]
    headline=""
    if len(main_summary):
        g=main_summary.iloc[0]
        verdict="样本均值为负，未显示扣成本后的盈利优势" if g.mean_notional_return_pct<0 else "样本均值为正，仍需要独立验证其稳健性"
        headline=f"<p><strong>当前结论：{verdict}。</strong>在60分钟对冲、完整样本保守处理口径下，GARCH买入/空仓策略在{int(g.cohorts)}个候选组合中实际入场{int(g.trades)}笔；候选组合平均净收益为入场标的名义金额的{g.mean_notional_return_pct:.3f}%，实际入场交易平均为{g.mean_active_notional_return_pct:.3f}%。这些是各自持有期结果，不能当作年化收益或已分配资金组合的收益率。</p>"
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多股票、多阶段、多期限波动率研究</title><style>body{{background:#faf9f6;color:#263238;font:16px/1.65 system-ui;margin:0}}main{{max-width:1180px;margin:auto;padding:40px 25px}}h1{{font-size:34px}}h2{{margin-top:40px}}table{{display:block;overflow:auto;border-collapse:collapse;font-size:12px;white-space:nowrap}}td,th{{padding:8px 12px;border-bottom:1px solid #d9dfdb;text-align:left}}th{{background:#e9efeb}}img{{width:100%}}.note{{background:#e9efeb;padding:18px;border-left:4px solid #3c7468}}a{{color:#23685c}}</style><main>
<h1>多股票 × 多历史阶段 × 多到期日</h1>
<p>固定样本：{", ".join(plan["universe"])}；2022–2025年每年两个起点；目标剩余7、30、90天；15分钟及60分钟对冲。</p>
<div class="note">本次使用真实历史股票和期权 NBBO 快照计算交易现金流。结果仍依赖成交、融资和定价模型假设，并非实盘成交记录。主要研究是买入跨式或空仓；卖出跨式结果单独作为探索性情景，未重现真实经纪商提前指派。</div>
{headline}
<h2>覆盖与账目检查</h2><p>计划 {nplan} 个股票/起点/期限组合；取得 {sum(c['status']=='acquired' for c in manifest['cohorts'])} 个组合；完成定价 {len(unique)} 个组合。完整主策略配对比较包含 {len(eligible)} 个“组合×对冲频率”。所有已完成回放最大现金核对误差：${max_residual:.8f}。失败及未完成数据全部保留在下载清单、定价失败和逐笔结果文件内。</p>
<h2>策略对比</h2>{table(summary)}<p>cohorts 包含空仓；trades 仅计实际入场。收益均扣除买卖价差、额外滑点、佣金、借股费用和融资差额，并减去同本金闲置现金基准。每笔一张call及一张put，标准乘数100；初始现金为标的100股名义金额的3倍。每个组合独立出资，表中平均数和逐组合结果不是一个可同时持有全部重叠期权的组合年化收益。没有用重叠样本计算Sharpe。</p>
<p><strong>两种平仓口径分别列出：</strong>base 为严格双边报价平仓，对照采用共同完整样本，共 {len(eligible)} 个“组合×频率”。zero_bid_writedown 为另列的保守敏感性分析，共 {len(zero_eligible)} 个共同“组合×频率”：如果多头腿有明确0 bid、仍有正ask及正常报价条件，则该腿按零回收并承诺到期不行权，放弃未来剩余权利；日志标记executed=false，不伪造卖出成交，也不收取卖出佣金。卖出腿平仓只需实际正ask及足够挂单量。abandoned_legs 为放弃的多头腿数量。这是发现零bid数据问题后新增的执行敏感性分析，不是收益调参，也不是原始预注册规则；它与严格样本不能混为同一回测。</p>
<p>mean_notional_return_pct 是每个组合的平均净损益/入场100股标的名义金额，包含空仓的0；mean_active_notional_return_pct 只计实际入场。所有百分比都是持有期结果，不是年化收益。origin_bootstrap 区间按历史起点整组重采样3000次，保留同一起点股票和期限的相关性；仅有8个起点，该区间用于描述样本不确定性，不能保证严格覆盖率或证明可推广的盈利。</p>
<h2>实际入场交易：损益与成本拆解（60分钟、保守口径，美元）</h2>{table(decomposition)}
<img src="primary_comparison.png" alt="不同期限及历史阶段的主策略比较">
{'<img src="stock_tenor_heatmap.png" alt="股票及期限结果热图">' if len(heat) else ''}
<h2>GARCH择时买入：按股票（60分钟、完整样本保守口径）</h2>{table(display_slice(full_stock))}<h2>按到期日目标</h2>{table(display_slice(full_tenor))}<h2>按历史起点</h2>{table(display_slice(full_period))}
<h2>预测、报价IV与之后实际波动率</h2>{table(forecast_scores)}
<p>尤其需要注意：每日模型将首次交易日的剩余方差按前一收盘到下一收盘的自然时间比例分摊，含周末和节假日。这是近似，对短期限结果可能有明显影响；没有根据本次收益调整这一规则。</p>
<p>表内波动率为小数：0.30即30%。GARCH 使用最多756个、至少504个过去的拆股调整日收益；每个起点重新拟合，完全不使用该起点之后的收益。对每张实际到期日分别汇总条件方差，并按ACT/365年化；部分日间区间均匀分配方差。实际波动率使用股票15分钟及隔夜对数收益平方和，包括除息价格变化。它不等同于期权的gamma加权实际方差，因此不能单凭RV大于IV就断定有利润。</p>
<h2>预先固定的交易规则</h2><p>仅使用起点09:45纽约时间已出现的有效报价选到期日与接近平值的共同执行价；10:00按当时bid/ask入场。买入信号要求预测波动率比两腿ask-IV的较高者高3个波动率百分点，并通过一个固定的粗略交易成本门槛；EWMA使用相同规则。门槛预计每次对冲周转两股，未按结果拟合。实际费用根据实际整股订单计算。15/60分钟对冲目标来自上一个15分钟观察点，不能用当前成交价格反推当前订单。正常退出为到期前一个交易日收市前15分钟；缺少有效双边报价时按既定规则等待下一个有效快照，若到期前仍无法平仓则标为未完成，不能凭模型价虚构成交。</p>
<p>garch_long_short 是探索性扩展：预测低于两腿bid-IV的较低者3个百分点且通过成本门槛时才卖出。当前报价显示看涨期权时间价值低于过去年度分红的季度代理值，或看跌期权时间价值低于一天利息代理值时，提交下一有效快照提前平仓。这是主动风险退出规则，无法保证避免现实中的提前指派。</p>
<h2>成本、公司行为及限制</h2><p>American期权IV及Delta采用QuantLib Bjerksund–Stensland近似，连续股息率来自起点之前实际分红，按当前股价换算；日期系数缩放保持精确到期剩余年数。实际持仓股票的现金股息在除息日作为应收/应付入账；支付日资金占用差异未细分。美式期权在离散除息日附近的最优行权边界只作近似，不应将卖出情景当作完整指派模拟。输入中的拆股调整仅用于价格和历史分红单位还原；若持仓窗口出现尚未支持的合约调整，组合被明确标记，不能默默跨过拆股。</p>
<p>SOFR来自纽约联储公开历史数据，至少滞后两个自然日进入模型。现金利率假设为SOFR减50bp（最低0），借款为SOFR加150bp，借股另计每年2%；不是用户经纪商真实费率。股票每股佣金0.0035美元、最低0.35美元，卖出附加0.1bp，额外滑点每边1bp；期权每张每腿每次1美元。higher_cost固定原有入场选择，将佣金翻倍、滑点调至3bp。股票报价数量按研究期内100股round-lot单位假设约束成交；该供应商单位映射、借股可得性、真实队列与市场冲击仍需独立审计。</p>
<p>报价是15分钟重采样快照，不能据此证明原始tick的新鲜度；实际挂单成交可能不同。所有非正常交易时段、包括提前收市后的快照均排除，正价格、挂单量和保守报价条件均检查。不能反演IV时只允许沿用之前有效IV（最多4天）估计后续对冲Delta；实际入场和平仓仍使用真实有效报价。表中保留沿用IV比例。缺少完整平仓报价的组合可能造成样本选择偏差，因此所有主策略只在同一组完整组合内比较，失败清单必须同时阅读。样本是固定的{len(plan['universe'])}只存续股票及8个起点，不能代表全部市场；不同期限及股票间存在相关性，热图最佳单元格不是经多重检验确认的投资机会。阈值没有按本次收益优化，但仍需要独立后续样本验证。</p>
<p>当某腿无双边报价而另一条相同行权价、相同到期日的腿仍可反演IV时，用该另一腿的当时IV估计缺失腿Delta，并记录iv_cross_leg_fraction；两腿都不可用时才沿用过去IV，最多4天。零bid放弃情景仅将未卖出的多头权利按零价值核算，其剩余权利若实际继续持有或执行，结果会不同。所有情景仍缺少真实清算和经纪商操作记录。</p>
<h2>原始结果和复现</h2><p><a href="all_results.csv">全部场景及失败原因</a> · <a href="completed_results.csv">完成结果</a> · <a href="strategy_summary.csv">策略汇总</a> · <a href="volatility_comparison.csv">波动率对比</a> · <a href="pricing_failures.json">定价失败</a> · <a href="../../{html.escape(plan.get('_plan_file','stock_study_plan.json'))}">固定实验参数</a> · <a href="../{html.escape(plan.get('_manifest_file','stock_study_acquisition.json'))}">下载清单</a></p>
<p>各组合的priced文件、逐笔成交、现金账目及参数均与本报告同目录保存；原始历史报价和请求参数、检索时间及SHA-256哈希位于data/stock_study/quotes。API key和会话凭据不包含在任何报告内。运行：<code>.venv/bin/python -m volatility_lab.stock_acquire</code>，然后 <code>.venv/bin/python -m volatility_lab.stock_study</code>。缓存支持中断后继续。</p>
<p><a href="run_configuration.json">本次参数、数值依赖及输入/模型哈希</a> · <a href="{'../stock_study/' if len(plan['universe'])>6 else ''}full_snapshot_selection_audit.json">原6股票完整09:45期权链选约审计</a>。原6股票的完整执行价核对发现并更正了4个NVDA组合的供应商附近执行价筛选偏差；新扩展直接使用完整执行价快照。更正依据当时股票中间价和当时期权链，与之后收益无关。模型或输入变化后必须选择新的输出目录，不能静默沿用旧结果。</p>
<p>方法参考：<a href="https://github.com/lballabio/QuantLib/blob/master/test-suite/americanoption.cpp">QuantLib美式期权引擎测试</a> · <a href="https://thetadata.net/docs/operations/option_history_quote.html">期权报价口径</a> · <a href="https://thetadata.net/docs/Articles/Errors-Exchanges-Conditions/Quote-Conditions.html">报价条件</a> · <a href="https://www.newyorkfed.org/markets/reference-rates/sofr">纽约联储SOFR</a>。</p></main></html>'''
    (out/"report.html").write_text(text)


if __name__=="__main__":main()
