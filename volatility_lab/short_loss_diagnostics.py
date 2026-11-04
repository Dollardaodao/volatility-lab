"""Largest unique short losses, frozen-entry ablations, and threshold subsets."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import argparse
import hashlib
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .stock_market import DailyInputs
from . import stock_replay
from .short_vol_study import account,trade_diagnostics
from .forecast import fit_variance,expiry_forecast
from .ml_vol import past_features

KEYS=['asset','origin','target_dte','model','direction']
COHORT=['asset','origin','target_dte']
SOURCE=ROOT/'artifacts'/'short_vol_study'


def threshold_analysis(out,plan):
    paired=pd.read_csv(SOURCE/'paired_results.csv',low_memory=False)
    actual=paired[paired.cost_scenario=='observed_costs'].copy()
    ideal=paired[paired.cost_scenario=='no_execution_cost'][KEYS+['net_pnl','return_on_notional']].rename(
        columns={'net_pnl':'ideal_net_pnl','return_on_notional':'ideal_return_on_notional'})
    actual=actual.merge(ideal,on=KEYS,validate='one_to_one')
    decisions=pd.read_csv(SOURCE/'decisions.csv',low_memory=False)
    decisions['gap_vol_points']=np.where(decisions.direction=='short',
        decisions.short_execution_iv-decisions.forecast_vol,
        decisions.forecast_vol-decisions.long_execution_iv)*100
    summaries=[];origins=[];periods=[];selected=[];decision_sets=[]
    def summarize(f,a,model,direction,threshold):
        scorable=a[a.vol_bet_status=='scorable']
        return {'model':model,'direction':direction,'threshold_vol_points':threshold,'candidates':len(f),'trades':len(a),
            'market_pnl':a.market_pnl.sum(),'no_execution_cost_net_pnl':a.ideal_net_pnl.sum(),
            'observed_net_pnl':a.net_pnl.sum(),'execution_cost':a.execution_cost.sum(),
            'net_financing':a.net_financing.sum(),'vol_bet_correct':int(scorable.vol_bet_correct.astype(bool).sum()),
            'vol_bet_scorable':len(scorable),'observed_net_wins':int((a.net_pnl>0).sum()),
            'no_cost_net_wins':int((a.ideal_net_pnl>0).sum()),
            'mean_active_observed_notional_pct':a.return_on_notional.mean()*100 if len(a) else np.nan,
            'mean_candidate_observed_notional_pct':a.return_on_notional.sum()/len(f)*100,
            'mean_candidate_no_cost_notional_pct':a.ideal_return_on_notional.sum()/len(f)*100,
            'worst_observed_trade':a.net_pnl.min() if len(a) else np.nan}
    for threshold in plan['thresholds_vol_points']:
        d=decisions.copy();d['threshold_vol_points']=threshold
        edge=np.where(d.direction=='short',d.estimated_short_edge,d.estimated_long_edge)
        d['selected']=(d.gap_vol_points>threshold)&(edge>0)&~((d.direction=='short')&d.decision_exercise_risk)
        if (d.selected&(d.side==0)).any():raise AssertionError('A higher threshold created a new trade')
        decision_sets.append(d)
        for (model,direction),f in actual.groupby(['model','direction']):
            f=f.merge(d[KEYS+['selected','gap_vol_points']],on=KEYS,validate='one_to_one')
            a=f[f.selected].copy();a['threshold_vol_points']=threshold;selected.append(a)
            summaries.append(summarize(f,a,model,direction,threshold))
            for origin,ff in f.groupby('origin'):
                origins.append({**summarize(ff,ff[ff.selected],model,direction,threshold),'origin':origin})
            for label,mask in [('2022_2024',pd.to_datetime(f.origin).dt.year<2025),('2025',pd.to_datetime(f.origin).dt.year==2025)]:
                ff=f[mask]
                periods.append({**summarize(ff,ff[ff.selected],model,direction,threshold),'period':label})
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'threshold_summary.csv',index=False)
    pd.DataFrame(origins).to_csv(out/'threshold_by_origin.csv',index=False)
    pd.DataFrame(periods).to_csv(out/'threshold_by_period.csv',index=False)
    pd.concat(selected,ignore_index=True).to_csv(out/'threshold_selected_trades.csv',index=False)
    pd.concat(decision_sets,ignore_index=True).to_csv(out/'threshold_candidate_decisions.csv',index=False)
    previous=pd.read_csv(SOURCE/'strategy_summary.csv')
    check=summary[summary.threshold_vol_points==3].merge(previous,on=['model','direction'],suffixes=('_now','_prior'),validate='one_to_one')
    assert np.array_equal(check.trades_now,check.trades_prior)
    assert np.allclose(check.observed_net_pnl_now,check.observed_net_pnl_prior,atol=1e-7,rtol=0)
    assert np.allclose(check.no_execution_cost_net_pnl_now,check.no_execution_cost_net_pnl_prior,atol=1e-7,rtol=0)
    return actual


def choose_cases(actual,plan):
    shorts=actual[(actual.direction=='short')&(actual.side==-1)]
    representatives=[]
    preference=['garch_t','ewma','ml_hist_gradient_boosting','rolling21','har_proxy_ridge']
    for _,f in shorts.groupby(COHORT):
        if not np.allclose(f.net_pnl,f.net_pnl.iloc[0],atol=1e-7,rtol=0):raise AssertionError('Same short position has different model cash flows')
        model=next(n for n in preference if n in set(f.model))
        r=f[f.model==model].iloc[0].to_dict()
        r['selected_by_models']=[n for n in preference if n in set(f.model)]
        r['case_id']=f'{r["asset"]}_{r["origin"]}_{r["target_dte"]}d'
        representatives.append(r)
    return sorted(representatives,key=lambda r:r['ideal_net_pnl'])[:plan['largest_loss_count']]


def audit_fills(panel,fills,result,costs):
    records=[]
    for r in fills.itertuples():
        time=pd.Timestamp(r.time);row=panel.loc[time];q=int(r.quantity)
        kind=r.instrument;option=kind in ('call','put');multiplier=100 if option else 1
        mid=float(row[kind+'_mid']) if option else float(row.spot)
        bid=float(row[kind+'_bid']) if option else float(row.stock_bid)
        ask=float(row[kind+'_ask']) if option else float(row.stock_ask)
        quote=ask if q>0 else bid
        expected_price=quote if option else quote*(1+np.sign(q)*costs.extra_slippage_bps/10000)
        commission=(abs(q)*costs.option_commission_per_contract if option else
                    max(costs.minimum_stock_commission,abs(q)*costs.stock_commission_per_share))*costs.fee_multiplier
        surcharge=0. if option else max(-q,0)*float(r.fill_price)*costs.stock_sell_fee_bps/10000*costs.fee_multiplier
        if abs(expected_price-r.fill_price)>1e-7 or abs(commission+surcharge-r.fee)>1e-7:
            raise AssertionError('Actual fill is inconsistent with quoted price/fee rules')
        records.append({'time_utc':str(time),'time_new_york':str(time.tz_convert('America/New_York')),
            'signal_time':r.decision_time,'instrument':kind,'quantity':q,'multiplier':multiplier,
            'bid':bid,'ask':ask,'mid':mid,'quote_side_price':quote,'fill_price':r.fill_price,
            'spread_cost':abs(q)*multiplier*abs(quote-mid),'added_slippage':abs(q)*multiplier*abs(r.fill_price-quote),
            'commission':commission,'sell_surcharge':surcharge,'total_fee':r.fee,
            'cash_flow':-q*multiplier*r.fill_price-r.fee,
            'midpoint_cash_flow':-q*multiplier*mid})
    f=pd.DataFrame(records)
    expected={'option_spread_cost':f.loc[f.instrument!='stock','spread_cost'].sum(),
        'stock_spread_cost':f.loc[f.instrument=='stock','spread_cost'].sum(),
        'slippage_cost':f.added_slippage.sum(),'option_fees':f.loc[f.instrument!='stock','total_fee'].sum(),
        'stock_fees':f.loc[f.instrument=='stock','total_fee'].sum()}
    for name,value in expected.items():
        if abs(value-result[name])>1e-5:raise AssertionError('Fill-cost total does not match cash replay: '+name)
    reconstructed=f.cash_flow.sum()+result['cash_interest']-result['benchmark_interest']-result['borrow_cost']+result['dividends']
    if abs(reconstructed-result['net_pnl'])>1e-5:raise AssertionError('Fill-by-fill cash reconciliation failed')
    return f,{'net_pnl_from_fill_cash':reconstructed,'net_pnl':result['net_pnl'],
        'cash_residual':reconstructed-result['net_pnl'],**expected}


def interval_attribution(panel,contract,ledger,daily,result):
    times=pd.DatetimeIndex(pd.to_datetime(ledger.time,utc=True))
    p=panel.loc[pd.Timestamp(contract['decision_time']):pd.Timestamp(contract['expiry_time'])]
    mark=(p.call_mid+p.put_mid).where(p.options_valid).ffill().reindex(times)
    mark.iloc[-1]=float(panel.loc[times[-1],'call_mid']+panel.loc[times[-1],'put_mid'])
    spot=panel.reindex(times).spot.to_numpy();shares=ledger.stock_shares.to_numpy()
    option=-100*np.diff(mark.to_numpy(),prepend=mark.iloc[0])
    hedge=np.r_[0,shares[:-1]]*np.diff(spot,prepend=spot[0])
    local_dates=times.tz_convert('America/New_York').date
    dividends=np.zeros(len(times));overnight=np.zeros(len(times),dtype=bool)
    for i in range(1,len(times)):
        overnight[i]=local_dates[i]!=local_dates[i-1]
        if overnight[i]:dividends[i]=shares[i-1]*daily.dividends.get(local_dates[i],0.)
    market=option+hedge+dividends
    if abs(option.sum()-result['gross_option_pnl'])>1e-5 or abs(hedge.sum()-result['hedge_gross_pnl'])>1e-5 or abs(market.sum()-result['market_pnl'])>1e-5:
        raise AssertionError('Interval mark attribution failed to reconcile')
    residual_delta=np.r_[0,ledger.delta_shares.to_numpy()[:-1]]
    return pd.DataFrame({'time':times,'time_new_york':times.tz_convert('America/New_York'),
        'spot':spot,'stock_return':np.r_[0,np.diff(np.log(spot))],'overnight':overnight,
        'shares_before_interval':np.r_[0,shares[:-1]],'delta_residual_before_interval':residual_delta,
        'option_mark_pnl':option,'hedge_pnl':hedge,'dividends':dividends,'market_pnl':market,
        'linear_delta_residual_estimate':residual_delta*np.diff(spot,prepend=spot[0]),
        'cumulative_market_pnl':np.cumsum(market)})


def model_forecast_diagnostics(panel,c,daily,case):
    decision=pd.Timestamp(c['decision_time']);entry=pd.Timestamp(case['entry']);exit=pd.Timestamp(case['exit'])
    expiry=pd.Timestamp(c['expiry_time']);rows=[];saved=None
    prices=daily.prices;raw=pd.read_csv(ROOT/'data'/'stock_study'/f'{c["asset"]}_daily_actions.csv')
    features,_=past_features(raw,prices.index);feature_time=prices.loc[:decision].index[-1]
    source=pd.read_csv(SOURCE/'active_trade_details.csv',low_memory=False)
    source=source[(source.direction=='short')&(source.asset==c['asset'])&(source.origin==c['origin'])&(source.target_dte==c['target_dte'])]
    for model in case['selected_by_models']:
        if model in ('garch_t','ewma','rolling21'):fit=fit_variance(prices,decision,horizon=84,model=model)
        else:
            if saved is None:saved=joblib.load(ROOT/'artifacts'/'ml_vol_study'/'models'/(c['origin']+'.joblib'))
            fit=saved[model].curve(features.loc[feature_time],c['asset'],feature_time)
        full=expiry_forecast(fit,decision,expiry,'equity')['annualized_vol']
        original=source[source.model==model].iloc[0]
        if abs(full-original.forecast_vol)>1e-8:raise AssertionError('Archived forecast could not be reproduced')
        holding=expiry_forecast(fit,entry,exit,'equity')['annualized_vol']
        entry_life=expiry_forecast(fit,entry,expiry,'equity')['annualized_vol']
        all_prices=panel.loc[entry:expiry].spot
        full_rv=np.sqrt(np.square(np.diff(np.log(all_prices.to_numpy()))).sum()/((expiry-entry).total_seconds()/(365*86400)))
        rows.append({'case_id':case['case_id'],'model':model,'forecast_vol_to_expiry':full,
            'forecast_vol_for_actual_holding':holding,'forecast_vol_entry_to_expiry':entry_life,
            'signal_bid_iv':original.short_execution_iv,'entry_bid_iv':original.entry_execution_iv,
            'actual_holding_rv':original.holding_realized_vol,'rv_entry_to_expiry':full_rv,
            'decision_gap_vol_points':(original.short_execution_iv-full)*100,
            'actual_entry_gap_vol_points':(original.entry_execution_iv-entry_life)*100,
            'trained_through':str(fit.trained_through)})
    return rows


def analyze_case(case,plan,outstr):
    out=Path(outstr);folder=out/'cases'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'diagnostics.json'
    if checkpoint.exists():return str(checkpoint)
    manifest=json.loads((ROOT/'artifacts'/'stock_study_acquisition_20.json').read_text())
    c=next(c for c in manifest['cohorts'] if all(c[k]==case[k] for k in COHORT))
    panel=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/(case['case_id']+'_priced.csv'),index_col=0,parse_dates=True)
    panel.index=pd.to_datetime(panel.index,utc=True);daily=DailyInputs(ROOT/'data'/'stock_study',c['asset'])
    source_costs=json.loads((ROOT/'short_vol_plan.json').read_text())
    costs={name:stock_replay.StockCosts(**source_costs['costs_midpoint' if name=='no_execution_cost' else 'costs_observed']) for name in plan['counterfactual_costs']}
    stem=f'{case["case_id"]}_{case["model"]}_short_'
    original={}
    all_source=pd.read_csv(SOURCE/'all_results.csv',low_memory=False)
    mask=(all_source.asset==case['asset'])&(all_source.origin==case['origin'])&(all_source.target_dte==case['target_dte'])&(all_source.model==case['model'])&(all_source.direction=='short')
    for scenario in plan['counterfactual_costs']:
        original[scenario]=all_source[mask&(all_source.cost_scenario==scenario)].iloc[0].to_dict()
        for kind in ('fills','ledger'):
            shutil.copy2(SOURCE/'trades'/(stem+scenario+'_'+kind+'.csv'),folder/('baseline_'+scenario+'_'+kind+'.csv'))
    fills=pd.read_csv(folder/'baseline_observed_costs_fills.csv')
    audit,audit_summary=audit_fills(panel,fills,original['observed_costs'],costs['observed_costs'])
    audit.to_csv(folder/'fill_cost_audit.csv',index=False)
    ledger=pd.read_csv(folder/'baseline_no_execution_cost_ledger.csv')
    intervals=interval_attribution(panel,c,ledger,daily,original['no_execution_cost'])
    intervals.to_csv(folder/'interval_attribution.csv',index=False)
    intervals.nsmallest(8,'market_pnl').to_csv(folder/'largest_loss_intervals.csv',index=False)
    forecasts=model_forecast_diagnostics(panel,c,daily,case)
    pd.DataFrame(forecasts).to_csv(folder/'forecast_horizon_audit.csv',index=False)
    held=panel.loc[pd.Timestamp(case['entry']):pd.Timestamp(case['exit'])]
    valid_stock=held.loc[held.stock_passes_snapshot_filters,'spot']
    years=(pd.Timestamp(case['exit'])-pd.Timestamp(case['entry'])).total_seconds()/(365*86400)
    quality={'holding_grid_rows':len(held),'invalid_stock_snapshots':int((~held.stock_passes_snapshot_filters).sum()),
        'invalid_option_pairs':int((~held.options_valid).sum()),
        'rv_all_snapshots':float(case['holding_realized_vol']),
        'rv_valid_stock_snapshots':float(np.sqrt(np.square(np.diff(np.log(valid_stock.to_numpy()))).sum()/years)),
        'exit_call_extrinsic_mid':float(held.iloc[-1].call_mid-max(float(held.iloc[-1].spot)-c['strike'],0)),
        'exit_put_extrinsic_mid':float(held.iloc[-1].put_mid-max(c['strike']-float(held.iloc[-1].spot),0)),
        'exit_quoted_stock_valid':bool(held.iloc[-1].stock_passes_snapshot_filters),
        'exit_options_valid':bool(held.iloc[-1].options_valid)}
    outcomes=[]
    for policy in plan['exit_diagnostics']:
        contract=dict(c)
        if policy=='fixed_original_actual_exit':contract['planned_exit']=case['exit']
        for frequency in plan['hedge_frequencies_minutes']:
            per_cost={};order_lists={}
            for scenario,cost in costs.items():
                base={'case_id':case['case_id'],'model':case['model'],'exit_policy':policy,
                      'hedge_minutes':frequency,'cost_scenario':scenario,'requested_exit':str(contract['planned_exit'])}
                try:
                    # Per-worker, scoped ablation. The preserved original
                    # financial source and all original results remain intact.
                    with patch.object(stock_replay,'exercise_risk',return_value=False):
                        r,path,fs=stock_replay.replay(panel,contract,daily,-1,frequency,cost)
                except Exception as exc:
                    per_cost[scenario]={**base,'status':'replay_failed','reason':str(exc),'error_type':type(exc).__name__}
                    continue
                diagnostics=trade_diagnostics(panel,contract,r,-1,fs);bridge=account(r,diagnostics)
                per_cost[scenario]={**base,**r,**diagnostics,**bridge,'status':'completed',
                    'exit_at_requested_time':pd.Timestamp(r['exit'])==pd.Timestamp(contract['planned_exit'])}
                order_lists[scenario]=fs[['time','instrument','quantity']].to_dict('records')
                name=f'{policy}_{frequency}m_{scenario}'
                path.to_csv(folder/(name+'_ledger.csv'),index=False);fs.to_csv(folder/(name+'_fills.csv'),index=False)
                if policy=='fixed_original_actual_exit' and frequency==60:
                    if abs(r['net_pnl']-original[scenario]['net_pnl'])>1e-5:
                        raise AssertionError('Fixed-exit reference did not reproduce baseline cash')
            if all(v['status']=='completed' for v in per_cost.values()):
                same=order_lists['observed_costs']==order_lists['no_execution_cost']
                for result in per_cost.values():result['same_orders_across_costs']=same
            outcomes.extend(per_cost.values())
    payload={'case':case,'audit':audit_summary,'quality':quality,'forecasts':forecasts,'counterfactuals':outcomes,
             'worst_intervals':intervals.nsmallest(5,'market_pnl').assign(time=lambda f:f.time.astype(str),time_new_york=lambda f:f.time_new_york.astype(str)).to_dict('records')}
    checkpoint.write_text(json.dumps(payload,indent=2,default=str,allow_nan=True))
    return str(checkpoint)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.workers<1:p.error('workers must be positive')
    plan=json.loads((ROOT/'short_loss_diagnostics_plan.json').read_text());out=ROOT/'artifacts'/'short_loss_diagnostics'
    (out/'cases').mkdir(parents=True,exist_ok=True)
    # Compute the threshold subsets once; there are no new fills to infer.
    actual=threshold_analysis(out,plan);cases=choose_cases(actual,plan)
    pd.DataFrame(cases).to_csv(out/'worst_unique_trades.csv',index=False)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    names=['short_loss_diagnostics.py','stock_replay.py','short_vol_study.py','forecast.py','ml_vol.py','american.py','stock_market.py']
    inputs=[SOURCE/'paired_results.csv',SOURCE/'all_results.csv',SOURCE/'decisions.csv',SOURCE/'active_trade_details.csv',
            ROOT/'artifacts'/'stock_study_acquisition_20.json',ROOT/'short_vol_plan.json']
    for c in cases:
        inputs.append(ROOT/'artifacts'/'stock_study_20'/(c['case_id']+'_priced.csv'))
        inputs.append(ROOT/'data'/'stock_study'/f'{c["asset"]}_daily_actions.csv')
        inputs.append(ROOT/'artifacts'/'ml_vol_study'/'models'/(c['origin']+'.joblib'))
    config={'plan':plan,'source_hashes':{n:digest(Path(__file__).parent/n) for n in names},
            'input_hashes':{str(f.relative_to(ROOT)):digest(f) for f in inputs},
            'selected_cases':[c['case_id'] for c in cases]}
    configfile=out/'run_configuration.json'
    if configfile.exists() and json.loads(configfile.read_text())!=config:raise ValueError('Diagnostic rules or data changed; preserve the old run before recomputing')
    configfile.write_text(json.dumps(config,indent=2))
    work=cases[:args.limit] if args.limit else cases;paths=[];errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map={pool.submit(analyze_case,c,plan,str(out)):c['case_id'] for c in work}
        for n,future in enumerate(as_completed(future_map),1):
            try:paths.append(future.result());print(f'{n}/{len(work)} {future_map[future]} complete',flush=True)
            except Exception as exc:
                errors.append({'case_id':future_map[future],'error_type':type(exc).__name__,'reason':str(exc)})
                print(json.dumps(errors[-1]),flush=True)
            (out/'errors.json').write_text(json.dumps(errors,indent=2))
    payloads=[json.loads(Path(path).read_text()) for path in sorted(paths)]
    rows=[r for payload in payloads for r in payload['counterfactuals']]
    pd.DataFrame([{k:v for k,v in r.items() if k!='cost_parameters'} for r in rows]).to_csv(out/'counterfactual_results.csv',index=False)
    pd.DataFrame([r for payload in payloads for r in payload['forecasts']]).to_csv(out/'matched_forecast_diagnostics.csv',index=False)
    pd.DataFrame([{'case_id':payload['case']['case_id'],**payload['audit']} for payload in payloads]).to_csv(out/'fill_audit_totals.csv',index=False)
    pd.DataFrame([{'case_id':payload['case']['case_id'],**payload['quality']} for payload in payloads]).to_csv(out/'quote_quality_audit.csv',index=False)
    if errors:raise RuntimeError('Some diagnostic jobs failed; no fallback P&L was fabricated')
    from .short_loss_report import make_report
    make_report(out,plan,complete=len(work)==len(cases))
    print(f'Report: {out/"report.html"}',flush=True)


if __name__=='__main__':main()
