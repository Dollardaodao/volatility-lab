"""True one-minute quote replay with separately controlled hedge signal delay."""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd
from .theta_sdk import ROOT,connect
from .theta import normalize_quotes
from .stock_market import MarketStore,DailyInputs,RateInputs,month_ranges,assemble_panel,local_time
from .american import enrich_american
from .stock_replay import StockCosts
from .highfreq_replay import replay
from .short_vol_study import account,trade_diagnostics


def cases_from_plan(plan):
    cases=pd.read_csv(ROOT/'artifacts'/'short_loss_diagnostics'/'worst_unique_trades.csv',low_memory=False)
    return [cases[cases.case_id==name].iloc[0].to_dict() for name in plan['cases']]


def quote_jobs(cases):
    manifest=json.loads((ROOT/'artifacts'/'stock_study_acquisition_20.json').read_text())['cohorts']
    jobs={};mapping={}
    for case in cases:
        c=next(c for c in manifest if all(c[k]==case[k] for k in ['asset','origin','target_dte']))
        stock_names=[];option_names=[]
        for first,last in month_ranges(case['origin'],pd.Timestamp(case['exit']).date()):
            name=f'{case["asset"]}_stock_1m_{first:%Y%m}'
            jobs[name]=('stock_history_quote',{'symbol':case['asset'],'start_date':first,'end_date':last,
                'interval':'1m','venue':'utp_cta'})
            stock_names.append(name)
            begin=max(first,pd.Timestamp(case['origin']).date());end=min(last,pd.Timestamp(case['exit']).date())
            name=f'{case["asset"]}_{c["expiry_date"]}_{c["strike"]:.3f}_1m_{begin}_{end}'
            jobs[name]=('option_history_quote',{'symbol':case['asset'],'expiration':pd.Timestamp(c['expiry_date']).date(),
                'strike':f'{c["strike"]:.3f}','right':'both','start_date':begin,'end_date':end,'interval':'1m'})
            option_names.append(name)
        mapping[case['case_id']]={'contract':c,'stock_names':stock_names,'option_names':option_names}
    return jobs,mapping


def acquire(jobs,root):
    missing=[(name,method,params) for name,(method,params) in jobs.items() if not (root/(name+'.csv')).exists()]
    if not missing:return
    try:client=connect()
    except Exception as exc:raise RuntimeError('Theta authentication failed ('+type(exc).__name__+'); no raw response is logged') from None
    store=MarketStore(root,client);errors=[]
    def fetch(name,method,params):
        try:
            frame=store.fetch(name,method,**params)
            return {'name':name,'status':'acquired','rows':len(frame)}
        except Exception as exc:
            result={'name':name,'status':'failed','error_type':type(exc).__name__}
            if callable(getattr(exc,'code',None)):result['grpc_status']=str(exc.code())
            return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fetch,name,method,params) for name,method,params in missing]
        for future in as_completed(futures):
            result=future.result();print(json.dumps(result),flush=True)
            if result['status']=='failed':errors.append(result)
    (root.parent/'acquisition_errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError('Minute quote acquisition failed; see sanitized acquisition_errors.json')


def load_panel(case,mapping,root):
    stock=normalize_quotes(pd.concat([pd.read_csv(root/(name+'.csv')) for name in mapping['stock_names']],ignore_index=True))
    stock=stock[stock.regular_session].drop_duplicates('sample_time_utc',keep='last').set_index('sample_time_utc').sort_index()
    option=normalize_quotes(pd.concat([pd.read_csv(root/(name+'.csv')) for name in mapping['option_names']],ignore_index=True))
    option=option[option.regular_session]
    panel=assemble_panel(stock,option,mapping['contract']).loc[:pd.Timestamp(case['exit'])].copy()
    bad=(~np.isfinite(panel.spot))|(panel.spot<=0)
    # Only a previously observed valid stock midpoint can mark unusable rows.
    # Their original execution flags remain false, so no fill uses the repair.
    if bad.any():
        panel.loc[bad,'spot']=panel.spot.where(panel.stock_passes_snapshot_filters).ffill().loc[bad]
        panel.loc[bad,'stock_passes_snapshot_filters']=False;panel.loc[bad,'all_valid']=False
    if not np.isfinite(panel.spot).all() or (panel.spot<=0).any():raise ValueError('No past stock mark for unusable row')
    return panel,int(bad.sum())


def run_case(case,mapping,plan,outstr):
    out=Path(outstr);folder=out/'cases'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'results.json'
    if checkpoint.exists():return str(checkpoint)
    root=ROOT/'data'/'near_continuous'/'quotes'
    panel,repairs=load_panel(case,mapping,root)
    day=panel.index.tz_convert('America/New_York').date
    diffs=panel.index.to_series().diff().dt.total_seconds().to_numpy()
    same_day=np.r_[False,day[1:]==day[:-1]]
    if not np.any(diffs[same_day]==60):raise AssertionError('Vendor did not supply actual one-minute intervals')
    if np.any(diffs[same_day]<60):raise AssertionError('Unexpected sub-minute samples')
    coarse=panel[panel.index.tz_convert('America/New_York').minute%15==0].copy()
    old=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/(case['case_id']+'_priced.csv'),index_col=0,parse_dates=True)
    old.index=pd.to_datetime(old.index,utc=True)
    common=coarse.index.intersection(old.index)
    price_cols=['spot','stock_bid','stock_ask','call_bid','call_ask','put_bid','put_ask']
    differences={col:float((coarse.loc[common,col]-old.loc[common,col]).abs().max()) for col in price_cols}
    daily=DailyInputs(ROOT/'data'/'stock_study',case['asset']);rates=RateInputs(ROOT/'data'/'stock_study'/'sofr.json')
    fine_cache=folder/'one_minute_priced.csv';coarse_cache=folder/'subsampled_15m_priced.csv'
    if fine_cache.exists():
        fine=pd.read_csv(fine_cache,index_col=0,parse_dates=True);fine.index=pd.to_datetime(fine.index,utc=True)
    else:
        fine=enrich_american(panel,mapping['contract']['strike'],daily,rates);fine.to_csv(fine_cache)
    if coarse_cache.exists():
        coarse=pd.read_csv(coarse_cache,index_col=0,parse_dates=True);coarse.index=pd.to_datetime(coarse.index,utc=True)
    else:
        coarse=enrich_american(coarse,mapping['contract']['strike'],daily,rates);coarse.to_csv(coarse_cache)
    contract=dict(mapping['contract']);contract['planned_exit']=case['exit']
    source_plan=json.loads((ROOT/'short_vol_plan.json').read_text())
    costs={'observed_costs':StockCosts(**source_plan['costs_observed']),
           'no_execution_cost':StockCosts(**source_plan['costs_midpoint'])}
    protocols=[('archived_15m',old.loc[:pd.Timestamp(case['exit'])],15,60,15),
               ('resampled_15m',coarse,15,60,15)]
    protocols.extend(('one_minute',fine,1,frequency,lag)
                     for lag in plan['signal_lags_minutes'] for frequency in plan['frequencies_minutes'])
    results=[]
    for label,p,grid,frequency,lag in protocols:
        group=[];orders={}
        for scenario,cost in costs.items():
            base={'case_id':case['case_id'],'data_source':label,'hedge_minutes':frequency,
                  'grid_minutes':grid,'signal_lag_minutes':lag,'cost_scenario':scenario}
            try:r,ledger,fills=replay(p,contract,daily,-1,frequency,cost,grid_minutes=grid,
                                    signal_lag_minutes=lag,allow_exercise_close=False)
            except Exception as exc:
                group.append({**base,'status':'replay_failed','error_type':type(exc).__name__,'reason':str(exc)})
                continue
            diagnostics=trade_diagnostics(p,contract,r,-1,fills);bridge=account(r,diagnostics)
            row={**base,**r,**diagnostics,**bridge,'status':'completed',
                 'exit_at_original_time':pd.Timestamp(r['exit'])==pd.Timestamp(case['exit'])}
            tag=f'{label}_{frequency}m_lag{lag}m_{scenario}'
            ledger.to_csv(folder/(tag+'_ledger.csv'),index=False);fills.to_csv(folder/(tag+'_fills.csv'),index=False)
            stock=fills[fills.instrument=='stock'];hedges=stock[pd.to_datetime(stock.time,utc=True)!=pd.Timestamp(r['exit'])]
            if len(hedges):
                elapsed=(pd.to_datetime(hedges.time,utc=True)-pd.to_datetime(hedges.decision_time,utc=True)).dt.total_seconds()
                if not np.all(elapsed==lag*60):raise AssertionError('Hedge used a different signal delay')
            row['stock_orders']=len(stock);row['stock_turnover_shares']=int(stock.quantity.abs().sum())
            orders[scenario]=fills[['time','instrument','quantity']].to_dict('records');group.append(row)
            if label=='archived_15m':
                reference=case['net_pnl'] if scenario=='observed_costs' else case['ideal_net_pnl']
                if abs(r['net_pnl']-reference)>1e-5:raise AssertionError('Generic engine failed archived cash reproduction')
        if all(r['status']=='completed' for r in group):
            same=orders['observed_costs']==orders['no_execution_cost']
            for r in group:r['same_orders_across_costs']=same
        results.extend(group)
    record={'case':case,'quality':{'fine_rows':len(fine),'coarse_rows':len(coarse),'old_overlap_rows':len(common),
        'nonpositive_stock_mark_repairs':repairs,'minute_grid_confirmed':True,'quote_differences_vs_archived_15m':differences,
        'fine_invalid_stock_rows':int((~fine.stock_passes_snapshot_filters).sum()),
        'fine_invalid_option_pairs':int((~fine.options_valid).sum()),
        'original_quote_tick_time_verified':False},'results':results}
    checkpoint.write_text(json.dumps(record,indent=2,default=str,allow_nan=True))
    return str(checkpoint)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);parser.add_argument('--acquire-only',action='store_true')
    args=parser.parse_args()
    plan=json.loads((ROOT/'near_continuous_plan.json').read_text());cases=cases_from_plan(plan)
    out=ROOT/'artifacts'/'near_continuous_study';(out/'cases').mkdir(parents=True,exist_ok=True)
    raw=ROOT/'data'/'near_continuous'/'quotes';raw.mkdir(parents=True,exist_ok=True)
    jobs,mapping=quote_jobs(cases);acquire(jobs,raw)
    if args.acquire_only:return
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sources=['near_continuous_study.py','highfreq_replay.py','stock_replay.py','american.py','stock_market.py','theta.py']
    config={'plan':plan,'source_hashes':{name:digest(ROOT/'volatility_lab'/name) for name in sources},
        'raw_hashes':{name:digest(raw/(name+'.csv')) for name in jobs},
        'daily_hashes':{c['asset']:digest(ROOT/'data'/'stock_study'/f'{c["asset"]}_daily_actions.csv') for c in cases},
        'sofr_hash':digest(ROOT/'data'/'stock_study'/'sofr.json'),
        'source_results_hash':digest(ROOT/'artifacts'/'short_loss_diagnostics'/'worst_unique_trades.csv'),
        'dependencies':{name:importlib.metadata.version(name) for name in ['pandas','numpy','QuantLib']}}
    cfg=out/'run_configuration.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Minute study inputs or numeric rules changed; preserve the run before recomputing')
    cfg.write_text(json.dumps(config,indent=2));paths=[];errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,c,mapping[c['case_id']],plan,str(out)):c['case_id'] for c in cases}
        for future in as_completed(futures):
            try:paths.append(future.result());print('Finished '+futures[future],flush=True)
            except Exception as exc:
                errors.append({'case_id':futures[future],'error_type':type(exc).__name__,'reason':str(exc)})
                print(json.dumps(errors[-1]),flush=True)
            (out/'errors.json').write_text(json.dumps(errors,indent=2))
    payloads=[json.loads(Path(p).read_text()) for p in paths]
    frame=pd.DataFrame([{k:v for k,v in r.items() if k!='cost_parameters'} for p in payloads for r in p['results']])
    frame.to_csv(out/'all_results.csv',index=False)
    (out/'quote_quality.json').write_text(json.dumps({p['case']['case_id']:p['quality'] for p in payloads},indent=2))
    if errors:raise RuntimeError('Minute replay errors occurred; no interpolated or fallback results were used')
    from .near_continuous_report import make_report
    make_report(out,plan)
    print(f'Report: {out/"report.html"}',flush=True)


if __name__=='__main__':main()
