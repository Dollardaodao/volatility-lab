"""Longer hedge clocks on all original monthly single-option entries."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .stock_market import DailyInputs
from .stock_replay import StockCosts
from .scheduled_single_leg_replay import replay
from .single_leg_counterfactual import midpoint_book
from .monthly_single_leg_study import diagnose

OUT=ROOT/'artifacts/longer_hedge_study'
SOURCE=ROOT/'artifacts/monthly_single_leg'
NUMERIC_SOURCES=['longer_hedge_study.py','scheduled_single_leg_replay.py','single_leg_replay.py',
                 'single_leg_counterfactual.py','stock_replay.py','stock_market.py','monthly_single_leg_study.py','american.py']


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_case(case_id,plan,source_hashes):
    source=SOURCE/'trades'/case_id;out=OUT/'trades'/case_id;out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'results.json'
    if checkpoint.exists():
        if json.loads(checkpoint.read_text())['numeric_source_hashes']!=source_hashes:raise ValueError('Incompatible cached schedule results')
        return str(checkpoint)
    original=json.loads((source/'results.json').read_text());c=dict(original['contract'])
    old_actual=next(r for r in original['results'] if r['hedge_minutes']==60 and r['cost_scenario']=='observed_costs')
    old_ideal=next(r for r in original['results'] if r['hedge_minutes']==60 and r['cost_scenario']=='no_execution_cost')
    assert old_actual['status']=='completed' and old_ideal['status']=='completed'
    c['planned_exit']=old_actual['exit']
    panel=pd.read_csv(source/'priced.csv.gz',index_col=0,parse_dates=True);panel.index=pd.to_datetime(panel.index,utc=True)
    daily=DailyInputs(ROOT/'data/stock_study',c['asset']);costs=StockCosts(**old_actual['cost_parameters'])
    rows=[];reproduction={}
    diagnostic_names=['holding_days','holding_rv_1m','holding_rv_15m','entry_ask_iv','vol_bet_status','vol_bet_correct_15m','vol_bet_correct_1m']
    for protocol in plan['protocols']:
        base={k:c[k] for k in ['case_id','asset','origin','target_dte','kind','strike','expiry_date','forecast_vol','decision_ask_iv','vol_gap']}
        base.update(protocol_id=protocol['id'],protocol_label=protocol['label'],hedge_minutes=protocol['frequency'],
                    schedule_kind=protocol['schedule_kind'],signal_lag_minutes=plan['signal_lag_minutes'])
        actual,ledger,fills=replay(panel,c,daily,protocol['frequency'],costs,
            schedule_kind=protocol['schedule_kind'],daily_time=protocol.get('daily_time','10:00'),
            grid_minutes=plan['quote_grid_minutes'],signal_lag_minutes=plan['signal_lag_minutes'])
        ideal,ideal_ledger,ideal_fills=midpoint_book(panel,actual,ledger,fills,costs)
        at_original=pd.Timestamp(actual['exit'])==pd.Timestamp(old_actual['exit'])
        diagnostics={k:old_actual[k] for k in diagnostic_names if k in old_actual} if at_original else diagnose(panel,c,actual)
        extra={'status':'completed','exit_at_original_time':at_original,'same_orders_across_costs':True,
               'counterfactual_exit_frozen_to_observed':True}
        active=ledger[ledger.option_contracts!=0]
        spot=panel.loc[pd.to_datetime(active.time,utc=True),'spot'].to_numpy()
        extra['mean_abs_delta_dollars']=float((active.delta_shares.abs().to_numpy()*spot).mean())
        adj=fills[(fills.instrument=='stock')&(pd.to_datetime(fills.time,utc=True)!=pd.Timestamp(actual['exit']))]
        extra['non_liquidation_stock_orders']=len(adj)
        extra['stock_order_days']=int(pd.to_datetime(adj.time,utc=True).dt.tz_convert('America/New_York').dt.date.nunique())
        for label,result,book,orders in [('observed_costs',actual,ledger,fills),('no_execution_cost',ideal,ideal_ledger,ideal_fills)]:
            tag=protocol['id']+'_'+label
            compression={'method':'gzip','compresslevel':1,'mtime':0}
            book.to_csv(out/(tag+'_ledger.csv.gz'),index=False,compression=compression)
            orders.to_csv(out/(tag+'_fills.csv.gz'),index=False,compression=compression)
            rows.append({**base,**result,**diagnostics,**extra,'cost_scenario':label})
        if protocol['id']=='elapsed_60m':
            observed_difference=actual['net_pnl']-old_actual['net_pnl'];ideal_difference=ideal['net_pnl']-old_ideal['net_pnl']
            assert abs(observed_difference)<1e-5 and abs(ideal_difference)<1e-5,'Old hourly cash changed'
            old_fills=pd.read_csv(source/'60m_observed_costs_fills.csv.gz')
            for frame in [old_fills,fills]:frame['time']=pd.to_datetime(frame.time,utc=True)
            assert old_fills[['time','instrument','quantity']].to_dict('records')==fills[['time','instrument','quantity']].to_dict('records')
            reproduction={'observed_cash_difference':observed_difference,'ideal_cash_difference':ideal_difference,'same_orders':True}
    temporary=checkpoint.with_suffix('.tmp');temporary.write_text(json.dumps({'contract':c,'results':rows,
        'numeric_source_hashes':source_hashes,'hourly_reproduction':reproduction},indent=2,default=str));temporary.replace(checkpoint)
    return str(checkpoint)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);args=parser.parse_args()
    plan=json.loads((ROOT/'longer_hedge_plan.json').read_text());OUT.mkdir(exist_ok=True)
    old_verification=json.loads((SOURCE/'verification.json').read_text());assert old_verification['status']=='passed'
    assert digest(SOURCE/'all_results.csv')==old_verification['numeric_results_sha256']
    cases=sorted(p.parent.name for p in (SOURCE/'trades').glob('*/results.json'));assert len(cases)==plan['selected_trades']
    hashes={n:digest(ROOT/'volatility_lab'/n) for n in NUMERIC_SOURCES}
    config={'plan':plan,'source_hashes':hashes,'original_results_sha256':digest(SOURCE/'all_results.csv'),
            'original_verification_sha256':digest(SOURCE/'verification.json'),
            'priced_panel_hashes':{name:digest(SOURCE/'trades'/name/'priced.csv.gz') for name in cases},
            'original_case_hashes':{name:digest(SOURCE/'trades'/name/'results.json') for name in cases},
            'daily_hashes':{s:digest(ROOT/'data/stock_study'/f'{s}_daily_actions.csv') for s in {json.loads((SOURCE/'trades'/n/'results.json').read_text())['contract']['asset'] for n in cases}},
            'dependencies':{n:importlib.metadata.version(n) for n in ['pandas','numpy','QuantLib']}}
    cfg=OUT/'configuration.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Schedule study inputs changed; preserve prior run')
    cfg.write_text(json.dumps(config,indent=2));paths=[];errors=[]
    lengths={n:json.loads((SOURCE/'trades'/n/'results.json').read_text())['contract']['target_dte'] for n in cases}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,name,plan,hashes):name for name in sorted(cases,key=lambda n:-lengths[n])}
        for future in as_completed(futures):
            try:
                paths.append(future.result())
                if len(paths)%25==0:print(f'Finished {len(paths)}/{len(cases)} single-option trades across all schedules',flush=True)
            except Exception as exc:
                errors.append({'case_id':futures[future],'error_type':type(exc).__name__,'reason':str(exc)});print(json.dumps(errors[-1]),flush=True)
            (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    payloads=[json.loads(Path(p).read_text()) for p in paths]
    frame=pd.DataFrame([{k:v for k,v in r.items() if k!='cost_parameters'} for p in payloads for r in p['results']]);frame.to_csv(OUT/'all_results.csv',index=False)
    pd.DataFrame([{'case_id':p['contract']['case_id'],**p['hourly_reproduction']} for p in payloads]).to_csv(OUT/'hourly_reproduction.csv',index=False)
    if errors:raise RuntimeError('Schedule cases failed; no missing trade may be hidden')
    from .longer_hedge_report import make_report
    make_report(OUT,plan)
    print(f'Report: {OUT/"report.html"}',flush=True)


if __name__=='__main__':main()
