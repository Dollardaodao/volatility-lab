"""Minute execution of newly selected monthly single-option long trades."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .theta import normalize_quotes
from .stock_market import DailyInputs, RateInputs, month_ranges
from .near_continuous_study import acquire
from .american import AmericanPricer
from .single_leg_replay import replay
from .stock_replay import StockCosts
from .monthly_single_leg_acquire import OUT, DATA

RAW = ROOT / 'data/near_continuous/quotes'


def quote_jobs(selected):
    jobs = {}; mapping = {}
    for c in selected:
        stocks = []; options = []
        for first,last in month_ranges(c['origin'],pd.Timestamp(c['expiry_date']).date()):
            name = f'{c["asset"]}_stock_1m_{first:%Y%m}'
            jobs[name] = ('stock_history_quote',{'symbol':c['asset'],'start_date':first,'end_date':last,'interval':'1m','venue':'utp_cta'})
            stocks.append(name)
            begin = max(first,pd.Timestamp(c['origin']).date()); end = min(last,pd.Timestamp(c['expiry_date']).date())
            both = f'{c["asset"]}_{c["expiry_date"]}_{c["strike"]:.3f}_1m_{begin}_{end}'
            name = both if (RAW/(both+'.csv')).exists() else f'{both}_{c["kind"]}'
            if name == both:
                params = {'symbol':c['asset'],'expiration':pd.Timestamp(c['expiry_date']).date(),'strike':f'{c["strike"]:.3f}',
                          'right':'both','start_date':begin,'end_date':end,'interval':'1m'}
            else:
                params = {'symbol':c['asset'],'expiration':pd.Timestamp(c['expiry_date']).date(),'strike':f'{c["strike"]:.3f}',
                          'right':c['kind'],'start_date':begin,'end_date':end,'interval':'1m'}
            jobs[name] = ('option_history_quote',params); options.append(name)
        mapping[c['case_id']] = {'stock':stocks,'option':options}
    return jobs,mapping


def load_panel(c, files):
    stock = normalize_quotes(pd.concat([pd.read_csv(RAW/(n+'.csv')) for n in files['stock']],ignore_index=True))
    stock = stock[stock.regular_session].drop_duplicates('sample_time_utc',keep='last').set_index('sample_time_utc').sort_index()
    option = normalize_quotes(pd.concat([pd.read_csv(RAW/(n+'.csv')) for n in files['option']],ignore_index=True))
    option = option[(option.right.str.upper()==c['kind'].upper())&option.regular_session]
    option = option.drop_duplicates('sample_time_utc',keep='last').set_index('sample_time_utc')
    columns = ['bid','ask','bid_size','ask_size','passes_snapshot_filters','conservative_condition_ok']
    panel = stock.rename(columns={k:'stock_'+k for k in columns})
    panel['spot'] = (panel.stock_bid+panel.stock_ask)/2
    panel = panel.join(option[columns].rename(columns={k:'option_'+k for k in columns}),how='left')
    # Keep quotes through expiry so a failed planned close can use a later
    # observed quote. Unresolved liquidations are retained as failures.
    panel = panel.loc[pd.Timestamp(c['decision_time']):pd.Timestamp(c['expiry_time'])].copy()
    panel['option_mid'] = (panel.option_bid+panel.option_ask)/2
    for k in ['stock_passes_snapshot_filters','option_passes_snapshot_filters','option_conservative_condition_ok']:
        panel[k] = panel[k].fillna(False).astype(bool)
    bad = (~np.isfinite(panel.spot))|(panel.spot<=0)
    repairs = int(bad.sum())
    if bad.any():
        panel.loc[bad,'spot'] = panel.spot.where(panel.stock_passes_snapshot_filters).ffill().loc[bad]
        panel.loc[bad,'stock_passes_snapshot_filters'] = False
    if not np.isfinite(panel.spot).all() or (panel.spot<=0).any(): raise ValueError('No past valid stock mark')
    panel['years_to_expiry'] = (pd.Timestamp(c['expiry_time'])-panel.index).total_seconds()/(365*86400)
    return panel,repairs


def enrich_single(panel,c,daily,rates):
    p = panel.copy(); days = p.index.tz_convert('America/New_York').date
    first_times = {d:t for d,t in zip(days,p.index)}
    rate_by_day = {d:rates.at(t) for d,t in first_times.items()}
    div_cash_by_day = {d:daily.yield_at(t,1.) for d,t in first_times.items()}
    p['rate'] = [rate_by_day[d] for d in days]
    p['dividend_yield'] = np.array([div_cash_by_day[d] for d in days])/p.spot.to_numpy()
    pricer = AmericanPricer(c['strike'],c['kind']); previous = None; previous_time = None
    ivs = []; deltas = []; carried = []
    for t,spot,years,r,q,mid,valid in zip(p.index,p.spot,p.years_to_expiry,p.rate,p.dividend_yield,p.option_mid,p.option_passes_snapshot_filters):
        if years <= 0:
            ivs.append(np.nan); deltas.append(np.nan); carried.append(False); continue
        pricer.configure(float(spot),float(years),float(r),float(q)); iv = np.nan; carry = False
        if valid:
            try: iv = pricer.implied(float(mid)); previous = iv; previous_time = t
            except (ValueError,RuntimeError): pass
        if not np.isfinite(iv) and previous is not None and (t-previous_time).total_seconds()<=4*86400:
            iv = previous; carry = True
        delta = np.nan
        if np.isfinite(iv):
            try: delta = pricer.delta(iv)
            except RuntimeError: pass
        ivs.append(iv); deltas.append(delta); carried.append(carry)
    p['option_iv'] = ivs; p['option_delta'] = deltas; p['iv_carried'] = carried
    return p


def diagnose(panel,c,result):
    entry = pd.Timestamp(result['entry']); exit_time = pd.Timestamp(result['exit']); held = panel.loc[entry:exit_time]
    coarse = held[held.index.tz_convert('America/New_York').minute%15==0]
    coarse = pd.concat([coarse,held.loc[[entry,exit_time]]]).sort_index(); coarse = coarse[~coarse.index.duplicated()]
    years = (exit_time-entry).total_seconds()/(365*86400)
    out = {'holding_days':(exit_time-entry).total_seconds()/86400,
           'holding_rv_1m':float(np.sqrt(np.square(np.diff(np.log(held.spot))).sum()/years)),
           'holding_rv_15m':float(np.sqrt(np.square(np.diff(np.log(coarse.spot))).sum()/years)),
           'vol_bet_status':'entry_iv_unavailable'}
    row = panel.loc[entry]; pricer = AmericanPricer(c['strike'],c['kind'])
    pricer.configure(float(row.spot),float(row.years_to_expiry),float(row.rate),float(row.dividend_yield))
    try: iv = pricer.implied(float(row.option_ask))
    except (ValueError,RuntimeError): return out
    return {**out,'entry_ask_iv':iv,'vol_bet_status':'scorable',
            'vol_bet_correct_15m':bool(out['holding_rv_15m']>iv),'vol_bet_correct_1m':bool(out['holding_rv_1m']>iv)}


def run_case(c,files,plan):
    folder = OUT/'trades'/c['case_id']; folder.mkdir(parents=True,exist_ok=True)
    checkpoint = folder/'results.json'
    sources={n:hashlib.sha256((ROOT/'volatility_lab'/n).read_bytes()).hexdigest() for n in
             ['monthly_single_leg_study.py','single_leg_replay.py','american.py','stock_market.py']}
    if checkpoint.exists():
        if json.loads(checkpoint.read_text()).get('numeric_source_hashes')!=sources:
            raise ValueError('Cached single-option results were produced by different numeric source')
        return str(checkpoint)
    daily = DailyInputs(ROOT/'data/stock_study',c['asset']); rates = RateInputs(ROOT/'data/stock_study/sofr.json')
    cache = folder/'priced.csv.gz'
    if cache.exists():
        panel = pd.read_csv(cache,index_col=0,parse_dates=True); panel.index = pd.to_datetime(panel.index,utc=True)
        repairs = int(json.loads((folder/'quality.json').read_text())['stock_mark_repairs'])
    else:
        panel,repairs = load_panel(c,files); panel = enrich_single(panel,c,daily,rates)
        panel.to_csv(cache,compression={'method':'gzip','compresslevel':1,'mtime':0})
    quality = {'minute_rows':len(panel),'invalid_stock_rows':int((~panel.stock_passes_snapshot_filters).sum()),
               'invalid_option_rows':int((~panel.option_passes_snapshot_filters).sum()),'stock_mark_repairs':repairs,
               'iv_carried_fraction':float(panel.iv_carried.mean()),'missing_delta_fraction':float(panel.option_delta.isna().mean()),
               'quote_tick_freshness_verified':False}
    (folder/'quality.json').write_text(json.dumps(quality,indent=2))
    costs = json.loads((ROOT/'short_vol_plan.json').read_text())
    results = []
    for frequency in plan['hedge_minutes']:
        group = []; orders = {}; observed_exit = None; observed_reason = None
        for scenario,name in [('observed_costs','costs_observed'),('no_execution_cost','costs_midpoint')]:
            base = {k:c[k] for k in ['case_id','asset','origin','target_dte','kind','strike','expiry_date','forecast_vol','decision_ask_iv','vol_gap']}
            base.update(hedge_minutes=frequency,signal_lag_minutes=plan['signal_lag_minutes'],cost_scenario=scenario)
            replay_contract = dict(c)
            if scenario == 'no_execution_cost' and observed_exit is not None:
                # The idealized comparison attributes costs on the actual
                # strategy's holding period, including a cost-driven reserve
                # exit. It is not an independently selected free-fee policy.
                replay_contract['planned_exit'] = observed_exit
            try:
                r,ledger,fills = replay(panel,replay_contract,daily,frequency,StockCosts(**costs[name]),
                                        grid_minutes=plan['quote_grid_minutes'],signal_lag_minutes=plan['signal_lag_minutes'])
            except (ValueError,AssertionError) as exc:
                rejected = isinstance(exc,ValueError) and str(exc) in ['Invalid observed entry quotes','Missing decision/entry grid','Missing decision Delta']
                group.append({**base,'status':'entry_rejected' if rejected else 'replay_failed',
                              'error_type':type(exc).__name__,'reason':str(exc),**({'side':0,'net_pnl':0.} if rejected else {})})
                continue
            if scenario == 'observed_costs':observed_exit = r['exit']; observed_reason = r['exit_reason']
            diag = diagnose(panel,c,r); tag = f'{frequency}m_{scenario}'
            compression={'method':'gzip','compresslevel':1,'mtime':0}
            fills.to_csv(folder/(tag+'_fills.csv.gz'),index=False,compression=compression)
            ledger.to_csv(folder/(tag+'_ledger.csv.gz'),index=False,compression=compression)
            orders[scenario] = fills[['time','instrument','quantity']].to_dict('records')
            group.append({**base,**r,**diag,'status':'completed',
                          'counterfactual_exit_frozen_to_observed':scenario=='no_execution_cost',
                          'observed_exit_reason':observed_reason,
                          'exit_at_planned_time':pd.Timestamp(r['exit'])==pd.Timestamp(c['planned_exit'])})
        if len(orders)==2:
            for row in group: row['same_orders_across_costs'] = orders['observed_costs']==orders['no_execution_cost']
        results.extend(group)
    temporary=checkpoint.with_suffix('.tmp')
    temporary.write_text(json.dumps({'contract':c,'quality':quality,'results':results,'numeric_source_hashes':sources},indent=2,default=str))
    temporary.replace(checkpoint)
    return str(checkpoint)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);parser.add_argument('--acquire-only',action='store_true')
    args=parser.parse_args();plan=json.loads((ROOT/'monthly_single_leg_plan.json').read_text())
    errors=json.loads((OUT/'selection_errors.json').read_text())
    if errors: raise RuntimeError('Complete monthly selection before replay; failed stock-months cannot be dropped')
    decisions=sorted(json.loads((OUT/'decisions.json').read_text()),key=lambda c:(c['asset'],c['origin'],c['target_dte']))
    assert len(decisions)==20*48*3
    (OUT/'decisions.json').write_text(json.dumps(decisions,indent=2,default=str))
    selected=[c for c in decisions if c['status']=='selected']
    for c in selected: c['case_id']=f'{c["asset"]}_{c["origin"]}_{c["target_dte"]}d_{c["kind"]}'
    jobs,mapping=quote_jobs(selected); RAW.mkdir(parents=True,exist_ok=True)
    print(f'{len(decisions)} monthly candidates; {len(selected)} selected single options; {len(jobs)} quote files '
          f'({sum(not (RAW/(n+".csv")).exists() for n in jobs)} missing)',flush=True)
    acquire(jobs,RAW)
    (OUT/'minute_acquisition_manifest.json').write_text(json.dumps({n:{'method':m,'parameters':p} for n,(m,p) in jobs.items()},indent=2,default=str))
    if args.acquire_only:return
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=['monthly_single_leg_study.py','single_leg_replay.py','stock_replay.py','american.py','forecast.py','stock_market.py','theta.py']
    config={'plan':plan,'source_hashes':{n:digest(ROOT/'volatility_lab'/n) for n in sources},
            'selection_configuration_sha256':digest(OUT/'selection_configuration.json'),
            'decisions_sha256':digest(OUT/'decisions.json'),'cost_configuration_sha256':digest(ROOT/'short_vol_plan.json'),
            'raw_hashes':{n:digest(RAW/(n+'.csv')) for n in jobs},
            'dependencies':{n:importlib.metadata.version(n) for n in ['pandas','numpy','QuantLib','arch','scipy']}}
    cfg=OUT/'replay_configuration.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Replay inputs or rules changed; preserve existing experiment')
    cfg.write_text(json.dumps(config,indent=2)); paths=[];failed=[]
    # Start longer holdings first to keep independent pricing workers busy.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,c,mapping[c['case_id']],plan):c['case_id'] for c in sorted(selected,key=lambda x:-x['target_dte'])}
        for future in as_completed(futures):
            case=futures[future]
            try:
                paths.append(future.result())
                if len(paths)%10==0:print(f'Replayed {len(paths)}/{len(selected)} selected contracts',flush=True)
            except Exception as exc:
                failed.append({'case_id':case,'error_type':type(exc).__name__});print(json.dumps(failed[-1]),flush=True)
            (OUT/'replay_errors.json').write_text(json.dumps(failed,indent=2))
    payloads=[json.loads(Path(p).read_text()) for p in paths]
    pd.DataFrame([{k:v for k,v in r.items() if k!='cost_parameters'} for p in payloads for r in p['results']]).to_csv(OUT/'all_results.csv',index=False)
    if failed:raise RuntimeError('Numerical/acquisition failures retained, not silently excluded')
    from .monthly_single_leg_report import make_report
    make_report(OUT)
    print(f'Report: {OUT/"report.html"}',flush=True)


if __name__=='__main__':main()
