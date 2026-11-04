"""Locked chronological holdout: five past-only forecasts, one-leg option books.

The forecast can change whether a cohort enters, but not which candidate wins:
within a cohort forecast minus ask-IV is maximized by the same minimum ask-IV.
An identical contract's market-IV hedge book is therefore shared exactly.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from .theta_sdk import ROOT, connect
from .stock_market import DailyInputs, RateInputs, MarketStore, local_time
from .monthly_single_leg_acquire import origins, select_candidates
from .monthly_single_leg_study import quote_jobs, load_panel, enrich_single, diagnose, RAW
from .ml_vol import StockDataset, fit_models
from .forecast import fit_variance, expiry_forecast
from .ml_study import score_forecast
from .theta import normalize_quotes
from .scheduled_single_leg_replay import replay
from .single_leg_counterfactual import midpoint_book
from .stock_replay import StockCosts

OUT = ROOT/'artifacts/oos_single_leg'
DATA = ROOT/'data/oos_single_leg/decision_quotes'
SOURCES = ['oos_single_leg.py', 'forecast.py', 'ml_vol.py', 'ml_study.py',
           'monthly_single_leg_acquire.py', 'monthly_single_leg_study.py',
           'scheduled_single_leg_replay.py', 'single_leg_counterfactual.py',
           'stock_replay.py', 'stock_market.py', 'american.py', 'theta.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def lock(plan):
    assert digest(ROOT/'oos_single_leg_plan.json') == json.loads((OUT/'protocol_lock.json').read_text())['plan_sha256']
    configuration = {'plan': plan, 'source_hashes': {s: digest(ROOT/'volatility_lab'/s) for s in SOURCES},
        'daily_hashes': {s: digest(ROOT/'data/stock_study'/f'{s}_daily_actions.csv') for s in plan['universe']},
        'cost_plan_sha256': digest(ROOT/'short_vol_plan.json'),
        'sofr_sha256': digest(ROOT/'data/stock_study/sofr.json'),
        'dependencies': {s: importlib.metadata.version(s) for s in ['numpy','pandas','arch','scikit-learn','QuantLib']}}
    path = OUT/'configuration.json'
    if path.exists() and json.loads(path.read_text()) != configuration:
        raise ValueError('Locked numeric source or inputs changed; preserve the experiment')
    save(path, configuration)
    return configuration['source_hashes']


def forecast_month(origin, plan):
    path = OUT/'months'/f'{origin}.json'
    if path.exists(): return str(path)
    training = plan['training']
    dataset = StockDataset(ROOT/'data/stock_study', plan['universe'], training['forward_variance_blocks'])
    decision = local_time(origin, '09:45')
    cutoff = dataset.daily[plan['universe'][0]].prices.loc[:decision].index[-1]
    models, audit = fit_models(dataset, cutoff, training)
    joblib.dump(models, OUT/'models'/f'{origin}.joblib')
    forecasts = {}; scores = []
    for symbol in plan['universe']:
        daily = dataset.daily[symbol]; feature_time = daily.prices.loc[:decision].index[-1]
        feature = dataset.observed_features(symbol, cutoff)
        fits = {name: m.curve(feature, symbol, feature_time) for name,m in models.items()}
        for name in ('garch_t','ewma','rolling21'):
            fits[name] = fit_variance(daily.prices, decision, horizon=plan['forecast_horizon_sessions'],
                                     window=plan['model_window_returns'], model=name)
        forecasts[symbol] = {name: {'trained_through':str(f.trained_through),
            'daily_variance':f.daily_variance.tolist(), 'diagnostics':f.diagnostics} for name,f in fits.items()}
        # No outcome enters any fit or feature. Score only after the five curves exist.
        position = daily.prices.index.get_loc(feature_time)
        for horizon in plan['forecast_scoring_horizons_sessions']:
            expiry = daily.prices.index[position+horizon]
            for name, f in fits.items():
                prediction = expiry_forecast(f, decision, expiry, 'equity')
                scores.append({'asset':symbol,'origin':origin,'model':name,'horizon_sessions':horizon,
                    'feature_time':str(feature_time),'label_end':str(expiry),
                    'annualized_vol':prediction['annualized_vol'],
                    'integrated_variance':prediction['integrated_variance'],
                    **score_forecast(daily.prices,feature_time,decision,expiry,prediction)})
    save(path, {'origin':origin,'training_audit':audit,'forecasts':forecasts,'evaluations':scores})
    return str(path)


def forecasts(plan, workers):
    for name in ['months','models']: (OUT/name).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(forecast_month,o,plan):o for o in origins(plan)}
        for future in as_completed(futures):
            future.result(); print('Forecasts locked: '+futures[future], flush=True)
    rows = [r for o in origins(plan) for r in json.loads((OUT/'months'/f'{o}.json').read_text())['evaluations']]
    pd.DataFrame(rows).to_csv(OUT/'forecast_predictions.csv', index=False)


def download(jobs, directory, manifest_name):
    directory.mkdir(parents=True, exist_ok=True)
    missing = [(n,m,p) for n,(m,p) in jobs.items() if not (directory/(n+'.csv')).exists()]
    print(f'{manifest_name}: {len(jobs)} files, {len(missing)} missing', flush=True)
    if not missing: return
    client = None
    for attempt in range(3):
        try: client = connect(); break
        except Exception as exc:
            print(json.dumps({'stage':'authentication','attempt':attempt+1,'error_type':type(exc).__name__}),flush=True)
            if attempt < 2: time.sleep(2)
    if client is None: raise RuntimeError('Authentication unavailable; no raw response logged')
    store = MarketStore(directory, client)
    def fetch(job):
        name, method, params = job
        error = {}
        for attempt in range(3):
            try:
                frame = store.fetch(name, method, **params)
                return {'name':name,'status':'acquired','rows':len(frame)}
            except Exception as exc:
                error = {'name':name,'status':'failed','error_type':type(exc).__name__}
                if callable(getattr(exc,'code',None)): error['grpc_status'] = str(exc.code())
                if error.get('grpc_status') == 'StatusCode.INVALID_ARGUMENT': break
                if attempt < 2: time.sleep([1,3][attempt])
        return error
    errors = []; done = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fetch,j) for j in missing]):
            result = future.result(); done += 1
            if result['status']=='failed': errors.append(result); print(json.dumps(result),flush=True)
            if done%20==0: print(f'{manifest_name}: {done}/{len(missing)} attempted, {len(errors)} failed',flush=True)
            save(OUT/(manifest_name+'_errors.json'),errors)
    if errors: raise RuntimeError('Incomplete acquisition; all failures retained, no cohorts dropped')


def decision_jobs(plan):
    jobs = {}
    for origin in origins(plan):
        for symbol in plan['universe']:
            jobs[f'{symbol}_{origin}_stock_decision'] = ('stock_history_quote', {
                'symbol':symbol,'date':pd.Timestamp(origin).date(),'interval':'15m',
                'start_time':'09:45:00','end_time':'09:46:00','venue':'utp_cta'})
            jobs[f'{symbol}_{origin}_full_origin_snapshot'] = ('option_history_quote', {
                'symbol':symbol,'date':pd.Timestamp(origin).date(),'expiration':'*','interval':'15m',
                'start_time':'09:45:00','end_time':'09:46:00','max_dte':115})
    return jobs


def selection(plan):
    from .forecast import Fit
    rates = RateInputs(ROOT/'data/stock_study/sofr.json'); rows = []; raw_hashes = {}
    for origin in origins(plan):
        month = json.loads((OUT/'months'/f'{origin}.json').read_text())
        for symbol in plan['universe']:
            daily = DailyInputs(ROOT/'data/stock_study',symbol)
            stock_file = DATA/f'{symbol}_{origin}_stock_decision.csv'
            option_file = DATA/f'{symbol}_{origin}_full_origin_snapshot.csv'
            raw_hashes.update({str(p.relative_to(ROOT)):digest(p) for p in [stock_file,option_file]})
            stock = normalize_quotes(pd.read_csv(stock_file))
            stock = stock[stock.sample_time_utc==local_time(origin,'09:45')]
            if len(stock)!=1 or not bool(stock.iloc[0].passes_snapshot_filters):
                raise ValueError('Invalid decision stock; retained unresolved cohort '+symbol+' '+origin)
            spot = float((stock.iloc[0].bid+stock.iloc[0].ask)/2)
            snapshot = normalize_quotes(pd.read_csv(option_file))
            for name in plan['models']:
                info = month['forecasts'][symbol][name]
                fit = Fit(name,pd.Timestamp(info['trained_through']),np.array(info['daily_variance']),info['diagnostics'])
                rule = {**plan,'model':name}
                selected = select_candidates(snapshot,spot,origin,daily,rates,rule,fit)
                for c in selected:
                    c.update(asset=symbol,model=name,opportunity_id=f'{symbol}_{origin}_{c["target_dte"]}d')
                    if 'kind' in c: c['case_id']=c['opportunity_id']+'_'+c['kind']
                    rows.append(c)
        print('Selected all models: '+origin,flush=True)
    assert len(rows)==20*7*3*5
    save(OUT/'decisions.json',rows)
    pd.DataFrame([{k:v for k,v in c.items() if k not in ['candidates','fit_diagnostics']} for c in rows]).to_csv(OUT/'decisions.csv',index=False)
    save(OUT/'decision_input_hashes.json',raw_hashes)
    print(pd.crosstab(pd.DataFrame(rows).model,pd.DataFrame(rows).status).to_string(),flush=True)


def selected_union():
    rows = json.loads((OUT/'decisions.json').read_text()); unique = {}
    for c in rows:
        if c['status']=='selected':
            if c['case_id'] in unique:
                assert all(unique[c['case_id']][k]==c[k] for k in ['kind','strike','expiry_date','decision_ask_iv'])
            unique.setdefault(c['case_id'],c)
    return list(unique.values())


def run_case(c, files, plan, hashes):
    folder = OUT/'trades'/c['case_id']; folder.mkdir(parents=True,exist_ok=True)
    checkpoint = folder/'results.json'
    if checkpoint.exists():
        assert json.loads(checkpoint.read_text())['numeric_source_hashes']==hashes
        return str(checkpoint)
    daily = DailyInputs(ROOT/'data/stock_study',c['asset']); rates = RateInputs(ROOT/'data/stock_study/sofr.json')
    cache = folder/'priced.csv.gz'; compression = {'method':'gzip','compresslevel':1,'mtime':0}
    if cache.exists():
        panel = pd.read_csv(cache,index_col=0,parse_dates=True); panel.index=pd.to_datetime(panel.index,utc=True)
        repairs = json.loads((folder/'quality.json').read_text())['stock_mark_repairs']
    else:
        panel, repairs = load_panel(c,files); panel = enrich_single(panel,c,daily,rates)
        panel.to_csv(cache,compression=compression)
        save(folder/'quality.json',{'minute_rows':len(panel),'stock_mark_repairs':repairs,
            'invalid_stock_rows':int((~panel.stock_passes_snapshot_filters).sum()),
            'invalid_option_rows':int((~panel.option_passes_snapshot_filters).sum()),
            'missing_delta_fraction':float(panel.option_delta.isna().mean()),
            'iv_carried_fraction':float(panel.iv_carried.mean()),'quote_tick_freshness_verified':False})
    costs = StockCosts(**json.loads((ROOT/'short_vol_plan.json').read_text())['costs_observed'])
    rows = []
    for protocol in plan['protocols']:
        try:
            actual, ledger, fills = replay(panel,c,daily,protocol['frequency'],costs,
                schedule_kind=protocol['schedule_kind'],grid_minutes=1,signal_lag_minutes=1)
        except ValueError as exc:
            if str(exc) not in ['Invalid observed entry quotes','Missing decision/entry grid','Missing decision Delta']: raise
            for scenario in ['observed_costs','no_execution_cost']:
                rows.append({'case_id':c['case_id'],'protocol_id':protocol['id'],'cost_scenario':scenario,
                    'status':'entry_rejected','reason':str(exc),'net_pnl':0.})
            continue
        ideal, ideal_ledger, ideal_fills = midpoint_book(panel,actual,ledger,fills,costs)
        diagnostics = diagnose(panel,c,actual)
        extra = {'exit_at_planned_time':pd.Timestamp(actual['exit'])==pd.Timestamp(c['planned_exit']),
                 'same_orders_across_costs':True}
        for scenario, result, book, orders in [('observed_costs',actual,ledger,fills),
                                              ('no_execution_cost',ideal,ideal_ledger,ideal_fills)]:
            tag = protocol['id']+'_'+scenario
            book.to_csv(folder/(tag+'_ledger.csv.gz'),index=False,compression=compression)
            orders.to_csv(folder/(tag+'_fills.csv.gz'),index=False,compression=compression)
            rows.append({'case_id':c['case_id'],'protocol_id':protocol['id'],'cost_scenario':scenario,
                         'status':'completed',**result,**diagnostics,**extra})
    save(checkpoint,{'contract':c,'results':rows,'numeric_source_hashes':hashes})
    return str(checkpoint)


def trading(plan, workers, hashes):
    selected = selected_union(); jobs, mapping = quote_jobs(selected)
    download(jobs,RAW,'minute_download')
    save(OUT/'minute_manifest.json',{'jobs':{n:{'method':m,'parameters':p,'sha256':digest(RAW/(n+'.csv'))}
                                           for n,(m,p) in jobs.items()},'mapping':mapping})
    print(f'{len(selected)} distinct selected single-option cohorts',flush=True)
    errors = []; paths = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_case,c,mapping[c['case_id']],plan,hashes):c['case_id']
                   for c in sorted(selected,key=lambda c:-c['target_dte'])}
        for future in as_completed(futures):
            try:
                paths.append(future.result())
                if len(paths)%10==0:print(f'Replayed {len(paths)}/{len(selected)} distinct cohorts',flush=True)
            except Exception as exc:
                errors.append({'case_id':futures[future],'error_type':type(exc).__name__,'reason':str(exc)})
                print(json.dumps(errors[-1]),flush=True)
            save(OUT/'replay_errors.json',errors)
    if errors: raise RuntimeError('Unresolved replay errors; comparison cannot omit them')
    results = [r for p in paths for r in json.loads(Path(p).read_text())['results']]
    pd.DataFrame([{k:v for k,v in r.items() if k!='cost_parameters'} for r in results]).to_csv(OUT/'unique_trade_results.csv',index=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage',choices=['forecasts','decisions','select','trade','all'])
    parser.add_argument('--workers',type=int,default=4); args=parser.parse_args()
    plan = json.loads((ROOT/'oos_single_leg_plan.json').read_text()); hashes = lock(plan)
    if args.stage in ['forecasts','all']: forecasts(plan,args.workers)
    if args.stage in ['decisions','all']: download(decision_jobs(plan),DATA,'decision_download')
    if args.stage in ['select','all']: selection(plan)
    if args.stage in ['trade','all']: trading(plan,args.workers,hashes)


if __name__=='__main__': main()
