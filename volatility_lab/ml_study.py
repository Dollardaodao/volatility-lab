"""Monthly walk-forward ML variance forecasts and unchanged option replay."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .ml_vol import StockDataset,fit_models
from .forecast import fit_variance,expiry_forecast
from .american import AmericanPricer
from .stock_market import sessions,local_time
from .stock_study import select_side
from .stock_replay import StockCosts,replay

_DATA=None;_PLAN=None;_OUT=None;_CONTRACTS=None;_BASE=None


def initialize(plan,out):
    global _DATA,_PLAN,_OUT,_CONTRACTS,_BASE
    _PLAN=plan;_OUT=Path(out)
    _DATA=StockDataset(ROOT/'data'/'stock_study',plan['universe'],plan['forward_variance_blocks'])
    manifest=json.loads((ROOT/'artifacts'/'stock_study_acquisition_20.json').read_text())
    _CONTRACTS=[c for c in manifest['cohorts'] if c['status']=='acquired']
    base=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/'all_results.csv',low_memory=False)
    base=base[(base.policy=='garch_long_flat')&(base.hedge_minutes==60)&(base.cost_scenario=='zero_bid_writedown')]
    _BASE=base.set_index(['asset','origin','target_dte'])


def score_forecast(prices,feature_time,decision,expiry,prediction):
    # This label function runs after forecasts have been generated. Training
    # receives none of these future returns or option-market outcomes.
    segment=prices.loc[feature_time:expiry]
    returns=np.diff(np.log(segment.to_numpy()))
    ends=segment.index[1:];starts=segment.index[:-1]
    overlap=np.maximum(0,(np.minimum(ends.asi8,expiry.value)-np.maximum(starts.asi8,decision.value))/1e9)
    weights=overlap/((ends.asi8-starts.asi8)/1e9)
    realized=float(np.dot(weights,returns**2))
    years=(expiry-decision).total_seconds()/(365*86400)
    actual_vol=np.sqrt(realized/years);ratio=realized/prediction['integrated_variance']
    return {'realized_daily_proxy_variance':realized,'realized_daily_proxy_vol':actual_vol,
            'qlike':ratio-np.log(max(ratio,1e-15))-1,
            'squared_vol_error':(prediction['annualized_vol']-actual_vol)**2,
            'abs_vol_error':abs(prediction['annualized_vol']-actual_vol)}


def trade_contract(contract,fits,decision,model_audit):
    stem=f'{contract["asset"]}_{contract["origin"]}_{contract["target_dte"]}d'
    panel=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/(stem+'_priced.csv'),index_col=0,parse_dates=True)
    panel.index=pd.to_datetime(panel.index,utc=True)
    row=panel.loc[decision];expiry=pd.Timestamp(contract['expiry_time'])
    original=_BASE.loc[(contract['asset'],contract['origin'],contract['target_dte'])]
    for name in ('garch_t','ewma'):
        value=expiry_forecast(fits[name],decision,expiry,'equity')['annualized_vol']
        if abs(value-float(original[name+'_forecast_vol']))>1e-9:
            raise AssertionError('Original baseline forecast was not reproduced')
    results=[]
    for name in ('har_proxy_ridge','ml_hist_gradient_boosting'):
        forecast=expiry_forecast(fits[name],decision,expiry,'equity')
        vol=forecast['annualized_vol'];fair=0.
        for kind in ('call','put'):
            p=AmericanPricer(contract['strike'],kind)
            p.configure(float(row.spot),float(row.years_to_expiry),float(row.rate),float(row.dividend_yield))
            fair+=p.value(vol)
        # Reuse the original cost/vol gate exactly. Only these three
        # decision-time inputs are passed; no future labels enter the gate.
        values={'garch_t_forecast_vol':vol,'garch_t_fair_sum':fair,
                'long_execution_iv':float(original.long_execution_iv),
                'short_execution_iv':float(original.short_execution_iv)}
        side,hurdle=select_side('garch_long_flat',values,row,contract,60,.03)
        base={**contract,'model':name,'policy':name+'_long_flat','forecast_vol':vol,
              'feature_time':str(fits[name].trained_through),'model_fit_cutoff':model_audit['fit_cutoff'],
              'long_execution_iv':float(original.long_execution_iv),'side':side,'hedge_minutes':60,**hurdle}
        for scenario in ('base','zero_bid_writedown'):
            costs=replace(StockCosts(),allow_unbid_long_abandonment=scenario=='zero_bid_writedown')
            try:
                r,ledger,fills=replay(panel,contract,_DATA.daily[contract['asset']],side,60,costs)
                extra={}
                if side:
                    entry,exit=pd.Timestamp(r['entry']),pd.Timestamp(r['exit']);held=panel.loc[entry:exit]
                    rv=np.sqrt(np.square(np.diff(np.log(held.spot.to_numpy()))).sum()/((exit-entry).total_seconds()/(365*86400)))
                    er=held.iloc[0];ivs=[]
                    for kind in ('call','put'):
                        p=AmericanPricer(contract['strike'],kind)
                        p.configure(float(er.spot),float(er.years_to_expiry),float(er.rate),float(er.dividend_yield))
                        ivs.append(p.implied(float(er[kind+'_ask'])))
                    extra={'holding_rv':rv,'entry_execution_ask_iv':max(ivs),'vol_bet_correct':bool(rv>max(ivs))}
                    tag=f'{stem}_{name}_{scenario}'
                    ledger.to_csv(_OUT/'trades'/(tag+'_ledger.csv'),index=False)
                    fills.to_csv(_OUT/'trades'/(tag+'_fills.csv'),index=False)
                results.append({**base,**r,**extra,'status':'completed','cost_scenario':scenario})
            except Exception as exc:
                results.append({**base,'status':'replay_failed','cost_scenario':scenario,
                                'error_type':type(exc).__name__,'reason':str(exc)})
    return results


def run_month(origin):
    checkpoint=_OUT/'months'/(origin+'.json')
    if checkpoint.exists():return str(checkpoint)
    decision=local_time(origin,'09:45')
    cutoff=_DATA.daily[_PLAN['universe'][0]].prices.loc[:decision].index[-1]
    models,audit=fit_models(_DATA,cutoff,_PLAN)
    joblib.dump(models,_OUT/'models'/(origin+'.joblib'))
    evaluations=[];trades=[];forecast_failures=[]
    for symbol in _PLAN['universe']:
        daily=_DATA.daily[symbol];observed=daily.prices.loc[:decision]
        feature_time=observed.index[-1]
        feature=_DATA.observed_features(symbol,decision)
        fits={name:model.curve(feature,symbol,feature_time) for name,model in models.items()}
        for name in ('garch_t','ewma','rolling21'):
            try:fits[name]=fit_variance(daily.prices,decision,horizon=84,model=name)
            except Exception as exc:
                forecast_failures.append({'asset':symbol,'origin':origin,'model':name,
                    'error_type':type(exc).__name__,'reason':str(exc)})
        position=daily.prices.index.get_loc(feature_time)
        # All predictions are generated before any future score is accessed.
        predictions=[]
        for horizon in _PLAN['forecast_scoring_horizons_sessions']:
            expiry=daily.prices.index[position+horizon]
            if expiry>pd.Timestamp(_PLAN['label_end_limit']):continue
            for name,fit in fits.items():
                pred=expiry_forecast(fit,decision,expiry,'equity')
                predictions.append((name,horizon,expiry,pred))
        for name,horizon,expiry,pred in predictions:
            evaluation=score_forecast(daily.prices,feature_time,decision,expiry,pred)
            evaluations.append({'asset':symbol,'origin':origin,'model':name,'horizon_sessions':horizon,
                'asof':str(decision),'feature_time':str(feature_time),'label_end':str(expiry),
                'model_fit_cutoff':str(cutoff),'annualized_vol':pred['annualized_vol'],
                'integrated_variance':pred['integrated_variance'],**evaluation})
        for c in _CONTRACTS:
            if c['asset']==symbol and c['origin']==origin:
                trades.extend(trade_contract(c,fits,decision,audit))
    payload={'origin':origin,'training_audit':audit,'evaluations':evaluations,'trades':trades,
             'forecast_failures':forecast_failures}
    checkpoint.write_text(json.dumps(payload,indent=2,default=str,allow_nan=True))
    return str(checkpoint)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.workers<1:p.error('workers must be positive')
    plan=json.loads((ROOT/'ml_vol_plan.json').read_text())
    out=ROOT/'artifacts'/'ml_vol_study'
    for sub in ('months','models','trades'):(out/sub).mkdir(parents=True,exist_ok=True)
    source_files=['ml_vol.py','ml_study.py','forecast.py','stock_replay.py','american.py','stock_market.py']
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    config={'plan':plan,'source_hashes':{s:digest(Path(__file__).parent/s) for s in source_files},
        'signal_source_hash':hashlib.sha256(inspect.getsource(select_side).encode()).hexdigest(),
        'daily_hashes':{s:digest(ROOT/'data'/'stock_study'/f'{s}_daily_actions.csv') for s in plan['universe']},
        'original_run_configuration_hash':digest(ROOT/'artifacts'/'stock_study_20'/'run_configuration.json'),
        'original_results_hash':digest(ROOT/'artifacts'/'stock_study_20'/'all_results.csv'),
        'dependencies':{s:importlib.metadata.version(s) for s in ('numpy','pandas','scipy','arch','scikit-learn','QuantLib')}}
    configuration=out/'run_configuration.json'
    if configuration.exists() and json.loads(configuration.read_text())!=config:
        raise ValueError('ML rules or source data changed; preserve this run and use a fresh output directory')
    configuration.write_text(json.dumps(config,indent=2))
    origins=[]
    for month in pd.date_range(plan['evaluation_start'],plan['evaluation_end'],freq='MS'):
        schedule=sessions(month,month+pd.offsets.MonthEnd(0))
        origins.append(str(schedule.index[0].date()))
    if args.limit:origins=origins[:args.limit]
    paths=[];errors=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(plan,str(out))) as pool:
        future_map={pool.submit(run_month,origin):origin for origin in origins}
        for i,future in enumerate(as_completed(future_map),1):
            origin=future_map[future]
            try:
                path=future.result();paths.append(path)
                payload=json.loads(Path(path).read_text())
                print(f'{i}/{len(origins)} {origin}: labels={payload["training_audit"]["training_rows"]}; forecast rows={len(payload["evaluations"])}; option scenario rows={len(payload["trades"])}',flush=True)
            except Exception as exc:
                error={'origin':origin,'error_type':type(exc).__name__,'reason':str(exc)}
                errors.append(error);print(json.dumps(error),flush=True)
            (out/'errors.json').write_text(json.dumps(errors,indent=2))
    evaluations=[];trades=[];forecast_failures=[]
    for path in sorted(paths):
        payload=json.loads(Path(path).read_text());evaluations.extend(payload['evaluations']);trades.extend(payload['trades'])
        forecast_failures.extend(payload.get('forecast_failures',[]))
    pd.DataFrame(evaluations).to_csv(out/'forecast_predictions.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='cost_parameters'} for r in trades]).to_csv(out/'new_model_trades.csv',index=False)
    (out/'forecast_failures.json').write_text(json.dumps(forecast_failures,indent=2))
    if errors:raise RuntimeError(f'{len(errors)} monthly jobs failed; inspect errors.json; no fallback model was used')
    from .ml_report import make_report
    make_report(out,plan,complete=len(origins)==48)
    print(f'Report: {out/"report.html"}',flush=True)


if __name__=='__main__':main()
