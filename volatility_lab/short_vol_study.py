"""Frozen-forecast long/short straddles, with matched execution-cost replays."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace,asdict
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .stock_market import DailyInputs
from .stock_replay import StockCosts,replay,exercise_risk
from .stock_study import select_side
from .forecast import fit_variance,expiry_forecast
from .american import AmericanPricer

RESULT_KEYS=['side','net_pnl','gross_option_pnl','hedge_gross_pnl','dividends','option_spread_cost',
    'stock_spread_cost','slippage_cost','option_fees','stock_fees','cash_interest','benchmark_interest',
    'borrow_cost','capital','return_on_capital','return_on_notional','fills','abandoned_long_legs',
    'max_drawdown_usd','reconciliation_error','entry','exit','exit_reason']
TX=['option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees']
_PLAN=None;_OUT=None;_DAILY=None;_BASE=None;_ML=None;_CONTRACTS=None


def initialize(plan,out):
    global _PLAN,_OUT,_DAILY,_BASE,_ML,_CONTRACTS
    _PLAN=plan;_OUT=Path(out)
    _DAILY={s:DailyInputs(ROOT/'data'/'stock_study',s) for s in plan['universe']}
    frame=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/'all_results.csv',low_memory=False)
    frame=frame[(frame.hedge_minutes==60)&(frame.cost_scenario=='zero_bid_writedown')]
    _BASE=frame.set_index(['asset','origin','target_dte','policy'])
    frame=pd.read_csv(ROOT/'artifacts'/'ml_vol_study'/'new_model_trades.csv',low_memory=False)
    _ML=frame[frame.cost_scenario=='zero_bid_writedown'].set_index(['asset','origin','target_dte','model'])
    _CONTRACTS=json.loads((ROOT/'artifacts'/'stock_study_acquisition_20.json').read_text())['cohorts']


def trade_diagnostics(panel,contract,result,side,fills):
    """Diagnostics cannot discard an otherwise reconciled completed trade."""
    if not side:return {'vol_bet_status':'no_trade','abandonment_writeoff':0.}
    entry,exit=pd.Timestamp(result['entry']),pd.Timestamp(result['exit'])
    abandoned=0.
    if len(fills) and 'executed' in fills:
        for r in fills.itertuples():
            flag=getattr(r,'executed',True)
            if pd.notna(flag) and flag in (False,'False'):
                abandoned+=abs(r.quantity)*100*float(panel.loc[pd.Timestamp(r.time),r.instrument+'_mid'])
    info={'holding_days':(exit-entry).total_seconds()/86400,'abandonment_writeoff':abandoned}
    if exit<=entry:return {**info,'vol_bet_status':'zero_holding_window'}
    held=panel.loc[entry:exit];years=(exit-entry).total_seconds()/(365*86400)
    rv=float(np.sqrt(np.square(np.diff(np.log(held.spot.to_numpy()))).sum()/years))
    info['holding_realized_vol']=rv
    er=panel.loc[entry];ivs=[];quote='ask' if side==1 else 'bid'
    try:
        for kind in ('call','put'):
            pricer=AmericanPricer(contract['strike'],kind)
            tau=(pd.Timestamp(contract['expiry_time'])-entry).total_seconds()/(365*86400)
            pricer.configure(float(er.spot),tau,float(er.rate),float(er.dividend_yield))
            iv=pricer.implied(float(er[kind+'_'+quote]));ivs.append(iv)
            info['entry_'+kind+'_'+quote+'_iv']=iv
    except (ValueError,RuntimeError) as exc:
        return {**info,'vol_bet_status':'entry_iv_unavailable','vol_bet_reason':str(exc)}
    reference=max(ivs) if side==1 else min(ivs)
    return {**info,'entry_execution_iv':reference,'vol_bet_status':'scorable',
            'vol_bet_correct':bool(rv>reference if side==1 else rv<reference)}


def account(result,diagnostics):
    if result['side']==0:return {'market_pnl':0.,'execution_cost':0.,'abandonment_writeoff':0.,'net_financing':0.,'accounting_residual':0.}
    market=result['gross_option_pnl']+result['hedge_gross_pnl']+result['dividends']
    abandonment=diagnostics['abandonment_writeoff']
    execution=sum(result[k] for k in TX)-abandonment
    financing=result['cash_interest']-result['benchmark_interest']-result['borrow_cost']
    residual=result['net_pnl']-(market-execution-abandonment+financing)
    if abs(residual)>1e-5 or execution< -1e-7:raise AssertionError('Execution-cost bridge does not reconcile')
    return {'market_pnl':market,'execution_cost':execution,'abandonment_writeoff':abandonment,
            'net_financing':financing,'accounting_residual':residual}


def cached_long(model,c,stem):
    """Keep prior long cash outcomes unchanged and reuse their actual orders."""
    index=(c['asset'],c['origin'],c['target_dte'])
    if model in ('garch_t','ewma'):
        policy='garch_long_flat' if model=='garch_t' else 'ewma_long_flat'
        raw=_BASE.loc[(*index,policy)].to_dict()
        path=ROOT/'artifacts'/'stock_study_20'/f'{stem}_{policy}_60_zero_bid_writedown_fills.csv'
    elif model in ('har_proxy_ridge','ml_hist_gradient_boosting'):
        raw=_ML.loc[(*index,model)].to_dict()
        path=ROOT/'artifacts'/'ml_vol_study'/'trades'/f'{stem}_{model}_zero_bid_writedown_fills.csv'
    else:return None
    result={k:raw[k] for k in RESULT_KEYS if k in raw}
    if raw['status']!='completed':
        return {'result':result,'status':raw['status'],'reason':raw.get('reason'),
                'error_type':raw.get('error_type'),'fills':pd.DataFrame()}
    fills=pd.read_csv(path) if int(raw['side']) else pd.DataFrame()
    return {'result':result,'status':'completed','fills':fills}


def run_origin(origin):
    path=_OUT/'origins'/(origin+'.json')
    if path.exists():return str(path)
    results=[];decisions=[];rolling_fits={}
    contracts=sorted([c for c in _CONTRACTS if c['origin']==origin and c['status']=='acquired'],
                     key=lambda c:(c['asset'],c['target_dte']))
    observed=StockCosts(**_PLAN['costs_observed']);midpoint=StockCosts(**_PLAN['costs_midpoint'])
    for c in contracts:
        stem=f'{c["asset"]}_{origin}_{c["target_dte"]}d'
        panel=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/(stem+'_priced.csv'),index_col=0,parse_dates=True)
        panel.index=pd.to_datetime(panel.index,utc=True)
        decision=pd.Timestamp(c['decision_time']);expiry=pd.Timestamp(c['expiry_time']);row=panel.loc[decision]
        old=_BASE.loc[(c['asset'],origin,c['target_dte'],'garch_long_flat')]
        risk=exercise_risk(row,c['strike'])
        if c['asset'] not in rolling_fits:
            rolling_fits[c['asset']]=fit_variance(_DAILY[c['asset']].prices,decision,horizon=84,model='rolling21')
        for model in _PLAN['models']:
            if model in ('garch_t','ewma'):
                vol=float(old[model+'_forecast_vol']);feature_time=old[model+'_trained_through']
            elif model=='rolling21':
                fit=rolling_fits[c['asset']];feature_time=str(fit.trained_through)
                vol=expiry_forecast(fit,decision,expiry,'equity')['annualized_vol']
            else:
                source=_ML.loc[(c['asset'],origin,c['target_dte'],model)]
                vol=float(source.forecast_vol);feature_time=source.feature_time
            if pd.Timestamp(feature_time)>=decision:raise AssertionError('Forecast contains future input')
            fair=0.
            for kind in ('call','put'):
                pricer=AmericanPricer(c['strike'],kind)
                pricer.configure(float(row.spot),(expiry-decision).total_seconds()/(365*86400),float(row.rate),float(row.dividend_yield))
                fair+=pricer.value(vol)
            values={'garch_t_forecast_vol':vol,'garch_t_fair_sum':fair,
                    'long_execution_iv':float(old.long_execution_iv),'short_execution_iv':float(old.short_execution_iv)}
            selected,hurdle=select_side('garch_long_short',values,row,c,60,_PLAN['signal_vol_buffer'])
            for direction in ('long','short'):
                side=1 if direction=='long' and selected==1 else -1 if direction=='short' and selected==-1 and not risk else 0
                base={**c,'model':model,'direction':direction,'side':side,'feature_time':str(feature_time),
                    'forecast_vol':vol,'long_execution_iv':float(old.long_execution_iv),
                    'short_execution_iv':float(old.short_execution_iv),'hedge_minutes':60,
                    'decision_exercise_risk':bool(risk),'rejected_known_exercise_risk':bool(direction=='short' and selected==-1 and risk),**hurdle}
                decisions.append(base)
                cached=cached_long(model,c,stem) if direction=='long' else None
                if cached is not None and int(cached['result']['side'])!=side:
                    raise AssertionError('Original long entry selection changed')
                outcomes={};orders={}
                for label,costs in [('observed_costs',observed),('no_execution_cost',midpoint)]:
                    ledger=pd.DataFrame();fills=pd.DataFrame()
                    try:
                        if cached is not None and label=='observed_costs':
                            if cached['status']!='completed':
                                outcomes[label]={**base,'cost_scenario':label,'status':cached['status'],
                                    'reason':cached['reason'],'error_type':cached['error_type']}
                                continue
                            r=cached['result'];fills=cached['fills']
                        else:r,ledger,fills=replay(panel,c,_DAILY[c['asset']],side,60,costs)
                    except Exception as exc:
                        outcomes[label]={**base,'cost_scenario':label,'status':'replay_failed',
                            'error_type':type(exc).__name__,'reason':str(exc)}
                        continue
                    # A missing diagnostic never turns a completed cash replay
                    # into a missing observation or silently removes its loss.
                    diagnostics=trade_diagnostics(panel,c,r,side,fills)
                    bridge=account(r,diagnostics)
                    outcomes[label]={**base,**r,**diagnostics,**bridge,'cost_scenario':label,'status':'completed'}
                    if side:
                        orders[label]=fills[['time','instrument','quantity']].assign(time=lambda f:pd.to_datetime(f.time,utc=True)).to_dict('records')
                        tag=f'{stem}_{model}_{direction}_{label}'
                        fills.to_csv(_OUT/'trades'/(tag+'_fills.csv'),index=False)
                        if len(ledger):ledger.to_csv(_OUT/'trades'/(tag+'_ledger.csv'),index=False)
                a,b=outcomes['observed_costs'],outcomes['no_execution_cost']
                if a['status']!=b['status']:raise AssertionError('Cost counterfactual changed data eligibility')
                if side and a['status']=='completed':
                    if pd.Timestamp(a['entry'])!=pd.Timestamp(b['entry']) or pd.Timestamp(a['exit'])!=pd.Timestamp(b['exit']):
                        raise AssertionError('Cost counterfactual changed holding period')
                    if orders['observed_costs']!=orders['no_execution_cost']:
                        raise AssertionError('Cost counterfactual changed orders')
                    if abs(a['market_pnl']-b['market_pnl'])>1e-6:raise AssertionError('Counterfactual market P&L changed')
                    if b['execution_cost']>1e-6:raise AssertionError('No-execution-cost scenario still charged execution costs')
                results.extend(outcomes.values())
    path.write_text(json.dumps({'origin':origin,'decisions':decisions,'results':results},indent=2,default=str,allow_nan=True))
    return str(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.workers<1:p.error('workers must be positive')
    plan=json.loads((ROOT/'short_vol_plan.json').read_text());out=ROOT/'artifacts'/'short_vol_study'
    for d in ('origins','trades','source_snapshot'):(out/d).mkdir(parents=True,exist_ok=True)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source_names=['short_vol_study.py','stock_replay.py','stock_market.py','forecast.py','american.py','ml_vol.py']
    sources={name:digest(Path(__file__).parent/name) for name in source_names}
    inputs=[ROOT/'artifacts'/'stock_study_20'/'all_results.csv',ROOT/'artifacts'/'ml_vol_study'/'new_model_trades.csv',
            ROOT/'artifacts'/'stock_study_acquisition_20.json',ROOT/'artifacts'/'ml_vol_study'/'run_configuration.json']
    config={'plan':plan,'source_hashes':sources,'input_hashes':{str(f.relative_to(ROOT)):digest(f) for f in inputs},
        'signal_source_hash':hashlib.sha256(inspect.getsource(select_side).encode()).hexdigest(),
        'daily_hashes':{s:digest(ROOT/'data'/'stock_study'/f'{s}_daily_actions.csv') for s in plan['universe']}}
    configpath=out/'run_configuration.json'
    if configpath.exists() and json.loads(configpath.read_text())!=config:
        raise ValueError('Short-study inputs or rules changed; preserve this run before using a new configuration')
    configpath.write_text(json.dumps(config,indent=2))
    origins=plan['origins'][:args.limit] if args.limit else plan['origins']
    paths=[];errors=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=(plan,str(out))) as pool:
        futures={pool.submit(run_origin,origin):origin for origin in origins}
        for n,future in enumerate(as_completed(futures),1):
            origin=futures[future]
            try:
                path=future.result();paths.append(path);payload=json.loads(Path(path).read_text())
                print(f'{n}/{len(origins)} {origin}: {len(payload["results"])} matched model/direction/cost scenarios',flush=True)
            except Exception as exc:
                errors.append({'origin':origin,'error_type':type(exc).__name__,'reason':str(exc)})
                print(json.dumps(errors[-1]),flush=True)
            (out/'errors.json').write_text(json.dumps(errors,indent=2))
    results=[];decisions=[]
    for path in sorted(paths):
        payload=json.loads(Path(path).read_text());results.extend(payload['results']);decisions.extend(payload['decisions'])
    pd.DataFrame([{k:v for k,v in r.items() if k!='cost_parameters'} for r in results]).to_csv(out/'all_results.csv',index=False)
    pd.DataFrame(decisions).to_csv(out/'decisions.csv',index=False)
    if errors:raise RuntimeError('Origin jobs failed; inspect errors.json; no financial fallback was used')
    from .short_vol_report import make_report
    make_report(out,plan,complete=len(origins)==len(plan['origins']))
    print(f'Report: {out/"report.html"}',flush=True)


if __name__=='__main__':main()
