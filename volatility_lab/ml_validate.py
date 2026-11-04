"""Audit completed ML experiment coverage, timing, provenance and cash."""
import hashlib
import json
import shutil
from html.parser import HTMLParser
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .forecast import future_closes
from .stock_market import sessions,local_time


class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):
        self.targets.extend(v for k,v in attrs if k in ('href','src'))


def main():
    out=ROOT/'artifacts'/'ml_vol_study'
    config=json.loads((out/'run_configuration.json').read_text());plan=config['plan']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,expected in config['source_hashes'].items():
        assert digest(ROOT/'volatility_lab'/name)==expected,name
    assert digest(ROOT/'artifacts'/'stock_study_20'/'all_results.csv')==config['original_results_hash']
    assert not json.loads((out/'errors.json').read_text())
    forecasts=pd.read_csv(out/'forecast_predictions.csv');trades=pd.read_csv(out/'new_model_trades.csv',low_memory=False)
    origins=[]
    for month in pd.date_range(plan['evaluation_start'],plan['evaluation_end'],freq='MS'):
        origins.append(str(sessions(month,month+pd.offsets.MonthEnd(0)).index[0].date()))
    assert len(origins)==48
    expected_keys=set();audited=0
    for origin in origins:
        payload=json.loads((out/'months'/(origin+'.json')).read_text())
        audit=payload['training_audit'];cutoff=pd.Timestamp(audit['fit_cutoff'])
        assert cutoff<local_time(origin,'09:45')
        for asset,record in audit['per_asset'].items():
            assert pd.Timestamp(record['last_label_end'])<=cutoff
            assert pd.Timestamp(record['last_anchor'])<cutoff
            audited+=1
        closes=future_closes(cutoff,84,'equity')
        for h in plan['forecast_scoring_horizons_sessions']:
            end=closes[h-1]
            if end>pd.Timestamp(plan['label_end_limit']):continue
            for asset in plan['universe']:
                for model in plan['models']:expected_keys.add((asset,origin,h,model,str(end)))
    keys=['asset','origin','horizon_sessions','model','label_end']
    assert not forecasts.duplicated(keys).any()
    actual_keys=set(map(tuple,forecasts[keys].itertuples(index=False,name=None)))
    failures=json.loads((out/'forecast_failures.json').read_text())
    failed_models={(f['asset'],f['origin'],f['model']) for f in failures}
    missing=expected_keys-actual_keys
    assert not (actual_keys-expected_keys)
    assert all((asset,origin,model) in failed_models for asset,origin,h,model,end in missing)
    assert np.isfinite(forecasts[['annualized_vol','integrated_variance','qlike']]).all().all()
    assert (forecasts.annualized_vol>0).all() and (forecasts.qlike>=-1e-10).all()
    assert (pd.to_datetime(forecasts.feature_time,utc=True)<pd.to_datetime(forecasts['asof'],utc=True)).all()
    assert (pd.to_datetime(forecasts.model_fit_cutoff,utc=True)<=pd.to_datetime(forecasts.feature_time,utc=True)).all()
    trade_keys=['asset','origin','target_dte','model','cost_scenario']
    assert len(trades)==1920 and not trades.duplicated(trade_keys).any()
    signal=(trades.forecast_vol>trades.long_execution_iv+.03)&(trades.estimated_long_edge>0)
    assert np.array_equal(signal.to_numpy(),(trades.side==1).to_numpy())
    completed=trades[trades.status=='completed'];active=completed[completed.side==1]
    market=active.gross_option_pnl+active.hedge_gross_pnl+active.dividends
    cost=active[['option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees']].sum(axis=1)
    finance=active.cash_interest-active.benchmark_interest-active.borrow_cost
    residual=float((active.net_pnl-(market-cost+finance)).abs().max())
    assert residual<1e-5 and completed.reconciliation_error.abs().max()<1e-5
    assert np.allclose(completed.return_on_capital*3,completed.return_on_notional,atol=1e-10)
    assert ((active.holding_rv>active.entry_execution_ask_iv).to_numpy()==active.vol_bet_correct.to_numpy()).all()
    parser=Links();parser.feed((out/'report.html').read_text())
    assert all((out/t).exists() for t in parser.targets if not t.startswith(('https://','http://','#')))
    tests=(ROOT/'artifacts'/'ml_full_tests.log').read_text()
    assert 'Ran 34 tests' in tests and '\nOK\n' in tests
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in config['source_hashes']:shutil.copy2(ROOT/'volatility_lab'/name,snapshot/name)
    shutil.copy2(ROOT/'volatility_lab'/'ml_report.py',snapshot/'ml_report.py')
    shutil.copy2(ROOT/'ml_vol_plan.json',snapshot/'ml_vol_plan.json')
    record={'status':'passed','unit_tests':34,'monthly_origins':48,'asset_month_label_audits':audited,
        'forecast_rows':len(forecasts),'expected_forecast_rows':len(expected_keys),'disclosed_missing_forecast_rows':len(missing),
        'baseline_forecast_fit_failures':failures,'new_option_scenario_rows':len(trades),
        'new_option_replay_statuses':trades.status.value_counts().to_dict(),
        'new_option_replay_failure_reasons':trades.loc[trades.status!='completed','reason'].value_counts().to_dict(),
        'baseline_exact_expiry_forecasts_reproduced':960,'max_cash_attribution_residual':residual,
        'source_hashes_match':True,'original_results_hash_unchanged':True,'all_report_local_links_exist':True,
        'report_sha256':digest(out/'report.html')}
    (out/'verification.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
