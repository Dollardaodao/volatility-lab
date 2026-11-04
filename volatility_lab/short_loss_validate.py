"""Validate threshold subsets, individual loss cash audits and exit flags."""
import hashlib
import json
import shutil
from html.parser import HTMLParser
import numpy as np
import pandas as pd
from .theta_sdk import ROOT


class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):self.targets.extend(v for k,v in attrs if k in ('href','src'))


def main():
    out=ROOT/'artifacts'/'short_loss_diagnostics';config=json.loads((out/'run_configuration.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,value in config['source_hashes'].items():assert digest(ROOT/'volatility_lab'/name)==value,name
    for name,value in config['input_hashes'].items():assert digest(ROOT/name)==value,name
    assert not json.loads((out/'errors.json').read_text())
    thresholds=pd.read_csv(out/'threshold_summary.csv');decisions=pd.read_csv(out/'threshold_candidate_decisions.csv')
    assert len(thresholds)==40 and len(decisions)==19200
    keys=['asset','origin','target_dte','model','direction']
    prior_sets={}
    for t in [3,5,8,10]:
        f=decisions[decisions.threshold_vol_points==t]
        edge=np.where(f.direction=='short',f.estimated_short_edge,f.estimated_long_edge)
        expected=(f.gap_vol_points>t)&(edge>0)&~((f.direction=='short')&f.decision_exercise_risk)
        assert np.array_equal(expected,f.selected)
        selected=set(map(tuple,f[f.selected][keys].itertuples(index=False,name=None)))
        if prior_sets:assert selected.issubset(prior_sets)
        prior_sets=selected
    old=pd.read_csv(ROOT/'artifacts'/'short_vol_study'/'strategy_summary.csv')
    merged=thresholds[thresholds.threshold_vol_points==3].merge(old,on=['model','direction'],suffixes=('_new','_old'),validate='one_to_one')
    assert np.array_equal(merged.trades_new,merged.trades_old)
    assert np.allclose(merged.observed_net_pnl_new,merged.observed_net_pnl_old,atol=1e-7,rtol=0)
    assert np.allclose(merged.no_execution_cost_net_pnl_new,merged.no_execution_cost_net_pnl_old,atol=1e-7,rtol=0)
    cases=pd.read_csv(out/'worst_unique_trades.csv');assert len(cases)==8
    assert not cases.duplicated(['asset','origin','target_dte']).any()
    counter=pd.read_csv(out/'counterfactual_results.csv',low_memory=False)
    assert len(counter)==128 and (counter.status=='completed').all()
    assert counter.same_orders_across_costs.all()
    fixed=counter[counter.exit_policy=='fixed_original_actual_exit'];assert fixed.exit_at_requested_time.all()
    assert counter[counter.cost_scenario=='no_execution_cost'].execution_cost.abs().max()<1e-6
    assert counter.accounting_residual.abs().max()<1e-5
    exceptions=counter[~counter.exit_at_requested_time]
    assert (exceptions.exit_reason=='research_capital_reserve_close').all()
    exceptions.to_csv(out/'counterfactual_exit_exceptions.csv',index=False)
    baseline=fixed[fixed.hedge_minutes==60].merge(cases[['case_id','net_pnl','ideal_net_pnl']],on='case_id',suffixes=('_current','_reference'),validate='many_to_one')
    expected=np.where(baseline.cost_scenario=='observed_costs',baseline.net_pnl_reference,baseline.ideal_net_pnl)
    assert np.allclose(baseline.net_pnl_current,expected,atol=1e-5,rtol=0)
    cash=pd.read_csv(out/'fill_audit_totals.csv');assert len(cash)==8 and cash.cash_residual.abs().max()<1e-5
    for row in cases.itertuples():
        interval=pd.read_csv(out/'cases'/row.case_id/'interval_attribution.csv')
        assert abs(interval.market_pnl.sum()-row.market_pnl)<1e-5
        audit=pd.read_csv(out/'cases'/row.case_id/'fill_cost_audit.csv')
        assert np.allclose(audit.cash_flow,-audit.quantity*audit.multiplier*audit.fill_price-audit.total_fee,atol=1e-7,rtol=0)
    forecast=pd.read_csv(out/'matched_forecast_diagnostics.csv')
    assert np.isfinite(forecast.forecast_vol_for_actual_holding).all()
    assert (forecast.forecast_vol_for_actual_holding>0).all()
    tests=(ROOT/'artifacts'/'short_loss_tests.log').read_text();assert 'Ran 39 tests' in tests and '\nOK\n' in tests
    from .short_loss_report import make_report
    make_report(out,config['plan'],complete=True)
    triggers=pd.read_csv(out/'exit_trigger_snapshot_audit.csv')
    assert triggers.trigger_matches_original_rule.all()
    assert (pd.to_datetime(triggers.trigger_time,utc=True)<pd.to_datetime(triggers.actual_exit,utc=True)).all()
    parser=Links();parser.feed((out/'report.html').read_text())
    assert all((out/t).exists() for t in parser.targets if not t.startswith(('http://','https://','#')))
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in config['source_hashes']:shutil.copy2(ROOT/'volatility_lab'/name,snapshot/name)
    for name in ('short_loss_report.py','short_loss_validate.py'):shutil.copy2(ROOT/'volatility_lab'/name,snapshot/name)
    shutil.copy2(ROOT/'short_loss_diagnostics_plan.json',snapshot/'short_loss_diagnostics_plan.json')
    record={'status':'passed','unit_tests':39,'unique_loss_cases':len(cases),'counterfactual_scenarios':len(counter),
        'counterfactual_failures':0,'fixed_exit_frequency_tests_reached_original_exit':True,
        'cost_pairs_have_same_orders':True,'scheduled_exit_capital_reserve_exceptions':len(exceptions),
        'exception_cases':sorted(exceptions.case_id.unique()),'threshold_summary_rows':len(thresholds),
        'threshold_candidate_rows':len(decisions),'higher_thresholds_are_nested_subsets':True,
        'three_point_cash_results_unchanged':True,'fill_cash_maximum_reconciliation_error':float(cash.cash_residual.abs().max()),
        'counterfactual_cash_maximum_reconciliation_error':float(counter.accounting_residual.abs().max()),
        'original_inputs_and_financial_source_hashes_unchanged':True,'exit_trigger_snapshots_audited':len(triggers),
        'negative_trigger_mid_extrinsic_cases':int(((triggers.call_extrinsic_mid<0)|(triggers.put_extrinsic_mid<0)).sum()),
        'all_report_local_links_exist':True,'report_sha256':digest(out/'report.html')}
    (out/'verification.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
