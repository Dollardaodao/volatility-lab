"""Audit side selection, frozen cost replay and the preserved long results."""
import hashlib
import inspect
import json
import shutil
from html.parser import HTMLParser
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .stock_study import select_side


class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):self.targets.extend(v for k,v in attrs if k in ('href','src'))


def main():
    out=ROOT/'artifacts'/'short_vol_study';config=json.loads((out/'run_configuration.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert not json.loads((out/'errors.json').read_text())
    for name,value in config['source_hashes'].items():assert digest(ROOT/'volatility_lab'/name)==value,name
    for name,value in config['input_hashes'].items():assert digest(ROOT/name)==value,name
    assert hashlib.sha256(inspect.getsource(select_side).encode()).hexdigest()==config['signal_source_hash']
    for symbol,value in config['daily_hashes'].items():assert digest(ROOT/'data'/'stock_study'/f'{symbol}_daily_actions.csv')==value
    frame=pd.read_csv(out/'all_results.csv',low_memory=False)
    decisions=pd.read_csv(out/'decisions.csv',low_memory=False)
    keys=['asset','origin','target_dte','model','direction']
    assert len(frame)==9600 and len(decisions)==4800
    assert not frame.duplicated(keys+['cost_scenario']).any()
    assert not decisions.duplicated(keys).any()
    assert frame.groupby(keys).size().eq(2).all()
    assert set(frame.model)==set(config['plan']['models'])
    assert (pd.to_datetime(decisions.feature_time,utc=True)<pd.to_datetime(decisions.decision_time,utc=True)).all()
    short=decisions[decisions.direction=='short'];long=decisions[decisions.direction=='long']
    expected_short=(short.forecast_vol<short.short_execution_iv-.03)&(short.estimated_short_edge>0)&(~short.decision_exercise_risk)
    expected_long=(long.forecast_vol>long.long_execution_iv+.03)&(long.estimated_long_edge>0)
    assert np.array_equal(expected_short.to_numpy(),(short.side==-1).to_numpy())
    assert np.array_equal(expected_long.to_numpy(),(long.side==1).to_numpy())
    actual=frame[frame.cost_scenario=='observed_costs'];ideal=frame[frame.cost_scenario=='no_execution_cost']
    pair=actual.merge(ideal,on=keys,suffixes=('_actual','_ideal'),validate='one_to_one')
    assert (pair.status_actual==pair.status_ideal).all()
    valid=pair[pair.status_actual=='completed'];active=valid[valid.side_actual!=0]
    for field in ('entry','exit'):
        assert (pd.to_datetime(active[field+'_actual'],utc=True)==pd.to_datetime(active[field+'_ideal'],utc=True)).all()
    assert np.allclose(active.market_pnl_actual,active.market_pnl_ideal,atol=1e-6,rtol=0)
    assert valid.execution_cost_ideal.abs().max()<1e-6
    assert valid.accounting_residual_actual.abs().max()<1e-5
    assert valid.accounting_residual_ideal.abs().max()<1e-5
    drag=active.net_pnl_ideal-active.net_pnl_actual
    bridge=(active.execution_cost_actual-active.execution_cost_ideal+
            active.abandonment_writeoff_actual-active.abandonment_writeoff_ideal+
            active.net_financing_ideal-active.net_financing_actual)
    assert np.allclose(drag,bridge,atol=1e-5,rtol=0)
    scorable=actual[(actual.status=='completed')&(actual.side!=0)&(actual.vol_bet_status=='scorable')]
    correct=np.where(scorable.direction=='short',scorable.holding_realized_vol<scorable.entry_execution_iv,
                     scorable.holding_realized_vol>scorable.entry_execution_iv)
    assert np.array_equal(correct,scorable.vol_bet_correct.to_numpy())
    # Compare all original candidates, not just a favorable completed subset.
    old=pd.read_csv(ROOT/'artifacts'/'stock_study_20'/'all_results.csv',low_memory=False)
    old=old[(old.hedge_minutes==60)&(old.cost_scenario=='zero_bid_writedown')&old.policy.isin(['garch_long_flat','ewma_long_flat'])].copy()
    old['model']=old.policy.map({'garch_long_flat':'garch_t','ewma_long_flat':'ewma'})
    ml=pd.read_csv(ROOT/'artifacts'/'ml_vol_study'/'new_model_trades.csv',low_memory=False)
    ml=ml[ml.cost_scenario=='zero_bid_writedown']
    reference=pd.concat([old,ml],ignore_index=True,sort=False)
    comparison_keys=['asset','origin','target_dte','model']
    kept=actual[(actual.direction=='long')&(actual.model!='rolling21')]
    merged=kept.merge(reference,on=comparison_keys,suffixes=('_now','_prior'),validate='one_to_one')
    assert len(merged)==1920 and (merged.status_now==merged.status_prior).all()
    assert np.allclose(merged.net_pnl_now,merged.net_pnl_prior,atol=1e-8,rtol=0,equal_nan=True)
    assert np.array_equal(merged.side_now.to_numpy(),merged.side_prior.to_numpy())
    tests=(ROOT/'artifacts'/'short_vol_tests.log').read_text()
    assert 'Ran 37 tests' in tests and '\nOK\n' in tests
    parser=Links();parser.feed((out/'report.html').read_text())
    assert all((out/t).exists() for t in parser.targets if not t.startswith(('http://','https://','#')))
    for name in config['source_hashes']:shutil.copy2(ROOT/'volatility_lab'/name,out/'source_snapshot'/name)
    shutil.copy2(ROOT/'volatility_lab'/'short_vol_report.py',out/'source_snapshot'/'short_vol_report.py')
    shutil.copy2(ROOT/'short_vol_plan.json',out/'source_snapshot'/'short_vol_plan.json')
    record={'status':'passed','unit_tests':37,'scenario_rows':len(frame),'model_direction_decisions':len(decisions),
        'original_long_candidates_preserved':len(merged),'original_long_status_and_cash_identical':True,
        'max_original_long_cash_difference':float((merged.net_pnl_now-merged.net_pnl_prior).abs().max()),
        'cost_replay_statuses_identical':True,'all_active_order_lists_checked_in_runner':True,
        'same_entries_exits_and_market_pnl':True,'no_cost_max_execution_charge':float(valid.execution_cost_ideal.abs().max()),
        'max_cash_bridge_error':float(max(valid.accounting_residual_actual.abs().max(),valid.accounting_residual_ideal.abs().max())),
        'completed_scenarios':int((frame.status=='completed').sum()),
        'replay_failure_reasons':frame.loc[frame.status!='completed','reason'].value_counts().to_dict(),
        'undefined_vol_bet_reasons':actual.loc[(actual.status=='completed')&(actual.side!=0)&(actual.vol_bet_status!='scorable'),'vol_bet_status'].value_counts().to_dict(),
        'known_exercise_risk_rejections':int(short.rejected_known_exercise_risk.sum()),
        'source_and_input_hashes_match':True,'report_links_exist':True}
    (out/'verification.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
