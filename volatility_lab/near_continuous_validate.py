"""Audit minute sampling, causal hedge signals, frozen exits and cash."""
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
    out=ROOT/'artifacts'/'near_continuous_study';cfg=json.loads((out/'run_configuration.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert not json.loads((out/'errors.json').read_text())
    for name,value in cfg['source_hashes'].items():assert digest(ROOT/'volatility_lab'/name)==value,name
    for name,value in cfg['raw_hashes'].items():assert digest(ROOT/'data'/'near_continuous'/'quotes'/(name+'.csv'))==value
    for symbol,value in cfg['daily_hashes'].items():assert digest(ROOT/'data'/'stock_study'/f'{symbol}_daily_actions.csv')==value
    assert digest(ROOT/'data'/'stock_study'/'sofr.json')==cfg['sofr_hash']
    assert digest(ROOT/'artifacts'/'short_loss_diagnostics'/'worst_unique_trades.csv')==cfg['source_results_hash']
    f=pd.read_csv(out/'all_results.csv',low_memory=False)
    keys=['case_id','data_source','grid_minutes','hedge_minutes','signal_lag_minutes','cost_scenario']
    assert len(f)==60 and not f.duplicated(keys).any()
    assert (f.status=='completed').all() and f.exit_at_original_time.all() and f.same_orders_across_costs.all()
    assert f.reconciliation_error.abs().max()<1e-5 and f.accounting_residual.abs().max()<1e-5
    assert f[f.cost_scenario=='no_execution_cost'].execution_cost.abs().max()<1e-7
    cases=pd.read_csv(ROOT/'artifacts'/'short_loss_diagnostics'/'worst_unique_trades.csv').set_index('case_id')
    quality=json.loads((out/'quote_quality.json').read_text());differences=[];hedge_orders_checked=0
    for case_id in cfg['plan']['cases']:
        q=quality[case_id];assert q['minute_grid_confirmed']
        assert all(v<1e-8 for v in q['quote_differences_vs_archived_15m'].values())
        for cost in ['observed_costs','no_execution_cost']:
            archived=f[(f.case_id==case_id)&(f.data_source=='archived_15m')&(f.cost_scenario==cost)].iloc[0]
            coarse=f[(f.case_id==case_id)&(f.data_source=='resampled_15m')&(f.cost_scenario==cost)].iloc[0]
            reference=cases.loc[case_id,'net_pnl' if cost=='observed_costs' else 'ideal_net_pnl']
            assert abs(archived.net_pnl-reference)<1e-5
            assert abs(coarse.net_pnl-archived.net_pnl)<1e-5
            differences.append({'case_id':case_id,'cost_scenario':cost,'resampled_minus_archived_cash':coarse.net_pnl-archived.net_pnl})
    for r in f.itertuples():
        path=out/'cases'/r.case_id/f'{r.data_source}_{r.hedge_minutes}m_lag{r.signal_lag_minutes}m_{r.cost_scenario}_fills.csv'
        fills=pd.read_csv(path);stock=fills[(fills.instrument=='stock')&(pd.to_datetime(fills.time,utc=True)!=pd.Timestamp(r.exit))]
        elapsed=(pd.to_datetime(stock.time,utc=True)-pd.to_datetime(stock.decision_time,utc=True)).dt.total_seconds()
        assert (elapsed==r.signal_lag_minutes*60).all();hedge_orders_checked+=len(stock)
    pd.DataFrame(differences).to_csv(out/'archived_cash_reproduction.csv',index=False)
    tests=(ROOT/'artifacts'/'near_continuous_tests.log').read_text();assert 'Ran 42 tests' in tests and '\nOK\n' in tests
    from .near_continuous_report import make_report
    make_report(out,cfg['plan'])
    delta=pd.read_csv(out/'delta_quality.csv');assert (delta.mean_abs_delta_shares>=0).all()
    parser=Links();parser.feed((out/'report.html').read_text())
    assert all((out/t).exists() for t in parser.targets if not t.startswith(('http://','https://','#')))
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in cfg['source_hashes']:shutil.copy2(ROOT/'volatility_lab'/name,snapshot/name)
    for name in ['near_continuous_report.py','near_continuous_validate.py']:shutil.copy2(ROOT/'volatility_lab'/name,snapshot/name)
    shutil.copy2(ROOT/'near_continuous_plan.json',snapshot/'near_continuous_plan.json')
    result={'status':'passed','unit_tests':42,'cases':3,'completed_scenarios':len(f),
        'actual_one_minute_stock_and_option_snapshots':True,'interpolated_subinterval_prices':False,
        'all_original_coarse_quote_prices_reproduced':True,'archived_cash_results_reproduced':6,
        'minute_data_resampled_to_coarse_cash_reproduced':6,'causal_non_liquidation_stock_orders_checked':hedge_orders_checked,
        'all_cost_pairs_same_orders':True,'all_exits_at_original_time':True,
        'max_cash_reconciliation_error':float(f.reconciliation_error.abs().max()),
        'fine_rows_by_case':{name:q['fine_rows'] for name,q in quality.items()},
        'source_and_input_hashes_match':True,'all_report_local_links_exist':True}
    (out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
