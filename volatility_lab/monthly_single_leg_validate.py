"""Audit monthly opportunity coverage, single-leg positions and causal fills."""
import hashlib
from html.parser import HTMLParser
import json
import re
import shutil
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .monthly_single_leg_acquire import OUT,origins
from .monthly_single_leg_study import RAW
from .single_leg_counterfactual import midpoint_book
from .stock_replay import StockCosts


class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):self.targets.extend(v for k,v in attrs if k in ['href','src'])


def main():
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=json.loads((OUT/'replay_configuration.json').read_text());plan=cfg['plan']
    scfg=json.loads((OUT/'selection_configuration.json').read_text())
    assert not json.loads((OUT/'selection_errors.json').read_text())
    assert not json.loads((OUT/'replay_errors.json').read_text())
    assert digest(OUT/'selection_configuration.json')==cfg['selection_configuration_sha256']
    assert digest(OUT/'decisions.json')==cfg['decisions_sha256']
    assert digest(ROOT/'short_vol_plan.json')==cfg['cost_configuration_sha256']
    assert digest(ROOT/'volatility_lab/monthly_single_leg_acquire.py')==scfg['selection_source_sha256']
    for n,v in cfg['source_hashes'].items():assert digest(ROOT/'volatility_lab'/n)==v,n
    for n,v in scfg['source_hashes'].items():assert digest(ROOT/'volatility_lab'/n)==v,n
    for n,v in scfg['daily_hashes'].items():assert digest(ROOT/'data/stock_study'/f'{n}_daily_actions.csv')==v,n
    assert digest(ROOT/'data/stock_study/sofr.json')==scfg['sofr_sha256']
    for n,v in cfg['raw_hashes'].items():
        assert digest(RAW/(n+'.csv'))==v,n
        meta=json.loads((RAW/(n+'.json')).read_text());assert meta['sha256']==v and meta['parameters']['interval']=='1m'
    decisions=json.loads((OUT/'decisions.json').read_text())
    expected={(s,o,t) for s in plan['universe'] for o in origins(plan) for t in plan['target_calendar_dte']}
    assert len(decisions)==len(expected)==2880
    assert {(c['asset'],c['origin'],c['target_dte']) for c in decisions}==expected
    checkpoints=list((OUT/'decisions').glob('*.json'));assert len(checkpoints)==20*48
    input_hashes={}
    for p in checkpoints:input_hashes.update(json.loads(p.read_text())['input_hashes'])
    repaired_snapshots=0
    for n,v in input_hashes.items():
        source=ROOT/n;assert digest(source)==v,n
        meta=source.with_suffix('.json')
        if meta.exists():
            record=json.loads(meta.read_text())
            if record.get('method')=='actual_quote_union_by_expiry':
                assert record['sha256']==v and record['price_interpolation'] is False
                for component,value in record['component_hashes'].items():
                    assert digest(source.parent/'fallback_components'/(component+'.csv'))==value
                listed=source.parent/'fallback_components'/f'{record["parameters"]["symbol"]}_{record["parameters"]["date"]}_listed_contracts.csv'
                assert digest(listed)==record['listed_contracts_sha256']
                repaired_snapshots+=1
    selected=[]
    for c in decisions:
        if 'feature_time' in c:assert pd.Timestamp(c['feature_time'])<pd.Timestamp(c['decision_time'])
        valid=[r for r in c.get('candidates',[]) if r['status']=='scorable']
        for r in valid:assert abs(c['forecast_vol']-r['decision_ask_iv']-r['vol_gap'])<1e-10
        if c['status'] in ['selected','no_signal']:
            winner=max(valid,key=lambda r:(r['vol_gap'],r['kind']=='call'))
            assert c['kind']==winner['kind'] and c['strike']==winner['strike']
            assert (c['status']=='selected')==(winner['vol_gap']>plan['signal_vol_buffer'])
        if c['status']=='selected':
            c['case_id']=f'{c["asset"]}_{c["origin"]}_{c["target_dte"]}d_{c["kind"]}';selected.append(c)
    f=pd.read_csv(OUT/'all_results.csv',low_memory=False)
    assert len(f)==len(selected)*len(plan['hedge_minutes'])*2
    assert not f.duplicated(['case_id','hedge_minutes','cost_scenario']).any()
    assert set(f.case_id)=={c['case_id'] for c in selected}
    assert f.status.isin(['completed','entry_rejected']).all(),'Unresolved selected holdings cannot be hidden in a completed report'
    complete=f[f.status=='completed'];assert (complete.side==1).all()
    assert complete.reconciliation_error.abs().max()<1e-5
    bridge=complete.market_pnl-complete.execution_cost-complete.abandonment_writeoff+complete.net_financing
    assert (complete.net_pnl-bridge).abs().max()<1e-5
    assert complete[complete.cost_scenario=='no_execution_cost'].execution_cost.abs().max()<1e-7
    costplan=json.loads((ROOT/'short_vol_plan.json').read_text());orders_checked=0;fills_checked=0;pairs_checked=0;all_exits_planned=True
    midpoint_errors=[]
    for n,c in enumerate(selected,1):
        folder=OUT/'trades'/c['case_id'];panel=pd.read_csv(folder/'priced.csv.gz',index_col=0,parse_dates=True)
        panel.index=pd.to_datetime(panel.index,utc=True);subset=f[f.case_id==c['case_id']];orders={}
        record=json.loads((folder/'results.json').read_text())
        assert all(record['numeric_source_hashes'][name]==cfg['source_hashes'][name] for name in record['numeric_source_hashes'])
        assert panel.index.is_unique and panel.index.is_monotonic_increasing
        for r in subset.itertuples():
            if r.status=='entry_rejected':continue
            assert r.kind==c['kind']
            assert pd.Timestamp(r.entry)==pd.Timestamp(c['decision_time'])+pd.Timedelta(minutes=15)
            all_exits_planned &= bool(r.exit_at_planned_time)
            fills=pd.read_csv(folder/f'{r.hedge_minutes}m_{r.cost_scenario}_fills.csv.gz')
            fills['time']=pd.to_datetime(fills.time,utc=True);fills['decision_time']=pd.to_datetime(fills.decision_time,utc=True)
            option=fills[fills.instrument!='stock'];assert len(option)==2 and set(option.instrument)=={c['kind']}
            assert option.quantity.tolist()==[1,-1] and fills[fills.instrument=='stock'].quantity.sum()==0
            stock=fills[(fills.instrument=='stock')&(fills.time!=pd.Timestamp(r.exit))]
            delay=(stock.time-stock.decision_time).dt.total_seconds();assert (delay==plan['signal_lag_minutes']*60).all()
            assert panel.loc[stock.time,'stock_passes_snapshot_filters'].all()
            assert panel.loc[stock.decision_time,'stock_passes_snapshot_filters'].all()
            assert np.isfinite(panel.loc[stock.decision_time,'option_delta']).all();orders_checked+=len(stock)
            costs=costplan['costs_observed' if r.cost_scenario=='observed_costs' else 'costs_midpoint']
            for instrument,rows in fills.groupby('instrument'):
                quotes=panel.loc[rows.time];change=rows.quantity.to_numpy();buy=change>0;prefix='stock' if instrument=='stock' else 'option'
                mid=quotes.spot.to_numpy() if instrument=='stock' else quotes.option_mid.to_numpy()
                market_quote=np.where(buy,quotes[prefix+'_ask'],quotes[prefix+'_bid'])
                fraction=costs['stock_spread_fraction' if instrument=='stock' else 'option_spread_fraction']
                price=mid+(market_quote-mid)*fraction
                if instrument=='stock':
                    price*=1+np.sign(change)*costs['extra_slippage_bps']/10000
                    fee=np.maximum(costs['minimum_stock_commission'],np.abs(change)*costs['stock_commission_per_share'])*costs['fee_multiplier']
                    fee+=np.maximum(-change,0)*price*costs['stock_sell_fee_bps']/10000*costs['fee_multiplier']
                else:
                    abandoned=(~buy)&(quotes.option_bid.to_numpy()==0)
                    price=np.where(abandoned,0.,price)
                    fee=np.where(abandoned,0.,np.abs(change)*costs['option_commission_per_contract']*costs['fee_multiplier'])
                    assert rows.executed.astype(bool).tolist()==(~abandoned).tolist()
                assert np.max(np.abs(rows.fill_price.to_numpy()-price),initial=0.)<1e-7
                assert np.max(np.abs(rows.fee.to_numpy()-fee),initial=0.)<1e-7;fills_checked+=len(rows)
            orders[(r.hedge_minutes,r.cost_scenario)]=fills[['time','instrument','quantity']].to_dict('records')
        for frequency,g in subset.groupby('hedge_minutes'):
            assert g.status.nunique()==1
            if (g.status=='completed').all():
                same=orders[(frequency,'observed_costs')]==orders[(frequency,'no_execution_cost')]
                assert same and g.same_orders_across_costs.all(),'Costs altered inventory; report separately before claiming a pure cost pair'
                assert np.ptp(g.market_pnl)<1e-6 and g.exit.nunique()==1;pairs_checked+=1
                actual=next(r for r in record['results'] if r['hedge_minutes']==frequency and r['cost_scenario']=='observed_costs')
                ledger=pd.read_csv(folder/f'{frequency}m_observed_costs_ledger.csv.gz');ledger['time']=pd.to_datetime(ledger.time,utc=True)
                fills=pd.read_csv(folder/f'{frequency}m_observed_costs_fills.csv.gz');fills['time']=pd.to_datetime(fills.time,utc=True)
                derived,_,_=midpoint_book(panel,actual,ledger,fills,StockCosts(**costplan['costs_observed']))
                reference=float(g[g.cost_scenario=='no_execution_cost'].net_pnl.iloc[0])
                error=abs(derived['net_pnl']-reference);assert error<1e-5;midpoint_errors.append(error)
        if n%50==0:print(f'Audited {n}/{len(selected)} single-option decisions and all their fills',flush=True)
    tests=(ROOT/'artifacts/monthly_single_leg_tests.log').read_text();assert '\nOK\n' in tests
    match=re.search(r'Ran (\d+) tests',tests);assert match and int(match.group(1))>=48
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    source_names=set(cfg['source_hashes'])|set(scfg['source_hashes'])|{'monthly_single_leg_acquire.py','monthly_single_leg_download.py','monthly_single_leg_repair.py','monthly_single_leg_report.py','monthly_single_leg_validate.py','single_leg_counterfactual.py'}
    for name in source_names:shutil.copy2(ROOT/'volatility_lab'/name,snapshot/name)
    shutil.copy2(ROOT/'monthly_single_leg_plan.json',snapshot/'monthly_single_leg_plan.json')
    verification={'status':'passed','unit_tests':int(match.group(1)),'stock_month_decisions':len(checkpoints),
        'total_opportunities':len(decisions),'selected_single_contracts':len(selected),'execution_scenarios':len(f),
        'completed_execution_scenarios':len(complete),'rejected_entry_scenarios':int((f.status=='entry_rejected').sum()),
        'unresolved_execution_scenarios':0,'one_option_leg_per_trade':True,'past_only_forecast_times':True,
        'all_winners_match_ex_ante_vol_gap_rule':True,'independently_audited_fill_prices_and_fees':fills_checked,
        'strictly_past_non_liquidation_stock_signals_checked':orders_checked,'identical_order_cost_pairs':pairs_checked,
        'independently_reconstructed_midpoint_cash_pairs':len(midpoint_errors),'maximum_midpoint_cash_difference':max(midpoint_errors,default=0.),
        'all_completed_exits_at_planned_time':all_exits_planned,'actual_one_minute_quotes':True,'interpolated_subminute_prices':False,
        'malformed_bulk_snapshots_recovered_from_actual_components':repaired_snapshots,
        'maximum_cash_reconciliation_error':float(complete.reconciliation_error.abs().max()),'input_and_source_hashes_match':True,
        'audit_source_hashes':{name:digest(ROOT/'volatility_lab'/name) for name in ['monthly_single_leg_validate.py','single_leg_counterfactual.py','monthly_single_leg_report.py']},
        'all_report_local_links_exist':True}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    parser=Links();parser.feed((OUT/'report.html').read_text())
    assert all((OUT/t).exists() for t in parser.targets if not t.startswith(('https://','http://','#')))
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':main()
