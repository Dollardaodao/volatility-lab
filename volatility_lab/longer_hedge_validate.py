"""Audit clocks, original cohort reproduction and every scheduled fill."""
import hashlib
from html.parser import HTMLParser
import json
import re
import shutil
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .stock_market import sessions
from .longer_hedge_study import OUT,SOURCE


class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):self.targets.extend(v for k,v in attrs if k in ['href','src'])


def main():
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=json.loads((OUT/'configuration.json').read_text());plan=cfg['plan']
    assert json.loads((ROOT/'longer_hedge_plan.json').read_text())==plan
    assert not json.loads((OUT/'errors.json').read_text())
    assert digest(SOURCE/'all_results.csv')==cfg['original_results_sha256']
    assert digest(SOURCE/'verification.json')==cfg['original_verification_sha256']
    for n,v in cfg['source_hashes'].items():assert digest(ROOT/'volatility_lab'/n)==v,n
    for n,v in cfg['priced_panel_hashes'].items():assert digest(SOURCE/'trades'/n/'priced.csv.gz')==v,n
    for n,v in cfg['original_case_hashes'].items():assert digest(SOURCE/'trades'/n/'results.json')==v,n
    for n,v in cfg['daily_hashes'].items():assert digest(ROOT/'data/stock_study'/f'{n}_daily_actions.csv')==v
    f=pd.read_csv(OUT/'all_results.csv',low_memory=False);n_cases=plan['selected_trades']
    assert len(f)==n_cases*len(plan['protocols'])*2 and f.status.eq('completed').all()
    assert not f.duplicated(['case_id','protocol_id','cost_scenario']).any()
    assert f.side.eq(1).all() and f.same_orders_across_costs.all()
    assert f.exit_at_original_time.all(),'Longer clocks changed exits; separate this effect before comparing volatility hits'
    assert f.reconciliation_error.abs().max()<1e-5
    expected=f.market_pnl-f.execution_cost-f.abandonment_writeoff+f.net_financing
    assert (f.net_pnl-expected).abs().max()<1e-5
    ideal=f[f.cost_scenario=='no_execution_cost'];assert ideal.execution_cost.abs().max()<1e-7
    reproduction=pd.read_csv(OUT/'hourly_reproduction.csv');assert len(reproduction)==n_cases and reproduction.same_orders.all()
    assert reproduction.observed_cash_difference.abs().max()<1e-5 and reproduction.ideal_cash_difference.abs().max()<1e-5
    protocols={p['id']:p for p in plan['protocols']};fills_checked=0;signals_checked=0;clock_orders_checked=0;pairs=0
    for n,(case_id,g) in enumerate(f.groupby('case_id'),1):
        source=SOURCE/'trades'/case_id;record=json.loads((OUT/'trades'/case_id/'results.json').read_text())
        assert record['numeric_source_hashes']==cfg['source_hashes']
        original=json.loads((source/'results.json').read_text());contract=original['contract']
        panel=pd.read_csv(source/'priced.csv.gz',index_col=0,parse_dates=True);panel.index=pd.to_datetime(panel.index,utc=True).as_unit('ns')
        days=panel.index.tz_convert('America/New_York').date;ns=panel.index.asi8
        same_day=np.r_[False,days[1:]==days[:-1]]
        clock=pd.Series(np.cumsum(np.where(same_day,np.r_[0,np.diff(ns)],0)),index=panel.index)
        calendar=sessions(contract['origin'],contract['expiry_date'])
        close_map={t.date():row['close']-pd.Timedelta(minutes=15) for t,row in calendar.iterrows()}
        offsets={};opening={};total=0
        for day,row in calendar.iterrows():
            offsets[day.date()]=total;opening[day.date()]=row['open'].value
            total+=int((row['close']-row['open']).value)
        canonical=np.array([offsets[day]+int(time)-opening[day] for day,time in zip(days,ns)],dtype='int64')
        assert np.ptp(canonical-clock.to_numpy())==0,'Trading clock differs from the exchange calendar'
        for r in g.itertuples():
            protocol=protocols[r.protocol_id]
            fills=pd.read_csv(OUT/'trades'/case_id/f'{r.protocol_id}_{r.cost_scenario}_fills.csv.gz')
            fills['time']=pd.to_datetime(fills.time,utc=True);fills['decision_time']=pd.to_datetime(fills.decision_time,utc=True)
            option=fills[fills.instrument!='stock'];assert len(option)==2 and set(option.instrument)=={contract['kind']}
            assert option.quantity.tolist()==[1,-1]
            stock=fills[fills.instrument=='stock'];assert stock.quantity.sum()==0
            adjustments=stock[stock.time!=pd.Timestamp(r.exit)]
            elapsed=(adjustments.time-adjustments.decision_time).dt.total_seconds()
            assert (elapsed==plan['signal_lag_minutes']*60).all()
            assert panel.loc[adjustments.time,'stock_passes_snapshot_filters'].all()
            assert panel.loc[adjustments.decision_time,'stock_passes_snapshot_filters'].all()
            assert np.isfinite(panel.loc[adjustments.decision_time,'option_delta']).all()
            signals_checked+=len(adjustments)
            if len(adjustments)>1:
                if protocol['schedule_kind']=='elapsed':
                    assert (adjustments.time.diff().dropna().dt.total_seconds()>=protocol['frequency']*60).all()
                elif protocol['schedule_kind']=='trading_time':
                    clocks=clock.loc[adjustments.time].to_numpy();assert (np.diff(clocks)>=protocol['frequency']*60*1e9).all()
                else:
                    local=adjustments.time.dt.tz_convert('America/New_York')
                    assert local.dt.date.nunique()==len(local),'Daily schedule placed multiple stock adjustments on one day'
                    initial_day=pd.Timestamp(r.entry).tz_convert('America/New_York').date()
                    later=adjustments[local.dt.date!=initial_day]
                    for row in later.itertuples():
                        day=row.time.tz_convert('America/New_York').date()
                        target=pd.Timestamp(str(day)+' '+protocol.get('daily_time','10:00'),tz='America/New_York').tz_convert('UTC') if protocol['schedule_kind']=='daily_fixed' else close_map[day]
                        assert row.time>=target
                clock_orders_checked+=len(adjustments)
            costs=r.cost_scenario
            cost_params=next(x['cost_parameters'] for x in record['results'] if x['protocol_id']==r.protocol_id and x['cost_scenario']==costs)
            for instrument,rows in fills.groupby('instrument'):
                quotes=panel.loc[rows.time];change=rows.quantity.to_numpy();buy=change>0
                prefix='stock' if instrument=='stock' else 'option';mid=quotes.spot.to_numpy() if instrument=='stock' else quotes.option_mid.to_numpy()
                market_quote=np.where(buy,quotes[prefix+'_ask'],quotes[prefix+'_bid'])
                fraction=cost_params['stock_spread_fraction' if instrument=='stock' else 'option_spread_fraction']
                price=mid+(market_quote-mid)*fraction
                if instrument=='stock':
                    price*=1+np.sign(change)*cost_params['extra_slippage_bps']/10000
                    fee=np.maximum(cost_params['minimum_stock_commission'],np.abs(change)*cost_params['stock_commission_per_share'])*cost_params['fee_multiplier']
                    fee+=np.maximum(-change,0)*price*cost_params['stock_sell_fee_bps']/10000*cost_params['fee_multiplier']
                else:
                    abandoned=(~buy)&(quotes.option_bid.to_numpy()==0)
                    price=np.where(abandoned,0.,price);fee=np.where(abandoned,0.,np.abs(change)*cost_params['option_commission_per_contract']*cost_params['fee_multiplier'])
                    assert rows.executed.astype(bool).tolist()==(~abandoned).tolist()
                assert np.max(np.abs(price-rows.fill_price.to_numpy()),initial=0.)<1e-7
                assert np.max(np.abs(fee-rows.fee.to_numpy()),initial=0.)<1e-7;fills_checked+=len(rows)
        for protocol_id,pair in g.groupby('protocol_id'):
            assert np.ptp(pair.market_pnl)<1e-6 and pair.entry.nunique()==1 and pair.exit.nunique()==1
            files=[pd.read_csv(OUT/'trades'/case_id/f'{protocol_id}_{label}_fills.csv.gz') for label in ['observed_costs','no_execution_cost']]
            assert files[0][['time','instrument','quantity']].to_dict('records')==files[1][['time','instrument','quantity']].to_dict('records')
            pairs+=1
        if n%100==0:print(f'Audited {n}/{n_cases} cases and all schedules',flush=True)
    tests=(ROOT/'artifacts/longer_hedge_tests.log').read_text();assert '\nOK\n' in tests
    match=re.search(r'Ran (\d+) tests',tests);assert match and int(match.group(1))>=55
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in cfg['source_hashes']:shutil.copy2(ROOT/'volatility_lab'/name,snapshot/name)
    for name in ['longer_hedge_report.py','longer_hedge_validate.py']:shutil.copy2(ROOT/'volatility_lab'/name,snapshot/name)
    shutil.copy2(ROOT/'longer_hedge_plan.json',snapshot/'longer_hedge_plan.json')
    result={'status':'passed','unit_tests':int(match.group(1)),'single_option_trades':n_cases,'protocols':len(protocols),
            'completed_scenarios':len(f),'hourly_cash_outcomes_reproduced':n_cases*2,'hourly_orders_reproduced':n_cases,
            'same_order_cost_pairs':pairs,'audited_fill_prices_and_fees':fills_checked,'strictly_past_hedge_signals_checked':signals_checked,
            'clock_qualified_stock_orders_checked':clock_orders_checked,'all_exits_at_original_time':True,
            'max_cash_reconciliation_error':float(f.reconciliation_error.abs().max()),
            'max_hourly_reproduction_difference':float(reproduction[['observed_cash_difference','ideal_cash_difference']].abs().max().max()),
            'original_data_and_numeric_sources_unchanged':True,'input_and_source_hashes_match':True,'all_report_local_links_exist':True}
    result['all_trading_clocks_equivalent_to_exchange_calendar']=True
    (OUT/'trading_clock_coverage.json').write_text(json.dumps({'cases_checked':n_cases,'clock_equivalence_failures':[]},indent=2))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    parser=Links();parser.feed((OUT/'report.html').read_text())
    assert all((OUT/t).exists() for t in parser.targets if not t.startswith(('http://','https://','#')))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
