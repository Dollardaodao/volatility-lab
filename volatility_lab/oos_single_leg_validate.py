"""Independent timing, quote, fill, cash and cohort-completeness audit."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from html.parser import HTMLParser
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .oos_single_leg import OUT, ROOT, digest, origins, selected_union, SOURCES, lock
from .stock_market import DailyInputs, local_time, sessions


def validate_case(case):
    folder=OUT/'trades'/case['case_id']; payload=json.loads((folder/'results.json').read_text())
    panel=pd.read_csv(folder/'priced.csv.gz',index_col=0,parse_dates=True)
    panel.index=pd.to_datetime(panel.index,utc=True)
    daily=DailyInputs(ROOT/'data/stock_study',case['asset'])
    maxima={'cash':0.,'equity':0.,'benchmark':0.,'pnl':0.,'fill_price':0.,'fee':0.}
    fill_count=0; completed=0
    order_sets={}
    targets={t.date():r['close']-pd.Timedelta(minutes=15) for t,r in sessions(case['origin'],case['expiry_date']).iterrows()}
    for result in payload['results']:
        if result['status']=='entry_rejected': continue
        completed+=1;scenario=result['cost_scenario'];tag=result['protocol_id']+'_'+scenario
        book=pd.read_csv(folder/(tag+'_ledger.csv.gz'));fills=pd.read_csv(folder/(tag+'_fills.csv.gz'))
        times=pd.DatetimeIndex(pd.to_datetime(book.time,utc=True)).as_unit('ns');p=panel.loc[times]
        fees=result['cost_parameters']; fills['time']=pd.to_datetime(fills.time,utc=True)
        fills['decision_time']=pd.to_datetime(fills.decision_time,utc=True)
        assert (fills.time>=fills.decision_time).all()
        stock=fills[fills.instrument=='stock']; nonexit=stock[stock.time!=pd.Timestamp(result['exit'])]
        assert ((nonexit.time-nonexit.decision_time).dt.total_seconds()>=60).all()
        assert ((nonexit.time-nonexit.decision_time).dt.total_seconds()==60).all()
        assert panel.loc[stock.time,'stock_passes_snapshot_filters'].all()
        assert panel.loc[nonexit.decision_time,'stock_passes_snapshot_filters'].all()
        assert np.isfinite(panel.loc[nonexit.decision_time,'option_delta']).all()
        assert stock.quantity.sum()==0
        if result['protocol_id']=='daily_close':
            local_dates=nonexit.time.dt.tz_convert('America/New_York').dt.date
            assert local_dates.nunique()==len(nonexit)
            for t in nonexit.time:
                day=t.tz_convert('America/New_York').date()
                if day!=pd.Timestamp(result['entry']).tz_convert('America/New_York').date():assert t>=targets[day]
        elif len(nonexit)>1:
            assert (nonexit.time.diff().dropna().dt.total_seconds()>=3600).all()
        key=result['protocol_id']
        orders=fills[['time','instrument','quantity']].to_dict('records')
        if key in order_sets:assert orders==order_sets[key]
        else:order_sets[key]=orders
        assert book.option_contracts.iloc[-1]==0 and book.stock_shares.iloc[-1]==0
        assert fills[fills.instrument!='stock'].quantity.tolist()==[1,-1]
        assert pd.Timestamp(result['entry'])==local_time(case['origin'],'10:00')
        assert pd.Timestamp(result['exit'])<=pd.Timestamp(case['expiry_time'])
        # Every recorded fill is checked against its own observed quote and fee rule.
        for instrument, group in fills.groupby('instrument'):
            q=panel.loc[pd.DatetimeIndex(group.time)]
            is_stock=instrument=='stock';mult=1 if is_stock else 100
            quantity=group.quantity.to_numpy();sign=np.sign(quantity)
            mid=q.spot.to_numpy() if is_stock else q.option_mid.to_numpy()
            prefix='stock_' if is_stock else 'option_'
            bid=q[prefix+'bid'].to_numpy();ask=q[prefix+'ask'].to_numpy()
            quote=np.where(quantity>0,ask,bid)
            fraction=fees['stock_spread_fraction'] if is_stock else fees['option_spread_fraction']
            expected=mid+(quote-mid)*fraction
            if is_stock:expected*=1+sign*fees['extra_slippage_bps']/10000
            executed=group.executed.to_numpy(bool)
            assert q.loc[executed,prefix+'passes_snapshot_filters'].all()
            expected=np.where(executed,expected,0.)
            maxima['fill_price']=max(maxima['fill_price'],float(np.max(np.abs(expected-group.fill_price))))
            if is_stock:
                commission=np.maximum(fees['minimum_stock_commission'],np.abs(quantity)*fees['stock_commission_per_share'])
                expected_fee=(commission+np.maximum(-quantity,0)*expected*fees['stock_sell_fee_bps']/10000)*fees['fee_multiplier']
                size=np.where(quantity>0,q.stock_ask_size,q.stock_bid_size).astype(float)*fees['stock_quote_size_multiplier']
                assert (np.abs(quantity)<=size).all()
            else: expected_fee=np.where(executed,np.abs(quantity)*fees['option_commission_per_contract']*fees['fee_multiplier'],0.)
            maxima['fee']=max(maxima['fee'],float(np.max(np.abs(expected_fee-group.fee))))
        # Recompute all cash transitions from held shares, elapsed time and orders.
        dt=np.r_[0.,np.diff(times.asi8)/1e9/(365*86400)]
        cash=book.cash.to_numpy();prev_cash=np.r_[result['capital'],cash[:-1]]
        hedge=book.stock_shares.to_numpy();prev_hedge=np.r_[0.,hedge[:-1]]
        rate=np.r_[p.rate.iloc[0],p.rate.to_numpy()[:-1]]
        interest=prev_cash*np.expm1(np.where(prev_cash>=0,np.maximum(rate-fees['lending_spread'],0.),rate+fees['debit_spread'])*dt)
        prev_spot=np.r_[p.spot.iloc[0],p.spot.to_numpy()[:-1]]
        borrow=np.maximum(-prev_hedge,0)*prev_spot*fees['short_stock_borrow_rate']*dt
        dates=times.tz_convert('America/New_York').date
        changed=np.r_[False,dates[1:]!=dates[:-1]]
        dividends=prev_hedge*np.array([daily.dividends.get(d,0.) if change else 0. for d,change in zip(dates,changed)])
        mult=np.where(fills.instrument=='stock',1,100)
        flow=pd.Series(fills.quantity.to_numpy()*fills.fill_price.to_numpy()*mult+fills.fee.to_numpy(),index=pd.DatetimeIndex(fills.time))
        flow=flow.groupby(level=0).sum().reindex(times,fill_value=0.).to_numpy()
        expected_cash=prev_cash+interest-borrow+dividends-flow
        maxima['cash']=max(maxima['cash'],float(np.max(np.abs(cash-expected_cash))))
        benchmark=book.benchmark.to_numpy();prev_bench=np.r_[result['capital'],benchmark[:-1]]
        expected_bench=prev_bench*np.exp(np.maximum(rate-fees['lending_spread'],0.)*dt)
        maxima['benchmark']=max(maxima['benchmark'],float(np.max(np.abs(benchmark-expected_bench))))
        marks=panel.option_mid.where(panel.option_passes_snapshot_filters).ffill().loc[times].to_numpy()
        equity=cash+hedge*p.spot.to_numpy()+book.option_contracts.to_numpy()*100*marks
        maxima['equity']=max(maxima['equity'],float(np.max(np.abs(equity-book.equity))))
        maxima['pnl']=max(maxima['pnl'],abs(float(cash[-1]-benchmark[-1])-result['net_pnl']))
        fill_count+=len(fills)
    assert max(maxima.values())<1e-5,(case['case_id'],maxima)
    return {'case_id':case['case_id'],'scenario_rows':completed,'fill_records':fill_count,**maxima}


def main():
    plan=json.loads((ROOT/'oos_single_leg_plan.json').read_text());hashes=lock(plan)
    protocol_lock=json.loads((OUT/'protocol_lock.json').read_text())
    assert digest(ROOT/'artifacts/monthly_single_leg/all_results.csv')==protocol_lock['prior_results_sha256']
    d=json.loads((OUT/'decisions.json').read_text());assert len(d)==2100
    frame=pd.DataFrame(d)
    assert not frame.duplicated(['opportunity_id','model']).any()
    assert all(v==420 for v in frame.groupby('model').size())
    assert (frame.dropna(subset=['case_id']).groupby('opportunity_id').case_id.nunique()<=1).all()
    audit_rows=[]
    for origin in origins(plan):
        m=json.loads((OUT/'months'/f'{origin}.json').read_text());a=m['training_audit']
        cutoff=pd.Timestamp(a['fit_cutoff']);decision=local_time(origin,'09:45')
        assert cutoff<decision
        for symbol,row in a['per_asset'].items():
            assert pd.Timestamp(row['last_label_end'])<=cutoff
            assert pd.Timestamp(row['last_anchor'])<=cutoff
            audit_rows.append({'origin':origin,'asset':symbol,'cutoff':str(cutoff),**row})
        for symbol,models in m['forecasts'].items():
            assert set(models)==set(plan['models'])
            for name,f in models.items():
                assert pd.Timestamp(f['trained_through'])<=cutoff
                assert np.isfinite(f['daily_variance']).all() and min(f['daily_variance'])>0
    for c in d:
        if 'kind' in c:
            valid=[v for v in c['candidates'] if v['status']=='scorable']
            winner=min(valid,key=lambda v:(v['decision_ask_iv'],v['kind']!='call'))
            assert c['kind']==winner['kind'] and c['strike']==winner['strike']
            assert abs(c['vol_gap']-(c['forecast_vol']-c['decision_ask_iv']))<1e-12
            assert (c['status']=='selected')==(c['vol_gap']>.03)
    for filename in ['decision_download_errors.json','minute_download_errors.json','replay_errors.json']:
        p=OUT/filename
        assert not p.exists() or json.loads(p.read_text())==[]
    config=json.loads((OUT/'configuration.json').read_text())
    for name,h in config['daily_hashes'].items():assert digest(ROOT/'data/stock_study'/f'{name}_daily_actions.csv')==h
    for name,h in json.loads((OUT/'decision_input_hashes.json').read_text()).items(): assert digest(ROOT/name)==h
    manifest=json.loads((OUT/'minute_manifest.json').read_text())
    for name,job in manifest['jobs'].items():assert digest(ROOT/'data/near_continuous/quotes'/(name+'.csv'))==job['sha256']
    selected=selected_union();rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(validate_case,c) for c in selected]
        for future in as_completed(futures): rows.append(future.result())
    pd.DataFrame(rows).to_csv(OUT/'cash_fill_audit.csv',index=False)
    pd.DataFrame(audit_rows).to_csv(OUT/'training_timing_audit.csv',index=False)
    results=pd.read_csv(OUT/'unique_trade_results.csv')
    assert len(results)==4*len(selected)
    assert set(results.case_id)=={c['case_id'] for c in selected}
    assert set(results.status)<=set(['completed','entry_rejected'])
    assert len(pd.read_csv(OUT/'forecast_predictions.csv'))==2100
    assert 'Ran 55 tests' in (OUT/'unit_tests.log').read_text() and (OUT/'unit_tests.log').read_text().rstrip().endswith('OK')
    source=OUT/'source_snapshot';source.mkdir(exist_ok=True)
    for name in SOURCES:
        assert digest(ROOT/'volatility_lab'/name)==hashes[name]
        (source/name).write_bytes((ROOT/'volatility_lab'/name).read_bytes())
    for name in ['oos_single_leg_report.py','oos_single_leg_validate.py']:
        (source/name).write_bytes((ROOT/'volatility_lab'/name).read_bytes())
    verification={'status':'passed','opportunities_per_model':420,'models':5,'forecast_scores':2100,
        'distinct_selected_cohorts':len(selected),'scenario_rows':len(results),
        'entry_rejected_scenario_rows':int((results.status=='entry_rejected').sum()),
        'completed_scenario_rows':int((results.status=='completed').sum()),
        'unit_tests':55,'fill_records_checked':sum(r['fill_records'] for r in rows),
        'max_independent_cash_residual':max(r['cash'] for r in rows),
        'max_independent_fill_price_residual':max(r['fill_price'] for r in rows),
        'max_independent_fee_residual':max(r['fee'] for r in rows),
        'training_label_end_checked_for_all_stock_months':True,
        'numeric_results_sha256':digest(OUT/'unique_trade_results.csv'),
        'model_results_sha256':digest(OUT/'model_trade_results.csv'),
        'primary_summary_sha256':digest(OUT/'summary.csv')}
    verification.update(report_source_sha256=digest(ROOT/'volatility_lab/oos_single_leg_report.py'),
        audit_source_sha256=digest(ROOT/'volatility_lab/oos_single_leg_validate.py'),
        report_html_sha256=digest(OUT/'report.html'),prior_results_unchanged=True)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and not v.startswith(('https:','http:','#')):assert (OUT/v).exists(),v
    Links().feed((OUT/'report.html').read_text())
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
