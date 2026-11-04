"""Repair two malformed bulk snapshots using actual per-expiry requests."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import time
import pandas as pd
from .theta_sdk import ROOT,connect
from .theta import normalize_quotes
from .stock_market import MarketStore,local_time
from .monthly_single_leg_acquire import DATA,OUT


def main():
    try:client=connect()
    except Exception as exc:raise RuntimeError('Repair authentication failed ('+type(exc).__name__+')') from None
    store=MarketStore(DATA/'decision_quotes/fallback_components',client)
    source=ROOT/'data/stock_study/quotes';audit=[]
    for symbol in ['AAPL','AMZN']:
        origin='2022-04-01';target=DATA/'decision_quotes'/f'{symbol}_{origin}_full_origin_snapshot.csv'
        if target.exists():continue
        contracts=store.fetch(f'{symbol}_{origin}_listed_contracts','option_list_contracts',
                             request_type='quote',date=pd.Timestamp(origin).date(),symbol=symbol,max_dte=115)
        expirations=pd.to_datetime(contracts.expiration).dt.date
        expiry_dates=sorted({d for d in expirations if 1<(d-pd.Timestamp(origin).date()).days<=115})
        stock=normalize_quotes(pd.read_csv(source/f'{symbol}_stock_202204.csv'))
        s=stock[stock.sample_time_utc==local_time(origin,'09:45')].iloc[0];spot=float((s.bid+s.ask)/2)
        frames=[];components=[];errors=[]
        def fetch(expiry):
            name=f'{symbol}_{origin}_{expiry}_full_expiry_snapshot'
            params={'symbol':symbol,'expiration':expiry,'date':pd.Timestamp(origin).date(),
                    'interval':'15m','start_time':'09:45:00','end_time':'09:46:00'}
            error={}
            for attempt in range(3):
                try:return {'frame':store.fetch(name,'option_history_quote',**params),'name':name,'expiry':str(expiry)}
                except Exception as exc:
                    error={'expiry':str(expiry),'error_type':type(exc).__name__}
                    if callable(getattr(exc,'code',None)):error['grpc_code']=str(exc.code())
                    if attempt<2:time.sleep([1,3][attempt])
            return {'error':error,'expiry':str(expiry)}
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fetch,e) for e in expiry_dates]
            for future in as_completed(futures):
                result=future.result()
                if 'frame' in result:frames.append(result['frame']);components.append(result['name'])
                else:errors.append(result['error'])
        unresolved=[]
        for error in errors:
            expiry=pd.Timestamp(error['expiry']).date()
            if error['error_type']=='NoDataFoundError':continue
            # Some old bulk files mix legacy row formats. If necessary, walk
            # the day's available strikes in distance order. A leg is retained
            # only after an actual valid quote at the decision timestamp exists.
            strikes=sorted(pd.to_numeric(contracts.loc[expirations==expiry,'strike']).unique(),key=lambda k:(abs(k-spot),k))
            found={};component_names=[];unknown=False
            for strike in strikes:
                name=f'{symbol}_{origin}_{expiry}_{strike:.3f}_point_snapshot'
                try:
                    raw=store.fetch(name,'option_history_quote',symbol=symbol,expiration=expiry,strike=f'{strike:.3f}',
                                    date=pd.Timestamp(origin).date(),interval='15m',start_time='09:45:00',end_time='09:46:00')
                except Exception as exc:
                    if type(exc).__name__=='NoDataFoundError':continue
                    unresolved.append({'expiry':str(expiry),'strike':float(strike),'error_type':type(exc).__name__});unknown=True;break
                normalized=normalize_quotes(raw)
                valid=normalized[(normalized.sample_time_utc==local_time(origin,'09:45'))&normalized.passes_snapshot_filters]
                for kind in ['CALL','PUT']:
                    if kind not in found and (valid.right==kind).any():
                        found[kind]=raw[raw.right.str.upper()==kind];component_names.append(name)
                if len(found)==2:break
            if unknown:continue
            if found:frames.extend(found.values());components.extend(component_names)
        if unresolved:
            (OUT/'snapshot_repair_unresolved.json').write_text(json.dumps(unresolved,indent=2))
            raise RuntimeError('A nearer contract cannot be verified; refusing to substitute a farther strike')
        if not frames:raise RuntimeError('No actual decision quotes recovered')
        combined=pd.concat(frames,ignore_index=True).drop_duplicates()
        combined.to_csv(target,index=False)
        meta={'method':'actual_quote_union_by_expiry','parameters':{'symbol':symbol,'date':origin,'interval':'15m','max_dte':'115'},
              'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'rows':len(combined),
              'component_hashes':{n:hashlib.sha256((store.root/(n+'.csv')).read_bytes()).hexdigest() for n in set(components)},
              'listed_contracts_sha256':hashlib.sha256((store.root/f'{symbol}_{origin}_listed_contracts.csv').read_bytes()).hexdigest(),
              'bulk_errors':errors,'price_interpolation':False,
              'selection': 'actual 09:45 quotes; malformed expiry bulk requests use nearest valid call/put found in ascending strike distance'}
        target.with_suffix('.json').write_text(json.dumps(meta,indent=2));audit.append(meta)
        print(f'Repaired {symbol} {origin}: {len(combined)} actual rows, {len(set(components))} component requests',flush=True)
    (OUT/'snapshot_repair_audit.json').write_text(json.dumps(audit,indent=2))


if __name__=='__main__':main()
