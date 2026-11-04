"""One authenticated client for monthly decision-quote downloads.

Keep network concurrency separate from QuantLib pricing and sanitize RPC
errors before crossing a process boundary. Existing decision logic is reused.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import time
import pandas as pd
from .theta_sdk import ROOT,connect,configured_key
from .stock_market import MarketStore
from .monthly_single_leg_acquire import OUT,DATA,origins


def main():
    plan=json.loads((ROOT/'monthly_single_leg_plan.json').read_text())
    cache=ROOT/'data/stock_study/quotes';dest=DATA/'decision_quotes';dest.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for origin in origins(plan):
        for symbol in plan['universe']:
            if (OUT/'decisions'/f'{symbol}_{origin}.json').exists():continue
            month=cache/f'{symbol}_stock_{pd.Timestamp(origin):%Y%m}.csv'
            name=f'{symbol}_{origin}_stock_decision'
            if not month.exists() and not (dest/(name+'.csv')).exists():
                jobs.append((name,'stock_history_quote',{'symbol':symbol,'date':pd.Timestamp(origin).date(),
                    'interval':'15m','start_time':'09:45:00','end_time':'09:46:00','venue':'utp_cta'}))
            name=f'{symbol}_{origin}_full_origin_snapshot'
            if not (cache/(name+'.csv')).exists() and not (dest/(name+'.csv')).exists():
                jobs.append((name,'option_history_quote',{'symbol':symbol,'date':pd.Timestamp(origin).date(),
                    'expiration':'*','interval':'15m','start_time':'09:45:00','end_time':'09:46:00','max_dte':115}))
    print(f'Missing decision quote files: {len(jobs)}',flush=True)
    client=None
    for attempt in range(4):
        try:client=connect();break
        except Exception as exc:
            print(json.dumps({'stage':'authentication','attempt':attempt+1,'error_type':type(exc).__name__}),flush=True)
            if attempt<3:time.sleep([2,5,10][attempt])
    if client is None:raise RuntimeError('Data authentication network connection failed after retries')
    store=MarketStore(dest,client);failed=[];completed=0
    redactions=[configured_key(),getattr(client,'email',''),getattr(getattr(client,'auth_token',None),'session_uuid','')]
    def fetch(job):
        name,method,params=job;error={}
        for attempt in range(4):
            try:
                frame=store.fetch(name,method,**params)
                return {'name':name,'status':'acquired','rows':len(frame)}
            except Exception as exc:
                error={'name':name,'status':'failed','error_type':type(exc).__name__}
                if callable(getattr(exc,'code',None)):error['grpc_code']=str(exc.code())
                if callable(getattr(exc,'details',None)):
                    detail=exc.details()
                    for secret in redactions:
                        if secret:detail=detail.replace(str(secret),'[redacted]')
                    error['detail']=detail[:500]
                if error.get('grpc_code')=='StatusCode.INVALID_ARGUMENT':
                    break
                if attempt<3:time.sleep([.5,2,5][attempt])
        return error
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fetch,j) for j in jobs]
        for future in as_completed(futures):
            result=future.result();completed+=1
            if result['status']=='failed':failed.append(result);print(json.dumps(result),flush=True)
            if completed%20==0:print(f'Downloaded/attempted {completed}/{len(jobs)}; failures {len(failed)}',flush=True)
            (OUT/'decision_download_errors.json').write_text(json.dumps(failed,indent=2))
    print(f'Decision download complete: {completed-len(failed)}/{len(jobs)}',flush=True)


if __name__=='__main__':main()
