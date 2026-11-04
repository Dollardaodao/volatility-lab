"""Price/replay ready contracts while the one-client downloader is working."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse
import hashlib
import json
import time
from .theta_sdk import ROOT
from .monthly_single_leg_acquire import OUT
from .monthly_single_leg_study import RAW,quote_jobs,run_case


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    plan=json.loads((ROOT/'monthly_single_leg_plan.json').read_text())
    decisions=json.loads((OUT/'decisions.json').read_text());selected=[c for c in decisions if c['status']=='selected']
    for c in selected:c['case_id']=f'{c["asset"]}_{c["origin"]}_{c["target_dte"]}d_{c["kind"]}'
    jobs,mapping=quote_jobs(selected)
    sources={n:hashlib.sha256((ROOT/'volatility_lab'/n).read_bytes()).hexdigest() for n in
             ['monthly_single_leg_study.py','single_leg_replay.py','american.py','stock_market.py']}
    (OUT/'prefill_source_hashes.json').write_text(json.dumps(sources,indent=2))
    pending={c['case_id']:c for c in sorted(selected,key=lambda c:-c['target_dte'])};active={};finished=0;errors=[];last_print=0.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while pending or active:
            for name,c in list(pending.items()):
                if len(active)>=args.workers:break
                files=mapping[name]
                if not all((RAW/(n+'.csv')).exists() for n in files['stock']+files['option']):continue
                future=pool.submit(run_case,c,files,plan);active[future]=name;del pending[name]
            if active:
                done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                for future in done:
                    name=active.pop(future)
                    try:future.result();finished+=1
                    except Exception as exc:
                        errors.append({'case_id':name,'error_type':type(exc).__name__,'reason':str(exc)})
                        print(json.dumps(errors[-1]),flush=True)
            else:time.sleep(2)
            now=time.monotonic()
            if now-last_print>=20 or not pending and not active:
                print(f'Priced/replayed {finished}/{len(selected)}; running {len(active)}; waiting for quotes {len(pending)}; failures {len(errors)}',flush=True)
                (OUT/'prefill_errors.json').write_text(json.dumps(errors,indent=2));last_print=now
    print('Ready-contract replay complete',flush=True)


if __name__=='__main__':main()
