"""Acquire the predeclared multi-stock, multi-period, multi-tenor study grid."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
from pathlib import Path
import pandas as pd
from .theta_sdk import connect,ROOT
from .stock_market import MarketStore,DailyInputs,select_contracts,assemble_panel


def acquire_one(store,symbol,origin,plan,root):
    job=root/f"{symbol}_{origin}_cohorts.json"
    if job.exists():return json.loads(job.read_text())
    daily=DailyInputs(root,symbol)
    day_stock=store.stock(symbol,origin,origin)
    snapshot=store.origin_snapshot(symbol,origin)
    contracts=select_contracts(snapshot,day_stock,origin,plan["target_calendar_dte"],plan["dte_tolerance_days"])
    eligible=[c for c in contracts if c["status"]=="selected"]
    if eligible:
        stock=store.stock(symbol,origin,max(c["expiry_date"] for c in eligible))
    for contract in contracts:
        contract.update({"asset":symbol,"origin":origin})
        if contract["status"]!="selected":continue
        try:
            split_dates=[str(d) for d in daily.splits if pd.Timestamp(origin).date()<d<=pd.Timestamp(contract["expiry_date"]).date()]
            if split_dates:
                contract.update({"status":"unsupported_split_in_holding_window","split_dates":split_dates});continue
            option=store.options(symbol,origin,contract["expiry_date"],contract["strike"])
            panel=assemble_panel(stock,option,contract)
            path=root/f'{symbol}_{origin}_{contract["target_dte"]}d_panel.csv'
            panel.to_csv(path)
            contract.update({"status":"acquired","panel_file":str(path),"stock_grid_rows":len(panel),
                             "valid_paired_snapshots":int(panel.all_valid.sum())})
        except Exception as exc:
            contract.update({"status":"acquisition_failed","error_type":type(exc).__name__})
            if callable(getattr(exc,"code",None)):contract["grpc_status"]=str(exc.code())
    job.write_text(json.dumps(contracts,indent=2))
    return contracts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",type=Path,default=ROOT/"stock_study_plan.json")
    parser.add_argument("--manifest",type=Path,default=ROOT/"artifacts"/"stock_study_acquisition.json")
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    root=ROOT/"data"/"stock_study"
    store=MarketStore(root/"quotes",connect())
    results=[];errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(acquire_one,store,symbol,origin,plan,root):(symbol,origin)
              for symbol in plan["universe"] for origin in plan["origins"]}
        for future in as_completed(jobs):
            symbol,origin=jobs[future]
            try:
                rows=future.result();results.extend(rows)
                print(json.dumps({"asset":symbol,"origin":origin,"cohorts":{str(r["target_dte"]):r["status"] for r in rows}}),flush=True)
            except Exception as exc:
                error={"asset":symbol,"origin":origin,"error_type":type(exc).__name__}
                if callable(getattr(exc,"code",None)):error["grpc_status"]=str(exc.code())
                errors.append(error);print(json.dumps(error),flush=True)
            args.manifest.parent.mkdir(parents=True,exist_ok=True)
            args.manifest.write_text(json.dumps({"cohorts":results,"errors":errors},indent=2))
    print(f"Acquisition finished: {len(results)} cohort records, {len(errors)} job errors",flush=True)


if __name__=="__main__":main()
