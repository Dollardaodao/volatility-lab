"""Official Python SDK access without a Terminal; never log credentials.

Usage: .venv/bin/python -m volatility_lab.theta_sdk
Requires Python 3.12+, the theta extra, and a locally entered API key.
"""
import argparse
from datetime import date
import json
import logging
import os
from pathlib import Path
from dotenv import dotenv_values
import pandas as pd
from .data import record_file
from .theta import normalize_quotes

ROOT=Path(__file__).resolve().parents[1]


def configured_key(path=ROOT/".env"):
    """Read only the named credential. Do not print or return it in a report."""
    key=(dotenv_values(path).get("THETADATA_API_KEY") or os.environ.get("THETADATA_API_KEY") or "").strip()
    if not key or key in ("YOUR_API_KEY","your_api_key_here","你的key"):
        raise ValueError("Add THETADATA_API_KEY to the project's local .env file, then run again.")
    return key


def connect(path=ROOT/".env"):
    key=configured_key(path)
    # The vendor SDK logs an authentication response at INFO level. Disable
    # its loggers before constructing the client so session tokens stay local.
    logging.getLogger("thetadata").disabled=True
    logging.getLogger("thetadata.client").disabled=True
    from thetadata import ThetaClient
    return ThetaClient(api_key=key,dataframe_type="pandas",mdds_type="PROD",
                       auth_url="https://nexus-api.thetadata.us/identity/terminal/auth_user",
                       mdds_host="mdds-01.thetadata.us",mdds_port="443")


def check_access(client,directory):
    """One recent EOD observation is used to validate the free access path."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    sample=date(2026,10,27)
    frame=client.stock_history_eod(symbol="AAPL",start_date=sample,end_date=sample)
    if not isinstance(frame,pd.DataFrame) or frame.empty:
        raise ValueError("No EOD observation returned")
    path=directory/"AAPL_20261027_eod_sdk.csv"
    frame.to_csv(path,index=False)
    # Restrict output to entitlements, data counts and source metadata.
    result={"authenticated":True,"eod_rows":len(frame),
            "stock_subscription":str(client.stock_subscription),
            "options_subscription":str(client.options_subscription),
            "historical_quote_access_tested":False,
            "source":record_file(path,"ThetaData official Python SDK stock_history_eod; AAPL; 2026-10-27")}
    (directory/"sdk_access_check.json").write_text(json.dumps(result,indent=2))
    return result


def audit_day(client,symbol,sample,expiration,directory):
    """Bounded quote audit; concrete expiry avoids wildcard SDK ambiguity."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    result={"symbol":symbol,"date":str(sample),"expiration":str(expiration),"sources":[]}
    for market in ("stock","options"):
        if market=="stock":
            frame=client.stock_history_quote(symbol=symbol,date=sample,interval="15m",venue="utp_cta")
        else:
            frame=client.option_history_quote(symbol=symbol,date=sample,expiration=expiration,
                                               interval="15m",strike_range=2)
        if frame.empty:
            raise ValueError("No quotes returned")
        path=directory/f"{symbol}_{sample}_{market}_sdk_raw.csv"
        frame.to_csv(path,index=False)
        normalized=normalize_quotes(frame)
        normalized.to_csv(directory/f"{symbol}_{sample}_{market}_sdk_normalized.csv",index=False)
        result[market]={"rows":len(frame),"valid_two_sided":int(normalized.valid_two_sided.sum())}
        result["sources"].append(record_file(path,f"ThetaData official Python SDK {market}_history_quote"))
    (directory/f"{symbol}_{sample}_sdk_quote_audit.json").write_text(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description="Check Theta API access without exposing the key")
    p.add_argument("--audit",action="store_true",help="Request one day of historical quotes after checking paid entitlements")
    p.add_argument("--symbol",choices=("AAPL","MSFT","NVDA"),default="AAPL")
    p.add_argument("--date",type=date.fromisoformat,default=date(2024,11,4))
    p.add_argument("--expiration",type=date.fromisoformat,default=date(2024,11,8))
    args=p.parse_args()
    # Fail before SDK import or network access if the user has not configured
    # the credential. Raw SDK exceptions are deliberately never echoed.
    try:
        configured_key()
    except ValueError:
        p.exit(2,"Please fill THETADATA_API_KEY in the local .env file. No connection attempted.\n")
    try:
        client=connect()
        result=check_access(client,ROOT/"data"/"theta")
        if args.audit:
            result["quote_audit"]=audit_day(client,args.symbol,args.date,args.expiration,ROOT/"data"/"theta")
        print(json.dumps(result,indent=2))
    except Exception as exc:
        p.exit(1,f"Theta access check did not complete ({type(exc).__name__}). No credential or raw server response is printed.\n")


if __name__=="__main__":main()
