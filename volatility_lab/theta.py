"""Bounded first-day quote audit via a user-authenticated local Theta Terminal.

No account creation, payments, credentials or remote authentication are handled.
This is a data acquisition adapter, not an American-option backtest engine.
"""
import argparse
from io import StringIO
import json
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import requests
import exchange_calendars as xcals
from .data import record_file


def normalize_quotes(frame, source_timezone="America/New_York"):
    required={"timestamp","bid","ask","bid_size","ask_size"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing expected Theta columns: {sorted(required-set(frame.columns))}")
    frame=frame.copy()
    first=pd.Timestamp(frame["timestamp"].dropna().iloc[0])
    if first.tzinfo is not None:
        timestamp=pd.to_datetime(frame["timestamp"],errors="raise",format="mixed",utc=True)
    else:
        timestamp=pd.to_datetime(frame["timestamp"],errors="raise",format="mixed")
        timestamp=timestamp.dt.tz_localize(source_timezone,ambiguous="raise",nonexistent="raise")
    frame["sample_time_utc"]=timestamp.dt.tz_convert("UTC")
    # Preserve provider sizes/conditions verbatim. Do not assume stock NBBO
    # size units or interval timestamps are original tick availability times.
    for col in ("bid","ask","bid_size","ask_size"):
        frame[col]=pd.to_numeric(frame[col],errors="raise")
    frame["valid_two_sided"]=(frame.bid>0)&(frame.ask>=frame.bid)&(frame.bid_size>0)&(frame.ask_size>0)
    frame["original_quote_time_verified"]=False
    # Resampled APIs can repeat a quote after an early exchange close. Keep
    # raw rows, but do not admit those snapshots into the research panel.
    local_dates=frame["sample_time_utc"].dt.tz_convert("America/New_York").dt.date
    cal=xcals.get_calendar("XNYS",start=str(pd.Timestamp(min(local_dates))-pd.Timedelta(days=7)),
                           end=str(pd.Timestamp(max(local_dates))+pd.Timedelta(days=7)))
    openings={t.date():row["open"] for t,row in cal.schedule.iterrows()}
    closings={t.date():row["close"] for t,row in cal.schedule.iterrows()}
    opened=pd.to_datetime(local_dates.map(openings),utc=True)
    closed=pd.to_datetime(local_dates.map(closings),utc=True)
    frame["regular_session"]=(frame.sample_time_utc>=opened)&(frame.sample_time_utc<=closed)
    # Conservative initial subset of the vendor's firm condition codes:
    # REGULAR=0, BID_ASK_AUTO_EXEC=1, NATIONAL_BBO=50. Preserve all raw codes.
    conditions=["bid_condition","ask_condition"]
    frame["conservative_condition_ok"]=False
    if all(c in frame.columns for c in conditions):
        frame["conservative_condition_ok"]=frame.bid_condition.isin([0,1,50])&frame.ask_condition.isin([0,1,50])
    frame["passes_snapshot_filters"]=frame.valid_two_sided&frame.regular_session&frame.conservative_condition_ok
    return frame.sort_values("sample_time_utc")


class ThetaClient:
    def __init__(self,base_url="http://127.0.0.1:25503/v3"):
        parsed=urlparse(base_url)
        if parsed.hostname not in ("127.0.0.1","localhost","::1") or parsed.scheme!="http":
            raise ValueError("Use the local Theta Terminal HTTP endpoint")
        self.base_url=base_url.rstrip("/")
        self.session=requests.Session()

    def csv(self,endpoint,params,destination):
        try:
            response=self.session.get(f"{self.base_url}/{endpoint}",params={**params,"format":"csv"},timeout=90)
        except requests.ConnectionError as exc:
            raise RuntimeError("Theta Terminal is not reachable at 127.0.0.1:25503. Start the vendor's v3 Terminal and sign in locally with the required quote entitlements.") from exc
        response.raise_for_status()
        if len(response.content)>100_000_000:
            raise ValueError("First-day audit response exceeds 100 MB; narrow the request")
        frame=pd.read_csv(StringIO(response.text))
        if frame.empty:
            raise ValueError("No observations returned; check date and account entitlements")
        Path(destination).write_bytes(response.content)
        return frame,record_file(destination,response.url)

    def audit_day(self,symbol,date,directory):
        if symbol not in ("AAPL","MSFT","NVDA"):
            raise ValueError("Initial coverage audit is restricted to AAPL, MSFT and NVDA")
        date=pd.Timestamp(date).strftime("%Y%m%d")
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        common={"symbol":symbol,"date":date,"interval":"15m"}
        sources,summary=[],{}
        for market,endpoint,params in (
            ("stock","stock/history/quote",{**common,"venue":"utp_cta"}),
            ("options","option/history/quote",{**common,"expiration":"*","strike_range":2,"max_dte":120}),
        ):
            frame,source=self.csv(endpoint,params,directory/f"{symbol}_{date}_{market}_raw.csv")
            normalized=normalize_quotes(frame)
            normalized.to_csv(directory/f"{symbol}_{date}_{market}_normalized.csv",index=False)
            sources.append(source)
            summary[market]={"rows":len(frame),"valid_two_sided":int(normalized.valid_two_sided.sum()),
                             "first_sample_utc":str(normalized.sample_time_utc.iloc[0]),
                             "last_sample_utc":str(normalized.sample_time_utc.iloc[-1])}
        result={"symbol":symbol,"date":date,"interval":"15m","coverage_audit":summary,
                "sources":sources,"downloaded_at_utc":str(pd.Timestamp.now(tz="UTC")),
                "status":"Acquired samples; quote-condition, timestamp semantics, size units and contract metadata still require audit before stock trading simulation"}
        (directory/f"{symbol}_{date}_audit.json").write_text(json.dumps(result,indent=2))
        return result


def main():
    p=argparse.ArgumentParser(description="Download one bounded day for an initial Theta quote audit")
    p.add_argument("--symbol",choices=("AAPL","MSFT","NVDA"),default="AAPL")
    p.add_argument("--date",default="2024-11-04")
    p.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"data"/"theta")
    args=p.parse_args()
    try:
        print(json.dumps(ThetaClient().audit_day(args.symbol,args.date,args.output),indent=2))
    except (RuntimeError,ValueError,requests.RequestException) as exc:
        p.exit(1,str(exc)+"\n")


if __name__=="__main__":main()
