"""Cached point-in-time quotes, daily forecasts, dividends and lagged SOFR."""
from datetime import date
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from .theta import normalize_quotes
from .forecast import exchange_closes


def calendar():
    return xcals.get_calendar("XNYS",start="2017-01-01",end="2027-12-31")


def sessions(start,end):
    return calendar().schedule.loc[str(pd.Timestamp(start).date()):str(pd.Timestamp(end).date())]


def local_time(day,clock):
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {clock}",tz="America/New_York").tz_convert("UTC")


def month_ranges(start,end):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    current=start.replace(day=1)
    while current<=end:
        finish=current+pd.offsets.MonthEnd(0)
        yield current.date(),finish.date()
        current=finish+pd.Timedelta(days=1)


class MarketStore:
    def __init__(self,root,client=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.client=client

    def fetch(self,name,method,**kwargs):
        path=self.root/(name+".csv")
        if path.exists():return pd.read_csv(path)
        frame=getattr(self.client,method)(**kwargs)
        if frame.empty:raise ValueError(f"No data for {name}")
        temporary=path.with_suffix(".tmp")
        frame.to_csv(temporary,index=False);temporary.replace(path)
        meta={"method":method,"parameters":{k:str(v) for k,v in kwargs.items()},
              "rows":len(frame),"retrieved_utc":str(pd.Timestamp.now(tz="UTC")),
              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        path.with_suffix(".json").write_text(json.dumps(meta,indent=2))
        return frame

    def stock(self,symbol,start,end):
        frames=[]
        for first,last in month_ranges(start,end):
            frames.append(self.fetch(f"{symbol}_stock_{first:%Y%m}","stock_history_quote",
                symbol=symbol,start_date=first,end_date=last,interval="15m",venue="utp_cta"))
        frame=normalize_quotes(pd.concat(frames,ignore_index=True))
        frame=frame[frame.regular_session].copy()
        frame=frame.drop_duplicates("sample_time_utc",keep="last").set_index("sample_time_utc").sort_index()
        return frame.loc[local_time(start,"09:30"):local_time(end,"16:00")]

    def origin_snapshot(self,symbol,origin):
        raw=self.fetch(f"{symbol}_{origin}_full_origin_snapshot","option_history_quote",
            symbol=symbol,date=pd.Timestamp(origin).date(),expiration="*",interval="15m",
            start_time="09:45:00",end_time="09:46:00",max_dte=115)
        frame=normalize_quotes(raw)
        return frame[(frame.sample_time_utc==local_time(origin,"09:45"))&frame.passes_snapshot_filters]

    def options(self,symbol,origin,expiry,strike):
        frames=[]
        for first,last in month_ranges(origin,expiry):
            first=max(first,pd.Timestamp(origin).date());last=min(last,pd.Timestamp(expiry).date())
            name=f"{symbol}_{expiry}_{strike:.3f}_{first}_{last}_options"
            frames.append(self.fetch(name,"option_history_quote",symbol=symbol,
                expiration=pd.Timestamp(expiry).date(),strike=f"{strike:.3f}",right="both",
                start_date=first,end_date=last,interval="15m"))
        frame=normalize_quotes(pd.concat(frames,ignore_index=True))
        return frame[frame.regular_session].copy()


class DailyInputs:
    def __init__(self,root,symbol):
        frame=pd.read_csv(Path(root)/f"{symbol}_daily_actions.csv")
        frame["day"]=pd.to_datetime(frame.iloc[:,0],utc=True).dt.tz_convert("America/New_York").dt.date
        close_map={t.date():t for t in exchange_closes("2017-12-20","2026-11-10")}
        self.prices=pd.Series(frame.Close.to_numpy(),index=pd.DatetimeIndex([close_map[d] for d in frame.day]).as_unit("ns"))
        self.splits={d:float(v) for d,v in zip(frame.day,frame["Stock Splits"]) if v>0 and v!=1}
        self.dividends={}
        for day,amount in zip(frame.day,frame.Dividends):
            if amount>0:
                # Yahoo dividends are adjusted for later splits. Undo those
                # unit adjustments to recover the per-share cash amount then.
                factor=np.prod([r for d,r in self.splits.items() if d>day])
                self.dividends[day]=float(amount*factor)

    def yield_at(self,asof,spot):
        day=pd.Timestamp(asof).tz_convert("America/New_York").date()
        lower=day-pd.Timedelta(days=365)
        total=0.
        for d,v in self.dividends.items():
            if lower<=d<day:
                factor=np.prod([r for sd,r in self.splits.items() if d<sd<=day])
                total+=v/factor
        return total/spot


class RateInputs:
    def __init__(self,path):
        raw=json.loads(Path(path).read_text())["refRates"]
        self.rows=sorted((pd.Timestamp(r["effectiveDate"]).date(),float(r["percentRate"])/100) for r in raw)

    def at(self,asof):
        # A two-calendar-day lag conservatively avoids assuming same-day SOFR
        # publication. Financing and pricing only see previously observed rates.
        cutoff=pd.Timestamp(asof).tz_convert("America/New_York").date()-pd.Timedelta(days=2)
        candidates=[v for d,v in self.rows if d<=cutoff]
        if not candidates:raise ValueError("No past rate available")
        return candidates[-1]


def select_contracts(snapshot,stock,origin,targets,tolerances):
    decision=local_time(origin,"09:45")
    s=stock.loc[decision]
    if not bool(s.passes_snapshot_filters):raise ValueError("Invalid stock at the origin")
    spot=float((s.bid+s.ask)/2)
    snapshot=snapshot[snapshot.sample_time_utc==decision].copy()
    snapshot["expiry_date"]=pd.to_datetime(snapshot.expiration).dt.date
    selected=[]
    for target,tolerance in zip(targets,tolerances):
        eligible=[]
        for expiry,f in snapshot.groupby("expiry_date"):
            dte=(expiry-pd.Timestamp(origin).date()).days
            if dte<=1 or abs(dte-target)>tolerance:continue
            common=set(f.loc[f.right=="CALL","strike"])&set(f.loc[f.right=="PUT","strike"])
            if common:eligible.append((abs(dte-target),expiry,min(common,key=lambda k:abs(k-spot))))
        if not eligible:
            selected.append({"target_dte":target,"status":"no_eligible_origin_quotes"});continue
        _,expiry,strike=min(eligible,key=lambda x:(x[0],x[1],abs(x[2]-spot)))
        schedule=sessions(origin,expiry)
        expiry_time=schedule.iloc[-1]["close"]
        previous=schedule.iloc[-2]["close"]
        exit_time=previous-pd.Timedelta(minutes=15)
        # The equity early-close schedule is used for both expiry and exit.
        selected.append({"target_dte":target,"expiry_date":str(expiry),"expiry_time":str(expiry_time),
                         "planned_exit":str(exit_time),"strike":float(strike),"decision_time":str(decision),
                         "decision_spot":spot,"actual_calendar_dte":(expiry_time-decision).total_seconds()/86400,
                         "status":"selected"})
    return selected


def assemble_panel(stock,option,contract):
    panel=stock.copy()
    panel["spot"]=(panel.bid+panel.ask)/2
    panel=panel.rename(columns={x:"stock_"+x for x in ("bid","ask","bid_size","ask_size","passes_snapshot_filters")})
    for kind in ("call","put"):
        frame=option[option.right==kind.upper()].drop_duplicates("sample_time_utc",keep="last").set_index("sample_time_utc")
        cols=["bid","ask","bid_size","ask_size","passes_snapshot_filters","conservative_condition_ok"]
        panel=panel.join(frame[cols].rename(columns={x:kind+"_"+x for x in cols}),how="left")
        panel[kind+"_mid"]=(panel[kind+"_bid"]+panel[kind+"_ask"])/2
    panel["options_valid"]=panel.call_passes_snapshot_filters.fillna(False)&panel.put_passes_snapshot_filters.fillna(False)
    panel["all_valid"]=panel.options_valid&panel.stock_passes_snapshot_filters
    start=pd.Timestamp(contract["decision_time"])
    expiry=pd.Timestamp(contract["expiry_time"])
    panel=panel.loc[start:expiry].copy()
    panel["years_to_expiry"]=(expiry-panel.index).total_seconds()/(365*86400)
    return panel
