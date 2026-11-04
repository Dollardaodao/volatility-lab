"""Public data adapters with raw caches, UTC availability, and provenance."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from .forecast import exchange_closes, future_closes

SYMBOLS = {"AAPL": "equity", "MSFT": "equity", "NVDA": "equity",
           "BTC-USD": "crypto", "ETH-USD": "crypto"}


def record_file(path, source):
    path = Path(path)
    return {"file": str(path), "source": source, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def daily_history(directory, refresh=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "yahoo_daily_close.csv"
    if refresh or not cache.exists():
        frame = yf.download(list(SYMBOLS), start="2018-01-01", end="2026-11-03",
                            auto_adjust=False, progress=False, threads=False)
        if frame.empty:
            raise RuntimeError("Yahoo returned no daily data")
        frame["Close"].to_csv(cache)
    close = pd.read_csv(cache, index_col=0, parse_dates=True)
    schedule = exchange_closes("2017-12-20", "2027-01-01")
    close_map = {t.date(): t for t in schedule}
    result = {}
    for symbol, market in SYMBOLS.items():
        series = close[symbol].dropna().copy()
        dates = pd.DatetimeIndex(series.index)
        if market == "crypto":
            series.index = pd.DatetimeIndex([pd.Timestamp(t.date(), tz="UTC") +
                                            pd.Timedelta(days=1) for t in dates])
        else:
            series.index = pd.DatetimeIndex([close_map[t.date()] for t in dates])
        series.index = series.index.as_unit("ns")
        series = series.sort_index()
        expected = future_closes(series.index[0],len(series)-1,market)
        if not expected.equals(series.index[1:]):
            raise ValueError(f"{symbol}: missing or unexpected daily closes; repair the input before fitting")
        series.name = symbol
        result[symbol] = series
    return result, record_file(cache, "Yahoo Finance via yfinance; Close, daily; end=2026-11-03 exclusive")


def fetch_tardis_sample(directory, symbol, date="2024-03-01"):
    if pd.Timestamp(date).day != 1:
        raise ValueError("This adapter only requests free first-of-month samples")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{symbol}.csv.gz"
    url = f"https://datasets.tardis.dev/v1/deribit/options_chain/{date.replace('-', '/')}/{symbol}.csv.gz"
    if not path.exists():
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        if len(response.content) > 40_000_000:
            raise ValueError("Unexpectedly large per-instrument sample")
        path.write_bytes(response.content)
    frame = pd.read_csv(path)
    if frame.empty or not (frame.symbol == symbol).all():
        raise ValueError(f"Missing or mismatched instrument: {symbol}")
    for col in ("timestamp", "local_timestamp", "expiration"):
        frame[col] = pd.to_datetime(frame[col], unit="us", utc=True)
    frame["available_at"] = frame[["timestamp", "local_timestamp"]].max(axis=1)
    return frame.sort_values("available_at"), record_file(path, url)


def coinbase_bars(directory, symbol, date="2024-03-01"):
    path = Path(directory) / f"coinbase_{symbol}_{date}_5min.json"
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    start = pd.Timestamp(date, tz="UTC")
    end = start + pd.Timedelta(days=1)
    if not path.exists():
        response = requests.get(url, params={"start": start.isoformat(), "end": end.isoformat(),
                                             "granularity": 300}, timeout=30)
        response.raise_for_status()
        path.write_text(json.dumps(response.json()))
    frame = pd.DataFrame(json.loads(path.read_text()), columns=["start", "low", "high", "open", "close", "volume"])
    frame["start"] = pd.to_datetime(frame["start"], unit="s", utc=True)
    frame = frame[(frame.start >= start) & (frame.start < end)].sort_values("start")
    # Candle close becomes observable at the end of its bucket, never its start.
    frame["time"] = frame.start + pd.Timedelta(minutes=5)
    return frame, record_file(path, url + f"?granularity=300&start={start.isoformat()}&end={end.isoformat()}")


def archived_instrument(directory,symbol):
    path=Path(directory)/f"{symbol}_instrument.json"
    url="https://history.deribit.com/api/v2/public/get_instrument"
    if not path.exists():
        r=requests.get(url,params={"instrument_name":symbol},timeout=25)
        r.raise_for_status()
        payload=r.json()
        if "error" in payload:
            raise ValueError(str(payload["error"]))
        path.write_text(json.dumps(payload,indent=2))
    metadata=json.loads(path.read_text())["result"]
    if metadata["instrument_name"]!=symbol or metadata["contract_size"]!=1:
        raise ValueError("This pilot adapter requires verified unit-size inverse contracts")
    return metadata,record_file(path,url+"?instrument_name="+symbol)


def align_sample(call, put, bars, max_age_seconds=60):
    panel = bars[["time", "close"]].rename(columns={"close": "spot"}).copy()
    for name, frame in (("call", call), ("put", put)):
        cols = ["available_at", "timestamp", "bid_price", "ask_price", "bid_amount", "ask_amount", "expiration", "strike_price", "symbol"]
        right = frame[cols].rename(columns={c: f"{name}_{c}" for c in cols})
        panel = pd.merge_asof(panel, right, left_on="time", right_on=f"{name}_available_at", direction="backward",
                              tolerance=pd.Timedelta(seconds=max_age_seconds))
        panel[f"{name}_age_seconds"] = (panel.time-panel[f"{name}_timestamp"]).dt.total_seconds()
        panel.loc[panel[f"{name}_age_seconds"] > max_age_seconds,
                  [f"{name}_bid_price", f"{name}_ask_price"]] = np.nan
    panel = panel.set_index("time")
    # Preserve gaps for audit. The engine rejects missing execution/hedging inputs.
    return panel
