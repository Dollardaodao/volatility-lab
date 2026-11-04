"""Zero-mean log-return models; all variances are in decimal return units.

Daily forecasts do not resolve intraday seasonality. For partial daily intervals
we allocate that interval's variance in proportion to elapsed wall-clock time.
This approximation is explicit, including weekends inside equity close intervals.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from arch import arch_model

YEAR_SECONDS = 365.0 * 86400
MODELS = ("garch_t", "ewma", "rolling21")


def utc(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Use a timezone-aware timestamp, preferably UTC")
    return t.tz_convert("UTC").as_unit("ns")


def exchange_closes(start, end):
    cal = xcals.get_calendar("XNYS", start=str(pd.Timestamp(start).date()),
                             end=str(pd.Timestamp(end).date()))
    return pd.DatetimeIndex(cal.schedule["close"]).as_unit("ns")


def future_closes(last_close, horizon, market):
    last_close = utc(last_close)
    if market == "crypto":
        return pd.date_range(last_close + pd.Timedelta(days=1), periods=horizon, freq="D").as_unit("ns")
    if market != "equity":
        raise ValueError("market must be equity or crypto")
    closes = exchange_closes(last_close - pd.Timedelta(days=3),
                             last_close + pd.Timedelta(days=horizon * 3 + 30))
    result = closes[closes > last_close][:horizon]
    if len(result) != horizon:
        raise ValueError("Insufficient calendar coverage")
    return result


@dataclass
class Fit:
    model: str
    trained_through: pd.Timestamp
    daily_variance: np.ndarray
    diagnostics: dict


def fit_variance(prices, asof, horizon=140, model="garch_t", window=756, min_returns=504):
    asof = utc(asof)
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        raise ValueError("Prices must be indexed by their UTC availability time")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Prices must have a strictly increasing index")
    observed = prices.loc[prices.index <= asof].dropna()
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("Prices must be finite and positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    r = np.log(observed).diff().dropna().iloc[-window:]
    if len(r) < min_returns:
        raise ValueError(f"Need {min_returns} returns, got {len(r)}")
    diagnostics = {"n_returns": len(r), "window_start": str(r.index[0])}
    if model == "garch_t":
        result = arch_model(r * 100, mean="Zero", vol="GARCH", p=1, q=1,
                            dist="StudentsT", rescale=False).fit(disp="off", show_warning=False)
        if result.convergence_flag != 0:
            raise RuntimeError(f"GARCH failed to converge: {result.optimization_result.message}")
        values = result.forecast(horizon=horizon, reindex=False).residual_variance.iloc[-1].to_numpy() / 10000
        diagnostics.update({str(k): float(v) for k, v in result.params.items()})
        diagnostics["persistence"] = float(result.params["alpha[1]"] + result.params["beta[1]"])
    elif model == "ewma":
        variance = float(np.mean(r.iloc[:21].to_numpy() ** 2))
        for value in r.iloc[21:]:
            variance = .94 * variance + .06 * float(value) ** 2
        values = np.full(horizon, variance)
    elif model == "rolling21":
        values = np.full(horizon, float(np.mean(r.iloc[-21:].to_numpy() ** 2)))
    else:
        raise ValueError(f"Unknown model {model}")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise RuntimeError("Invalid variance forecast")
    return Fit(model, observed.index[-1], values, diagnostics)


def expiry_forecast(fit, asof, expiry, market):
    asof, expiry = utc(asof), utc(expiry)
    if fit.trained_through > asof or expiry <= asof:
        raise ValueError("Require trained_through <= asof < expiry")
    ends = future_closes(fit.trained_through, len(fit.daily_variance), market)
    if ends[-1] < expiry:
        raise ValueError("Increase the fitted forecast horizon to cover expiry")
    starts = pd.DatetimeIndex([fit.trained_through, *ends[:-1]]).as_unit("ns")
    overlap = np.maximum(0., (np.minimum(ends.asi8, expiry.value) -
                              np.maximum(starts.asi8, asof.value)) / 1e9)
    fraction = overlap / ((ends.asi8 - starts.asi8) / 1e9)
    total = float(np.dot(fit.daily_variance, fraction))
    years = (expiry - asof).total_seconds() / YEAR_SECONDS
    return {"asof": asof, "expiry": expiry, "trained_through": fit.trained_through,
            "model": fit.model, "integrated_variance": total,
            "annualized_vol": float(np.sqrt(total / years)),
            "equivalent_daily_intervals": float(fraction.sum())}


def walk_forward(prices, market, start="2024-01-01", end="2025-12-31", stride=21):
    """Monthly-ish origins. Labels overlap at long horizons; no t-stat claims."""
    start = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    positions = np.flatnonzero((prices.index >= start) & (prices.index < end))[::stride]
    rows, errors = [], []
    log_prices = np.log(prices)
    for i in positions:
        asof = prices.index[i]
        for model in MODELS:
            try:
                fit = fit_variance(prices, asof, horizon=63, model=model)
                for h in (5, 21, 63):
                    if i + h >= len(prices) or prices.index[i + h] >= end:
                        continue
                    expiry = prices.index[i + h]
                    expected = future_closes(asof, h, market)
                    if not expected.equals(prices.index[i + 1:i + h + 1]):
                        raise ValueError("Missing daily observations in scoring window")
                    prediction = expiry_forecast(fit, asof, expiry, market)
                    realized = float(np.sum(np.diff(log_prices.iloc[i:i+h+1].to_numpy()) ** 2))
                    tau = (expiry-asof).total_seconds()/YEAR_SECONDS
                    f = prediction["integrated_variance"] / tau
                    v = realized / tau
                    rows.append({**prediction, "horizon_sessions": h,
                                 "realized_annualized_vol": np.sqrt(v),
                                 "qlike": v/f - np.log(max(v/f, 1e-15)) - 1,
                                 "squared_vol_error": (np.sqrt(f)-np.sqrt(v)) ** 2})
            except Exception as exc:
                errors.append({"asof": str(asof), "model": model, "error": str(exc)})
    return pd.DataFrame(rows), errors
