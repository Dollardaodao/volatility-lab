"""Run from the project directory: .venv/bin/python -m volatility_lab.run."""
import argparse
from dataclasses import replace
import html
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from .data import SYMBOLS, daily_history, fetch_tardis_sample, coinbase_bars, align_sample, archived_instrument, record_file
from .forecast import MODELS, fit_variance, expiry_forecast, walk_forward, exchange_closes
from .backtest import Costs, enrich_panel, run_pilot

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def deribit_snapshot(prices, out):
    """Current, point-in-time comparison, never fed into historical backtests."""
    rows, errors, raw = [], [], {}
    session = requests.Session()

    def rpc(method, **params):
        r = session.get(f"https://www.deribit.com/api/v2/public/{method}", params=params, timeout=25)
        r.raise_for_status()
        payload = r.json()
        if "error" in payload:
            raise RuntimeError(str(payload["error"]))
        raw[method+json.dumps(params,sort_keys=True)] = payload
        return payload["result"]

    for coin in ("BTC","ETH"):
        try:
            meta = pd.DataFrame(rpc("get_instruments",currency=coin,kind="option",expired="false"))
            meta["expiry"] = pd.to_datetime(meta.expiration_timestamp,unit="ms",utc=True)
            book = pd.DataFrame(rpc("get_book_summary_by_currency",currency=coin,kind="option"))
            reference_spot = float(book.underlying_price.dropna().median())
            observed = pd.Timestamp.now(tz="UTC")
            expiries = sorted(x for x in meta.expiry.unique() if x > observed)
            selected = sorted(set(min(expiries,key=lambda x:abs((x-observed).total_seconds()/86400-target)) for target in (7,30,90)))
            for expiry in selected:
                subset = meta[meta.expiry == expiry]
                # Select on currently observed reference prices, without future outcomes.
                strikes = set(subset.loc[subset.option_type=="call","strike"]) & set(subset.loc[subset.option_type=="put","strike"])
                strike = min(strikes,key=lambda x:abs(x-reference_spot))
                pair = subset[subset.strike==strike]
                quotes = {}
                for kind in ("call","put"):
                    name = pair.loc[pair.option_type==kind,"instrument_name"].iloc[0]
                    quotes[kind] = rpc("ticker",instrument_name=name)
                asof = max(pd.to_datetime(x["timestamp"],unit="ms",utc=True) for x in quotes.values())
                for quote in quotes.values():
                    if (asof-pd.to_datetime(quote["timestamp"],unit="ms",utc=True)).total_seconds()>60:
                        raise ValueError("Live call and put snapshots too far apart")
                fit = fit_variance(prices[f"{coin}-USD"],asof,horizon=400)
                prediction = expiry_forecast(fit,asof,expiry,"crypto")
                row = {"asset":coin,"strike":strike,**prediction,
                       "forecast_data_age_hours":(asof-fit.trained_through).total_seconds()/3600}
                for kind,q in quotes.items():
                    for key in ("instrument_name","best_bid_price","best_ask_price","bid_iv","ask_iv","mark_iv","index_price"):
                        row[f"{kind}_{key}"] = q.get(key)
                row["snapshot_only"] = True
                rows.append(row)
        except Exception as exc:
            errors.append({"asset":coin,"error":str(exc)})
    dump(out/"deribit_raw_snapshot.json",raw)
    pd.DataFrame(rows).to_csv(out/"deribit_expiry_comparison.csv",index=False)
    return rows, errors


def stock_snapshot(prices,out):
    """Indicative Yahoo chains. Last-trade timestamps are NOT quote timestamps."""
    rows,errors=[],[]
    for symbol in ("AAPL","MSFT","NVDA"):
        try:
            ticker=yf.Ticker(symbol)
            dates=list(ticker.options)
            observed=pd.Timestamp.now(tz="UTC")
            close_map={str(t.date()):t for t in exchange_closes(observed-pd.Timedelta(days=2),observed+pd.Timedelta(days=500))}
            eligible=[d for d in dates if d in close_map and close_map[d]>observed]
            selected=sorted(set(min(eligible,key=lambda d:abs((close_map[d]-observed).total_seconds()/86400-target)) for target in (7,30,90)))
            for date in selected:
                chain=ticker.option_chain(date)
                chain.calls.to_csv(out/f"{symbol}_{date}_current_calls.csv",index=False)
                chain.puts.to_csv(out/f"{symbol}_{date}_current_puts.csv",index=False)
                asof=pd.Timestamp.now(tz="UTC")
                spot=float(chain.underlying.get("regularMarketPrice",prices[symbol].iloc[-1]))
                valid={}
                for kind,frame in (("call",chain.calls),("put",chain.puts)):
                    valid[kind]=frame[(frame.bid>0)&(frame.ask>=frame.bid)&
                                      (frame.impliedVolatility>0)&(frame.contractSize=="REGULAR")]
                strikes=set(valid["call"].strike)&set(valid["put"].strike)
                if not strikes:
                    raise ValueError(f"No valid regular-size paired quotes for {date}")
                strike=min(strikes,key=lambda x:abs(x-spot))
                prediction=expiry_forecast(fit_variance(prices[symbol],asof,horizon=200),asof,close_map[date],"equity")
                row={"asset":symbol,"strike":strike,"reference_spot":spot,**prediction,
                     "quote_timestamp_verified":False,"snapshot_only":True}
                for kind in ("call","put"):
                    q=valid[kind].loc[valid[kind].strike==strike].iloc[0]
                    for key in ("contractSymbol","bid","ask","impliedVolatility","lastTradeDate","openInterest","volume"):
                        row[f"{kind}_{key}"]=q[key]
                rows.append(row)
        except Exception as exc:
            errors.append({"asset":symbol,"error":str(exc)})
    pd.DataFrame(rows).to_csv(out/"stock_expiry_comparison.csv",index=False)
    return rows,errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-daily",action="store_true")
    parser.add_argument("--skip-live",action="store_true")
    parser.add_argument("--skip-pilot",action="store_true")
    parser.add_argument("--skip-evaluation",action="store_true")
    parser.add_argument("--reuse-evaluation",action="store_true",help="Reuse cached walk-forward predictions")
    args = parser.parse_args()
    data, out = ROOT/"data", ROOT/"artifacts"
    out.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".mplconfig"))
    prices, source = daily_history(data,args.refresh_daily)
    sources, errors, coverage, terms, scores = [source], [], [], [], []
    for symbol,market in SYMBOLS.items():
        series = prices[symbol]
        asof = series.index[-1]
        coverage.append({"asset":symbol,"market":market,"observations":len(series),
                         "first_close":str(series.index[0]),"last_close":str(asof),
                         "historical_option_quote_coverage":"not acquired"})
        print(f"Forecasting {symbol}: {len(series)} daily closes",flush=True)
        for model in MODELS:
            fit = fit_variance(series,asof,horizon=150,model=model)
            for days in (7,30,90):
                terms.append({"asset":symbol,"calendar_days":days,
                              **expiry_forecast(fit,asof,asof+pd.Timedelta(days=days),market),
                              "fit_diagnostics":json.dumps(fit.diagnostics)})
        if not args.skip_evaluation and not args.reuse_evaluation:
            scored, failures = walk_forward(series,market)
            scored["asset"] = symbol
            scores.append(scored)
            errors.extend({"stage":"forecast_evaluation","asset":symbol,**e} for e in failures)
    term = pd.DataFrame(terms)
    term.to_csv(out/"forecast_term_structure.csv",index=False)
    pd.DataFrame(coverage).to_csv(out/"coverage.csv",index=False)
    score = pd.concat(scores,ignore_index=True) if scores else pd.DataFrame()
    if args.reuse_evaluation:
        score = pd.read_csv(out/"walk_forward_predictions.csv",parse_dates=["asof","expiry","trained_through"])
    summary = pd.DataFrame()
    if len(score):
        complete = score.groupby(["asset","asof","horizon_sessions"])["model"].transform("nunique") == len(MODELS)
        if not complete.all():
            errors.append({"stage":"paired_forecast_scoring","rows_removed":int((~complete).sum()),
                           "reason":"Model comparisons require all models at the same origin and horizon"})
            score = score[complete].copy()
        score.to_csv(out/"walk_forward_predictions.csv",index=False)
        summary = score.groupby(["asset","horizon_sessions","model"]).agg(
            n=("qlike","size"),mean_qlike=("qlike","mean"),
            rmse_vol=("squared_vol_error",lambda x:float(np.sqrt(np.mean(x))))).reset_index()
        summary.to_csv(out/"forecast_scores.csv",index=False)

    pilots = []
    if not args.skip_pilot:
        # A free, fixed sample date. It is an execution/accounting pilot, NOT a
        # statistically valid test of an expiry-horizon forecast trading edge.
        for coin,strike in (("BTC",62000),("ETH",3400)):
            try:
                print(f"Preparing real-quote execution pilot: {coin}",flush=True)
                call,src = fetch_tardis_sample(data,f"{coin}-29MAR24-{strike}-C");sources.append(src)
                put,src = fetch_tardis_sample(data,f"{coin}-29MAR24-{strike}-P");sources.append(src)
                call_meta,src=archived_instrument(data,f"{coin}-29MAR24-{strike}-C");sources.append(src)
                put_meta,src=archived_instrument(data,f"{coin}-29MAR24-{strike}-P");sources.append(src)
                units=max(call_meta["min_trade_amount"],put_meta["min_trade_amount"])
                bars,src = coinbase_bars(data,f"{coin}-USD");sources.append(src)
                panel = enrich_panel(align_sample(call,put,bars))
                panel.to_csv(out/f"{coin}_aligned_pilot_inputs.csv")
                decision = panel.index[0]
                expiry = panel.call_expiration.iloc[0]
                prediction = expiry_forecast(fit_variance(prices[f"{coin}-USD"],decision),decision,expiry,"crypto")
                entry_iv = float((panel.call_iv.iloc[0]+panel.put_iv.iloc[0])/2)
                # 3 vol points is an illustrative pre-set research threshold,
                # not optimized on this sample and not a calibrated cost hurdle.
                gap = prediction["annualized_vol"]-entry_iv
                model_side = 1 if gap > .03 else (-1 if gap < -.03 else 0)
                for frequency in (15,60):
                    for label,side in (("always_long",1),("always_short",-1),("garch_signal",model_side),("no_trade",0)):
                        cost_config=replace(Costs(),option_units=units,option_min_units=units,
                                            option_unit_step=units,hedge_every_minutes=frequency)
                        result,ledger,trades = run_pilot(panel,side,cost_config)
                        stem = f"{coin}_{label}_{frequency}min"
                        ledger.to_csv(out/f"{stem}_ledger.csv",index=False)
                        trades.to_csv(out/f"{stem}_fills.csv",index=False)
                        pilots.append({"asset":coin,"strategy":label,"hedge_minutes":frequency,
                                       "forecast_vol":prediction["annualized_vol"],"entry_mid_iv":entry_iv,
                                       "remaining_days":(expiry-decision).total_seconds()/86400,**result})
            except Exception as exc:
                errors.append({"stage":"execution_pilot","asset":coin,"error":str(exc)})
                print(f"Pilot {coin} rejected: {exc}",flush=True)
    dump(out/"pilot_results.json",pilots)
    pilot_frame = pd.DataFrame([{k:v for k,v in r.items() if k!="cost_assumptions"} for r in pilots])
    pilot_frame.to_csv(out/"pilot_results.csv",index=False)

    live_rows = []
    if not args.skip_live:
        print("Collecting current crypto expiry comparisons",flush=True)
        live_rows,live_errors = deribit_snapshot(prices,out)
        errors.extend({"stage":"live_crypto",**e} for e in live_errors)
        print("Collecting indicative stock expiry comparisons",flush=True)
        _,stock_errors=stock_snapshot(prices,out)
        errors.extend({"stage":"live_stock",**e} for e in stock_errors)
        # Option-chain access is probed, never confused with historical coverage.
        for row in coverage:
            if row["market"] == "equity":
                try:
                    expiries = yf.Ticker(row["asset"]).options
                    row["current_yahoo_expiries"] = len(expiries)
                except Exception as exc:
                    row["current_yahoo_expiries"] = "unavailable"
                    errors.append({"stage":"yahoo_current_chain","asset":row["asset"],"error":str(exc)})
        pd.DataFrame(coverage).to_csv(out/"coverage.csv",index=False)
    if not args.skip_live:
        for path in sorted(out.glob("*_current_*.csv")):
            sources.append(record_file(path,"Yahoo Finance via yfinance.option_chain; fetched-at timestamps in stock_expiry_comparison.csv"))
        sources.append(record_file(out/"deribit_raw_snapshot.json","https://www.deribit.com/api/v2/public/; request parameters and exchange timestamps retained in JSON"))
    dump(out/"provenance.json",{"generated_at_utc":str(pd.Timestamp.now(tz="UTC")),"sources":sources})
    dump(out/"errors.json",errors)
    render_report(out,term,summary,score,pilot_frame,coverage,live_rows,errors)
    print(f"Report: {out/'report.html'}",flush=True)
    print(f"Forecast predictions: {len(score)}; pilot scenarios: {len(pilots)}; recorded errors: {len(errors)}",flush=True)


def render_report(out,term,summary,score,pilots,coverage,live_rows,errors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor":"#faf9f6","axes.facecolor":"#faf9f6",
                         "axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    fig,axes = plt.subplots(1,2,figsize=(12,4),layout="constrained")
    for ax,market in zip(axes,("equity","crypto")):
        for symbol,m in SYMBOLS.items():
            if m==market:
                t = term[(term.asset==symbol)&(term.model=="garch_t")]
                ax.plot(t.calendar_days,t.annualized_vol*100,marker="o",label=symbol)
        ax.set(xlabel="Remaining calendar days",ylabel="Annualized volatility (%)",title=f"GARCH forecasts · {market}")
        ax.legend(frameon=False)
    fig.savefig(out/"forecast_term_structure.png",dpi=180);plt.close(fig)
    if len(score):
        fig,axes = plt.subplots(2,1,figsize=(12,6),layout="constrained")
        for ax,symbol in zip(axes,("AAPL","BTC-USD")):
            s = score[(score.asset==symbol)&(score.horizon_sessions==21)&(score.model=="garch_t")]
            ax.plot(pd.to_datetime(s["asof"]),s.annualized_vol*100,label="Forecast at origin")
            ax.plot(pd.to_datetime(s["asof"]),s.realized_annualized_vol*100,label="Subsequent realized")
            ax.set(title=f"{symbol} · next 21 observed daily intervals",ylabel="Annualized vol (%)")
            ax.legend(frameon=False,ncol=2)
        fig.savefig(out/"forecast_vs_realized.png",dpi=180);plt.close(fig)
    if len(pilots):
        fig,ax=plt.subplots(figsize=(11,4),layout="constrained")
        for coin in pilots.asset.unique():
            ledger=pd.read_csv(out/f"{coin}_garch_signal_60min_ledger.csv")
            ax.plot(pd.to_datetime(ledger.time),ledger.excess_pnl_usd,label=coin)
        ax.axhline(0,color="#555",linewidth=.7)
        ax.set(title="One-day execution pilot · not evidence of a trading edge",ylabel="P&L above cash benchmark (USD)")
        ax.legend(frameon=False)
        fig.savefig(out/"pilot_pnl.png",dpi=180);plt.close(fig)

    def table(frame):
        return frame.to_html(index=False,border=0,float_format=lambda x:f"{x:.4f}",escape=True) if len(frame) else "<p>No supported observations available; see errors below.</p>"
    forecast_table=term[term.model=="garch_t"][["asset","asof","calendar_days","annualized_vol"]].copy()
    forecast_table["annualized_vol"]*=100
    forecast_table=forecast_table.rename(columns={"annualized_vol":"forecast_vol_%"})
    pilot_table=pilots[["asset","strategy","hedge_minutes","net_excess_pnl_usd","trade_fills","reconciliation_error_usd"]] if len(pilots) else pd.DataFrame()
    score_table=summary[summary.horizon_sessions==21] if len(summary) else pd.DataFrame()
    live_table=pd.DataFrame(live_rows)
    if len(live_table):
        live_table=live_table[["asset","asof","expiry","strike","annualized_vol","call_bid_iv","call_ask_iv","put_bid_iv","put_ask_iv","forecast_data_age_hours"]].copy()
        live_table.annualized_vol*=100
        live_table=live_table.rename(columns={"annualized_vol":"forecast_vol_%"})
    stock_table=pd.DataFrame()
    stock_file=out/"stock_expiry_comparison.csv"
    if stock_file.exists() and stock_file.stat().st_size>2:
        stock_table=pd.read_csv(stock_file)[["asset","asof","expiry","strike","annualized_vol","call_bid","call_ask","call_impliedVolatility","put_bid","put_ask","put_impliedVolatility"]]
        for col in ("annualized_vol","call_impliedVolatility","put_impliedVolatility"):
            stock_table[col]*=100
        stock_table=stock_table.rename(columns={"annualized_vol":"forecast_vol_%","call_impliedVolatility":"call_IV_%","put_impliedVolatility":"put_IV_%"})
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Volatility Lab · Research Pilot</title>
<style>body{{margin:0;background:#faf9f6;color:#212b35;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1150px;margin:auto;padding:45px 30px}}h1{{font:600 42px/1.15 system-ui;margin:12px 0}}h2{{margin-top:45px}}.tag{{color:#49636a;font-size:13px;letter-spacing:2px}}.note{{padding:18px 22px;background:#edf0ed;border-left:4px solid #467264}}.warn{{background:#fff1dc;border-color:#a6762d}}img{{width:100%;margin:18px 0}}table{{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto;white-space:nowrap}}th,td{{padding:9px 12px;border-bottom:1px solid #d9dedb;text-align:left}}th{{background:#e9eeea}}a{{color:#23655a}}code{{background:#e9eeea;padding:2px 5px}}.sub{{color:#59646b}}</style><main>
<div class="tag">VOLATILITY LAB / FIRST RESEARCH BUILD</div><h1>Forecast volatility.<br>Then test what can be traded.</h1>
<p class="sub">Stocks + BTC + ETH · exact expiry endpoints · Student-t GARCH(1,1), EWMA, rolling variance</p>
<div class="note warn"><strong>No profitable strategy has been established.</strong> Forecast evaluation uses real daily history. The gamma-scalping example uses one free day of real option quotes with modeled spot execution and financing. It is an accounting pilot; it does not test held-to-expiry profitability across market regimes.</div>
<h2>1 / Data coverage and universe</h2>{table(pd.DataFrame(coverage))}
<p>AAPL, MSFT and NVDA are the initial forecast candidates because their daily histories are available. Final stock-option selection remains conditional on historical NBBO, corporate-action and event coverage. Current chains do not establish historical coverage. The chosen stocks are a convenience sample, not a point-in-time universe or an unbiased cross-sectional study.</p>
<h2>2 / Expiry-matched forecasts</h2><img src="forecast_term_structure.png" alt="Stock and crypto forecast term structures">{table(forecast_table)}
<p>Each daily conditional variance is summed across the remaining life, divided by ACT/365 time to expiry, then square-rooted. The 90-day forecast covers the entire next 90 days. Equity close intervals use the NYSE session calendar, including holidays and early closes. Partial intervals allocate variance uniformly over elapsed clock time; this daily-model approximation does not estimate intraday or earnings-specific variance. Crypto daily closes become observable at the following 00:00 UTC. Forecasts above are stamped at each asset's last available close, not represented as real-time quotes.</p>
<h2>3 / Walk-forward forecast evaluation</h2>{table(score_table)}
<p>Training uses at most 756 past returns, at least 504. Each origin refits independently with no future returns. Evaluation origins are spaced 21 observations apart in 2024–2025; 63-day labels overlap, so there are no independence-based significance claims. Lower QLIKE and volatility RMSE are better. All three models use the same origins. Predictive accuracy alone does not establish profitable options trading. Single-stock earnings and dividends are not modeled separately.</p>
{'<img src="forecast_vs_realized.png" alt="GARCH forecast versus subsequent realized volatility">' if len(score) else ''}
<h2>4 / Current option comparisons</h2><h3>Individual stocks — indicative Yahoo snapshots</h3>{table(stock_table)}
<p>Stock bid/ask premiums are USD per share, standard 100-share contracts only. Yahoo IVs are displayed in percent. Snapshot retrieval times are recorded, but quote timestamps are unavailable: lastTradeDate is a trade timestamp, not a quote timestamp. These observations are indicative comparisons, not verified executable prices or historical backtest inputs. Expiry uses the exchange's session-close timestamp; American exercise and dividends still need their own backtest treatment.</p><h3>Crypto — Deribit snapshots</h3>{table(live_table)}
<p>Bid/ask IVs are exchange percentage quotes; forecast vol is also shown in percent. These are point-in-time observations with separate call/put requests, not historical fills. Training-data age is shown explicitly. Current IV comparisons never enter the historical pilot. Selecting the nearest available expiry can produce fewer than three distinct expiries.</p>
<h2>5 / One-day gamma-scalping execution pilot</h2>{table(pilot_table)}
{'<img src="pilot_pnl.png" alt="One day of delta-hedged strategy P and L">' if len(pilots) else ''}
<p>Fixed date: 1 March 2024. Instruments: BTC March 29 $62,000 straddle and ETH March 29 $3,400 straddle, where supported by complete inputs. Each leg is 0.1 BTC or 1 ETH underlying-equivalent units, with corresponding lot-size constraints; committed capital is $100,000 per scenario. Minimum sizes and unit contract multipliers were verified using Deribit archive instrument metadata; new contracts require their own audit. The first five-minute observation determines the decision; entry occurs on the next observation and exit on the last. A 3-vol-point illustrative GARCH-versus-mid-IV threshold selects long, short or flat, without tuning. This threshold is not a calibrated execution-cost hurdle. The expiry forecast is approximately 28 days while the pilot holds for one day: this intentionally validates execution only and includes IV repricing exposure.</p>
<p><strong>Observed:</strong> option bid/ask and sizes from Tardis/Deribit, exchange and arrival timestamps; Coinbase five-minute spot candles. Backward as-of joins reject quotes more than 60 seconds old. Candle closes become available at bucket end. All hedge targets use the preceding observation; options execute using quotes after the decision. Missing values, crossed markets, size violations and data gaps reject a run.</p>
<p><strong>Modeled:</strong> spot half-spread 1 bp plus slippage 1 bp per side; spot fee 5 bps; option fee 3 bps of underlying notional capped at 12.5% of premium. These are explicit research assumptions, not a verified historical account fee schedule. Cash earns 2%, debit balances cost 6%, borrowed spot costs 8% annually. P&L is reported above the return on idle cash. A 4% pricing rate is used only to invert option mid-prices and derive USD spot deltas. Sensitivity to 15- versus 60-minute hedging is shown. No trading-cost parameter is optimized on this sample.</p>
<p>BTC/ETH-denominated option premiums are converted at the modeled spot bid/ask on entry and exit. Their USD mark is coin premium × spot. The USD cash ledger includes option and hedge spreads, conversion spread, fees, financing, borrow, and terminal liquidation. Every scenario reconciles to gross option change + hedge gains − costs + financing − cash benchmark. Cross-venue USD hedging basis, borrow availability, order-book impact and exchange margin/liquidation are not simulated; collateral uses a stated research reserve. The pilot closes before expiry and does not implement settlement.</p>
<h2>6 / What is required to test profitability</h2>
<p>Acquire continuous historical option quotes for multiple years, with synchronized hedge quotes, rate/borrow histories, contract metadata and sufficient market regimes. For stocks add dividends, splits, earnings dates, American exercise/assignment and actual margin rules. Then run contract-by-contract trades to their predeclared exits, with expiry-matched forecasts, cost hurdles chosen on a training period, portfolio-level overlapping exposure limits, and a final untouched test period. Benchmark against always-long, always-short and no-trade policies. Report net returns, drawdowns, capital requirements and tail losses; do not infer an edge from this one-day pilot.</p>
<h2>7 / Initial data budget: $100</h2>
<p>The proposed first paid stage is one month of ThetaData Options Value ($40/month) plus Stocks Value ($30/month): $70 at the listed monthly prices, before unverified checkout taxes or adjustments. Both are recurring subscriptions. The subscriptions documentation lists historical quote access for both Value tiers. Stocks Value's pricing page describes 15-minute intervals while the subscription documentation describes one-minute historical granularity; use a 15-minute common grid until actual entitlements are tested. This funds the initial stock-quote acquisition route; a paid continuous crypto-history package is not confirmed within the remaining budget. See <a href="../data_plan.json">the saved vendor comparison</a>.</p>
<p>A bounded, one-day acquisition client is prepared in <code>volatility_lab/theta.py</code>. It requires a user-controlled account and a locally running, signed-in Theta Terminal v3; it does not create accounts, subscribe, handle credentials, or simulate American stock options. Raw quote conditions and provider size units are retained for audit. No purchase has been made.</p>
<h2>Sources and reproducibility</h2>
<p><a href="https://arch.readthedocs.io/en/stable/univariate/forecasting.html">arch forecasting</a> · <a href="https://github.com/ranaroussi/yfinance">yfinance</a> · <a href="https://docs.tardis.dev/downloadable-csv-files/overview">Tardis samples</a> · <a href="https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles">Coinbase candle conventions</a> · <a href="https://docs.deribit.com/">Deribit API</a> · <a href="https://support.deribit.com/hc/en-us/articles/25944746248989-Fees">Deribit fee reference</a> · <a href="https://databento.com/docs/datasets/opra-pillar">Databento OPRA</a> · <a href="https://www.thetadata.net/pricing">ThetaData plans</a></p>
<p>Raw source URLs, SHA-256 hashes and timestamps: <a href="provenance.json">provenance.json</a>. Full predictions, scores, aligned pilot inputs, fill logs, ledgers and parameters are alongside this report. Daily cache source ends 3 November 2026 exclusive. No vendor account was created and no data was purchased.</p>
<h2>Recorded acquisition or validation failures</h2><pre>{html.escape(json.dumps(errors,indent=2,default=str))}</pre></main></html>'''
    (out/"report.html").write_text(report)


if __name__ == "__main__":
    main()
