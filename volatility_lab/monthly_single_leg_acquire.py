"""Monthly single-contract selection from decision-time quotes only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .theta_sdk import ROOT, connect
from .theta import normalize_quotes
from .stock_market import MarketStore, DailyInputs, RateInputs, sessions, local_time
from .forecast import fit_variance, expiry_forecast
from .american import AmericanPricer

OUT = ROOT / 'artifacts/monthly_single_leg'
DATA = ROOT / 'data/monthly_single_leg'
_CLIENT = None; _DAILY = {}; _RATES = None


def initialize(symbols):
    global _CLIENT, _DAILY, _RATES
    _CLIENT = connect()
    _DAILY = {s: DailyInputs(ROOT / 'data/stock_study', s) for s in symbols}
    _RATES = RateInputs(ROOT / 'data/stock_study/sofr.json')


def origins(plan):
    return [str(sessions(m, m + pd.offsets.MonthEnd(0)).index[0].date())
            for m in pd.date_range(plan['start'], plan['end'], freq='MS')]


def select_candidates(snapshot, spot, origin, daily, rates, plan, fit):
    decision = local_time(origin, '09:45'); records = []
    snapshot = snapshot[(snapshot.sample_time_utc == decision) & snapshot.passes_snapshot_filters].copy()
    snapshot['expiry_date'] = pd.to_datetime(snapshot.expiration).dt.date
    for target, tolerance in zip(plan['target_calendar_dte'], plan['dte_tolerance_days']):
        base = {'asset': daily.symbol if hasattr(daily, 'symbol') else '', 'origin': origin, 'target_dte': target,
                'decision_time': str(decision), 'decision_spot': spot, 'model': plan['model'], 'side': 0}
        eligible = [(abs((d - pd.Timestamp(origin).date()).days - target), d) for d in snapshot.expiry_date.unique()
                    if (d - pd.Timestamp(origin).date()).days > 1 and abs((d - pd.Timestamp(origin).date()).days - target) <= tolerance]
        if not eligible:
            records.append({**base, 'status': 'no_eligible_decision_quotes'}); continue
        _, expiration = min(eligible); schedule = sessions(origin, expiration)
        expiry = schedule.iloc[-1]['close']; exit_time = schedule.iloc[-2]['close'] - pd.Timedelta(minutes=15)
        prediction = expiry_forecast(fit, decision, expiry, 'equity')
        assert fit.trained_through < decision
        base.update(expiry_date=str(expiration), expiry_time=str(expiry), planned_exit=str(exit_time),
                    feature_time=str(fit.trained_through), forecast_vol=prediction['annualized_vol'],
                    integrated_variance=prediction['integrated_variance'], fit_diagnostics=fit.diagnostics)
        split_dates = [str(d) for d in daily.splits if pd.Timestamp(origin).date() < d <= expiration]
        candidates = []
        for kind in ['call', 'put']:
            rows = snapshot[(snapshot.expiry_date == expiration) & (snapshot.right == kind.upper())]
            if rows.empty: continue
            row = rows.assign(distance=(rows.strike - spot).abs()).sort_values(['distance', 'strike']).iloc[0]
            pricer = AmericanPricer(float(row.strike), kind)
            pricer.configure(spot, (expiry-decision).total_seconds()/(365*86400), rates.at(decision), daily.yield_at(decision, spot))
            try:
                iv = pricer.implied(float(row.ask)); fair = pricer.value(prediction['annualized_vol'])
            except (ValueError, RuntimeError):
                candidates.append({'kind': kind, 'strike': float(row.strike), 'status': 'model_inversion_unavailable'}); continue
            candidates.append({'kind': kind, 'strike': float(row.strike), 'decision_bid': float(row.bid),
                               'decision_ask': float(row.ask), 'decision_ask_iv': iv, 'model_fair_price': fair,
                               'vol_gap': prediction['annualized_vol'] - iv, 'status': 'scorable'})
        base['candidates'] = candidates
        if split_dates:
            records.append({**base, 'status': 'unsupported_split_in_holding_window', 'split_dates': split_dates}); continue
        valid = [c for c in candidates if c['status'] == 'scorable']
        if not valid:
            records.append({**base, 'status': 'no_model_scorable_single_option'}); continue
        selected = max(valid, key=lambda c: (c['vol_gap'], c['kind'] == 'call'))
        records.append({**base, **{k: v for k, v in selected.items() if k != 'status'},
                        'status': 'selected' if selected['vol_gap'] > plan['signal_vol_buffer'] else 'no_signal',
                        'side': int(selected['vol_gap'] > plan['signal_vol_buffer'])})
    return records


def scan_one(symbol, origin, plan):
    checkpoint = OUT / 'decisions' / f'{symbol}_{origin}.json'
    if checkpoint.exists(): return str(checkpoint)
    store = MarketStore(DATA / 'decision_quotes', _CLIENT); source = ROOT / 'data/stock_study/quotes'
    stock_cache = source / f'{symbol}_stock_{pd.Timestamp(origin):%Y%m}.csv'
    if stock_cache.exists(): stock_raw = pd.read_csv(stock_cache); stock_file = stock_cache
    else:
        name = f'{symbol}_{origin}_stock_decision'
        stock_raw = store.fetch(name, 'stock_history_quote', symbol=symbol, date=pd.Timestamp(origin).date(),
                                interval='15m', start_time='09:45:00', end_time='09:46:00', venue='utp_cta')
        stock_file = store.root / (name + '.csv')
    stock = normalize_quotes(stock_raw)
    stock = stock[stock.sample_time_utc == local_time(origin, '09:45')]
    if len(stock) != 1 or not bool(stock.iloc[0].passes_snapshot_filters):
        records = [{'asset': symbol, 'origin': origin, 'target_dte': t, 'status': 'invalid_decision_stock', 'side': 0} for t in plan['target_calendar_dte']]
        snapshot_file = None
    else:
        name = f'{symbol}_{origin}_full_origin_snapshot'; old = source / (name + '.csv')
        if old.exists(): raw = pd.read_csv(old); snapshot_file = old
        else:
            raw = store.fetch(name, 'option_history_quote', symbol=symbol, date=pd.Timestamp(origin).date(), expiration='*',
                              interval='15m', start_time='09:45:00', end_time='09:46:00', max_dte=115)
            snapshot_file = store.root / (name + '.csv')
        fit = fit_variance(_DAILY[symbol].prices, local_time(origin, '09:45'), horizon=plan['forecast_horizon_sessions'],
                           model=plan['model'], window=plan['model_window_returns'])
        spot = float((stock.iloc[0].bid + stock.iloc[0].ask) / 2)
        records = select_candidates(normalize_quotes(raw), spot, origin, _DAILY[symbol], _RATES, plan, fit)
        for row in records: row['asset'] = symbol
    inputs = [p for p in [stock_file, snapshot_file] if p is not None]
    payload = {'asset': symbol, 'origin': origin, 'records': records,
               'input_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    checkpoint.write_text(json.dumps(payload, indent=2, default=str))
    return str(checkpoint)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=2); args = parser.parse_args()
    plan = json.loads((ROOT / 'monthly_single_leg_plan.json').read_text())
    (OUT / 'decisions').mkdir(parents=True, exist_ok=True)
    config = {'plan': plan, 'selection_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_hashes': {n: hashlib.sha256((ROOT/'volatility_lab'/n).read_bytes()).hexdigest() for n in ['forecast.py','american.py','stock_market.py','theta.py']},
              'daily_hashes': {s: hashlib.sha256((ROOT/'data/stock_study'/f'{s}_daily_actions.csv').read_bytes()).hexdigest() for s in plan['universe']},
              'sofr_sha256': hashlib.sha256((ROOT/'data/stock_study/sofr.json').read_bytes()).hexdigest()}
    cfg = OUT / 'selection_configuration.json'
    if cfg.exists() and json.loads(cfg.read_text()) != config: raise ValueError('Selection source or inputs changed; preserve the experiment')
    cfg.write_text(json.dumps(config, indent=2)); paths = []; errors = []
    jobs = [(s, o) for o in origins(plan) for s in plan['universe']]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(plan['universe'],)) as pool:
        futures = {pool.submit(scan_one, s, o, plan): (s, o) for s, o in jobs}
        for future in as_completed(futures):
            symbol, origin = futures[future]
            try:
                paths.append(future.result())
                if len(paths) % 20 == 0: print(f'Scanned {len(paths)}/{len(jobs)} stock-months', flush=True)
            except Exception as exc:
                errors.append({'asset': symbol, 'origin': origin, 'error_type': type(exc).__name__})
                print(json.dumps(errors[-1]), flush=True)
            (OUT / 'selection_errors.json').write_text(json.dumps(errors, indent=2))
    payloads = [json.loads(Path(p).read_text()) for p in paths]
    rows = [r for p in payloads for r in p['records']]
    (OUT / 'decisions.json').write_text(json.dumps(rows, indent=2, default=str))
    frame = pd.DataFrame([{k: v for k, v in r.items() if k not in ['candidates','fit_diagnostics']} for r in rows])
    frame.to_csv(OUT / 'decisions.csv', index=False)
    print(frame.status.value_counts().to_string(), flush=True)
    if errors: raise RuntimeError('Monthly decision jobs failed; all failures retained for repair')


if __name__ == '__main__': main()
