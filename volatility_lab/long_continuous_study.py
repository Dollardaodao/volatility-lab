"""Actual-minute hedging of all 22 original long-volatility trades."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .near_continuous_study import acquire, load_panel, quote_jobs
from .stock_market import DailyInputs, RateInputs
from .american import enrich_american
from .stock_replay import StockCosts
from .highfreq_replay import replay
from .short_vol_study import account, trade_diagnostics

OUT = ROOT / 'artifacts' / 'long_continuous_study'
RAW = ROOT / 'data' / 'near_continuous' / 'quotes'


def selected_cases(plan):
    f = pd.read_csv(ROOT / plan['source_results'], low_memory=False)
    f = f[(f.policy == plan['policy']) & (f.cost_scenario == plan['source_cost_scenario']) &
          (f.hedge_minutes == plan['source_hedge_minutes']) & (f.side == 1)].copy()
    assert len(f) == plan['expected_trades'] and (f.status == 'completed').all()
    f['case_id'] = f.asset + '_' + f.origin + '_' + f.target_dte.astype(int).astype(str) + 'd'
    assert not f.case_id.duplicated().any()
    # Existing matched midpoint runs give an independent original-cash reference.
    old = pd.read_csv(ROOT / 'artifacts/short_vol_study/all_results.csv', low_memory=False)
    old = old[(old.model == 'garch_t') & (old.direction == 'long') &
              (old.side == 1) & (old.cost_scenario == 'no_execution_cost')]
    old = old.set_index(['asset', 'origin', 'target_dte'])
    f['ideal_net_pnl'] = [float(old.loc[(r.asset, r.origin, r.target_dte), 'net_pnl']) for r in f.itertuples()]
    return f.sort_values(['origin', 'asset', 'target_dte']).to_dict('records')


def run_case(case, mapping, plan):
    folder = OUT / 'cases' / case['case_id']; folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'results.json'
    if checkpoint.exists():
        return str(checkpoint)
    fine, repairs = load_panel(case, mapping, RAW)
    days = fine.index.tz_convert('America/New_York').date
    gaps = fine.index.to_series().diff().dt.total_seconds().to_numpy()
    same_day = np.r_[False, days[1:] == days[:-1]]
    assert np.any(gaps[same_day] == 60) and not np.any(gaps[same_day] < 60)
    coarse = fine[fine.index.tz_convert('America/New_York').minute % 15 == 0].copy()
    old = pd.read_csv(ROOT / 'artifacts/stock_study' / (case['case_id'] + '_priced.csv'), index_col=0, parse_dates=True)
    old.index = pd.to_datetime(old.index, utc=True); old = old.loc[:pd.Timestamp(case['exit'])]
    common = old.index.intersection(coarse.index)
    cols = ['spot', 'stock_bid', 'stock_ask', 'call_bid', 'call_ask', 'put_bid', 'put_ask']
    differences = {col: float((coarse.loc[common, col] - old.loc[common, col]).abs().max()) for col in cols}
    daily = DailyInputs(ROOT / 'data/stock_study', case['asset'])
    rates = RateInputs(ROOT / 'data/stock_study/sofr.json')
    for label, panel in [('one_minute', fine), ('resampled_15m', coarse)]:
        cache = folder / (label + '_priced.csv')
        if cache.exists():
            priced = pd.read_csv(cache, index_col=0, parse_dates=True); priced.index = pd.to_datetime(priced.index, utc=True)
        else:
            priced = enrich_american(panel, mapping['contract']['strike'], daily, rates); priced.to_csv(cache)
        if label == 'one_minute': fine = priced
        else: coarse = priced
    contract = dict(mapping['contract']); contract['planned_exit'] = case['exit']
    costs_plan = json.loads((ROOT / 'short_vol_plan.json').read_text())
    costs = {'observed_costs': StockCosts(**costs_plan['costs_observed']),
             'no_execution_cost': StockCosts(**costs_plan['costs_midpoint'])}
    protocols = [('archived_15m', old, 15, 60, 15), ('resampled_15m', coarse, 15, 60, 15)]
    protocols += [('one_minute', fine, 1, frequency, lag)
                  for lag in plan['signal_lags_minutes'] for frequency in plan['frequencies_minutes']]
    results = []
    for label, panel, grid, frequency, lag in protocols:
        group = []; orders = {}
        for scenario, cost in costs.items():
            base = {k: case[k] for k in ['case_id', 'asset', 'origin', 'target_dte', 'garch_t_forecast_vol', 'long_execution_iv']}
            base.update(data_source=label, grid_minutes=grid, hedge_minutes=frequency, signal_lag_minutes=lag, cost_scenario=scenario)
            result, ledger, fills = replay(panel, contract, daily, 1, frequency, cost,
                                          grid_minutes=grid, signal_lag_minutes=lag, allow_exercise_close=False)
            diagnostics = trade_diagnostics(panel, contract, result, 1, fills)
            bridge = account(result, diagnostics)
            row = {**base, **result, **diagnostics, **bridge, 'status': 'completed',
                   'exit_at_original_time': pd.Timestamp(result['exit']) == pd.Timestamp(case['exit'])}
            tag = f'{label}_{frequency}m_lag{lag}m_{scenario}'
            ledger.to_csv(folder / (tag + '_ledger.csv'), index=False)
            fills.to_csv(folder / (tag + '_fills.csv'), index=False)
            stock = fills[fills.instrument == 'stock']
            hedges = stock[pd.to_datetime(stock.time, utc=True) != pd.Timestamp(result['exit'])]
            delay = (pd.to_datetime(hedges.time, utc=True) - pd.to_datetime(hedges.decision_time, utc=True)).dt.total_seconds()
            assert (delay == lag * 60).all()
            row['stock_orders'] = len(stock); row['stock_turnover_shares'] = int(stock.quantity.abs().sum())
            active = ledger[ledger.option_contracts_per_leg != 0]
            absdelta = active.delta_shares.abs().dropna()
            row['mean_abs_delta_shares'] = float(absdelta.mean())
            row['p95_abs_delta_shares'] = float(absdelta.quantile(.95))
            row['missing_delta_fraction'] = float(active.delta_shares.isna().mean())
            orders[scenario] = fills[['time', 'instrument', 'quantity']].to_dict('records')
            if label == 'archived_15m':
                reference = case['net_pnl'] if scenario == 'observed_costs' else case['ideal_net_pnl']
                assert abs(result['net_pnl'] - reference) < 1e-5, 'Original long cash reproduction failed'
            group.append(row)
        same = orders['observed_costs'] == orders['no_execution_cost']
        for row in group: row['same_orders_across_costs'] = same
        results.extend(group)
    quality = {'fine_rows': len(fine), 'coarse_rows': len(coarse), 'archived_rows': len(old),
               'common_coarse_rows': len(common), 'minute_grid_confirmed': True,
               'nonpositive_stock_mark_repairs': repairs, 'quote_differences_vs_archived_15m': differences,
               'fine_invalid_stock_rows': int((~fine.stock_passes_snapshot_filters).sum()),
               'fine_invalid_option_pairs': int((~fine.options_valid).sum()),
               'iv_carried_fraction': float(fine.iv_carried.mean()), 'iv_cross_leg_fraction': float(fine.iv_cross_leg.mean()),
               'original_quote_tick_time_verified': False}
    checkpoint.write_text(json.dumps({'case': case, 'quality': quality, 'results': results}, indent=2, default=str))
    return str(checkpoint)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--acquire-only', action='store_true'); args = parser.parse_args()
    plan = json.loads((ROOT / 'long_continuous_plan.json').read_text()); cases = selected_cases(plan)
    OUT.mkdir(exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cases).to_csv(OUT / 'selected_trades.csv', index=False)
    jobs, mapping = quote_jobs(cases)
    # Shared vendor cache is usable only for the identical original contracts.
    manifest = json.loads((ROOT / plan['source_manifest']).read_text())['cohorts']
    for case in cases:
        original = next(c for c in manifest if all(c[k] == case[k] for k in ['asset', 'origin', 'target_dte']))
        for key in ['strike', 'expiry_time', 'decision_time', 'planned_exit']:
            assert mapping[case['case_id']]['contract'][key] == original[key], key
    print(f'Selected {len(cases)} original long trades; {len(jobs)} minute quote requests, '
          f'{sum(not (RAW/(name+".csv")).exists() for name in jobs)} missing', flush=True)
    acquire(jobs, RAW)
    # Preserve an acquisition manifest for this experiment independently of shared-cache logs.
    (OUT / 'acquisition_manifest.json').write_text(json.dumps({k: {'method': m, 'parameters': p} for k, (m, p) in jobs.items()}, indent=2, default=str))
    if args.acquire_only: return
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    sources = ['long_continuous_study.py', 'near_continuous_study.py', 'highfreq_replay.py', 'stock_replay.py',
               'american.py', 'stock_market.py', 'theta.py', 'short_vol_study.py']
    cfg = {'plan': plan, 'source_hashes': {n: digest(ROOT / 'volatility_lab' / n) for n in sources},
           'raw_hashes': {n: digest(RAW / (n + '.csv')) for n in jobs},
           'input_hashes': {n: digest(ROOT / n) for n in [plan['source_results'], plan['source_manifest'],
               'artifacts/short_vol_study/all_results.csv', 'short_vol_plan.json']},
           'daily_hashes': {s: digest(ROOT / 'data/stock_study' / (s + '_daily_actions.csv')) for s in {c['asset'] for c in cases}},
           'sofr_hash': digest(ROOT / 'data/stock_study/sofr.json'),
           'dependencies': {n: importlib.metadata.version(n) for n in ['pandas', 'numpy', 'QuantLib']}}
    path = OUT / 'run_configuration.json'
    if path.exists() and json.loads(path.read_text()) != cfg: raise ValueError('Preserve prior run before changing inputs or code')
    path.write_text(json.dumps(cfg, indent=2)); paths = []; errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, c, mapping[c['case_id']], plan): c['case_id'] for c in cases}
        for future in as_completed(futures):
            name = futures[future]
            try: paths.append(future.result()); print(f'Finished {len(paths)}/{len(cases)} {name}', flush=True)
            except Exception as exc:
                errors.append({'case_id': name, 'error_type': type(exc).__name__, 'reason': str(exc)})
                print(json.dumps(errors[-1]), flush=True)
            (OUT / 'errors.json').write_text(json.dumps(errors, indent=2))
    payloads = [json.loads(Path(p).read_text()) for p in paths]
    pd.DataFrame([{k: v for k, v in r.items() if k != 'cost_parameters'} for p in payloads for r in p['results']]).to_csv(OUT / 'all_results.csv', index=False)
    (OUT / 'quote_quality.json').write_text(json.dumps({p['case']['case_id']: p['quality'] for p in payloads}, indent=2))
    if errors: raise RuntimeError('Long minute replays failed; no cases were silently dropped')
    from .long_continuous_report import make_report
    make_report(OUT)
    print(f'Report: {OUT/"report.html"}', flush=True)


if __name__ == '__main__': main()
