"""Audit the complete original long book, causal hedges, and cost attribution."""
import hashlib
from html.parser import HTMLParser
import json
import shutil
import numpy as np
import pandas as pd
from .theta_sdk import ROOT
from .long_continuous_study import OUT, RAW


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.targets = []
    def handle_starttag(self, tag, attrs):
        self.targets.extend(v for k, v in attrs if k in ('href', 'src'))


def main():
    cfg = json.loads((OUT / 'run_configuration.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert not json.loads((OUT / 'errors.json').read_text())
    for name, value in cfg['source_hashes'].items(): assert digest(ROOT / 'volatility_lab' / name) == value, name
    for name, value in cfg['raw_hashes'].items():
        assert digest(RAW / (name + '.csv')) == value
        metadata = json.loads((RAW / (name + '.json')).read_text())
        assert metadata['sha256'] == value and metadata['parameters']['interval'] == '1m'
    for name, value in cfg['input_hashes'].items(): assert digest(ROOT / name) == value, name
    for name, value in cfg['daily_hashes'].items(): assert digest(ROOT / 'data/stock_study' / (name + '_daily_actions.csv')) == value
    assert digest(ROOT / 'data/stock_study/sofr.json') == cfg['sofr_hash']
    f = pd.read_csv(OUT / 'all_results.csv', low_memory=False)
    selected = pd.read_csv(OUT / 'selected_trades.csv').set_index('case_id')
    assert len(selected) == cfg['plan']['expected_trades'] == 22
    keys = ['case_id', 'data_source', 'hedge_minutes', 'signal_lag_minutes', 'cost_scenario']
    expected = len(selected) * (2 + len(cfg['plan']['frequencies_minutes']) * len(cfg['plan']['signal_lags_minutes'])) * 2
    assert len(f) == expected and not f.duplicated(keys).any()
    assert (f.status == 'completed').all() and (f.side == 1).all()
    assert f.exit_at_original_time.all() and f.same_orders_across_costs.all()
    assert f.reconciliation_error.abs().max() < 1e-5 and f.accounting_residual.abs().max() < 1e-5
    assert (f.net_pnl - (f.market_pnl - f.execution_cost - f.abandonment_writeoff + f.net_financing)).abs().max() < 1e-5
    assert f[f.cost_scenario == 'no_execution_cost'].execution_cost.abs().max() < 1e-7
    quality = json.loads((OUT / 'quote_quality.json').read_text())
    reproduction = []; hedge_orders = 0; cash_pairs = 0; audited_fills = 0
    costplan = json.loads((ROOT / 'short_vol_plan.json').read_text())
    for name in selected.index:
        q = quality[name]
        assert q['minute_grid_confirmed'] and q['common_coarse_rows'] == q['archived_rows']
        assert all(v < 1e-8 for v in q['quote_differences_vs_archived_15m'].values()), name
        subset = f[f.case_id == name]
        assert (pd.to_datetime(subset.entry, utc=True) == pd.Timestamp(selected.loc[name, 'entry'])).all()
        assert (pd.to_datetime(subset.exit, utc=True) == pd.Timestamp(selected.loc[name, 'exit'])).all()
        for scenario in cfg['plan']['cost_scenarios']:
            archived = subset[(subset.data_source == 'archived_15m') & (subset.cost_scenario == scenario)].iloc[0]
            coarse = subset[(subset.data_source == 'resampled_15m') & (subset.cost_scenario == scenario)].iloc[0]
            reference = selected.loc[name, 'net_pnl' if scenario == 'observed_costs' else 'ideal_net_pnl']
            assert abs(archived.net_pnl - reference) < 1e-5, (name, scenario)
            assert abs(coarse.net_pnl - archived.net_pnl) < 1e-5, (name, scenario)
            reproduction.append({'case_id': name, 'cost_scenario': scenario, 'archived_minus_original': archived.net_pnl-reference,
                                 'new_coarse_minus_archived': coarse.net_pnl-archived.net_pnl})
        panels = {}
        for source in ['one_minute', 'resampled_15m']:
            panel = pd.read_csv(OUT / 'cases' / name / (source + '_priced.csv'), index_col=0, parse_dates=True)
            panel.index = pd.to_datetime(panel.index, utc=True); panels[source] = panel
        archived = pd.read_csv(ROOT / 'artifacts/stock_study' / (name + '_priced.csv'), index_col=0, parse_dates=True)
        archived.index = pd.to_datetime(archived.index, utc=True); panels['archived_15m'] = archived
        orders = {}
        for r in subset.itertuples():
            tag = f'{r.data_source}_{r.hedge_minutes}m_lag{r.signal_lag_minutes}m_{r.cost_scenario}'
            fills = pd.read_csv(OUT / 'cases' / name / (tag + '_fills.csv'))
            fills['time'] = pd.to_datetime(fills.time, utc=True)
            fills['decision_time'] = pd.to_datetime(fills.decision_time, utc=True)
            option = fills[fills.instrument.isin(['call', 'put'])]
            assert len(option) == 4
            assert option.groupby('instrument').quantity.sum().eq(0).all()
            assert (option[option.time == pd.Timestamp(r.entry)].quantity == 1).all()
            stock = fills[(fills.instrument == 'stock') & (fills.time != pd.Timestamp(r.exit))]
            elapsed = (stock.time - stock.decision_time).dt.total_seconds()
            assert (elapsed == r.signal_lag_minutes * 60).all()
            panel = panels[r.data_source]
            assert panel.loc[stock.time, 'stock_passes_snapshot_filters'].all()
            assert panel.loc[stock.decision_time, 'stock_passes_snapshot_filters'].all()
            assert np.isfinite(panel.loc[stock.decision_time, 'delta_per_straddle']).all()
            hedge_orders += len(stock)
            # Independently reconstruct every filled price and fee from the
            # matching observed quote, rather than just trusting cash labels.
            costs = costplan['costs_observed' if r.cost_scenario == 'observed_costs' else 'costs_midpoint']
            for instrument, rows in fills.groupby('instrument'):
                observed = panel.loc[rows.time]
                change = rows.quantity.to_numpy(); buy = change > 0
                mid = observed.spot.to_numpy() if instrument == 'stock' else observed[instrument + '_mid'].to_numpy()
                prefix = 'stock' if instrument == 'stock' else instrument
                quote = np.where(buy, observed[prefix + '_ask'], observed[prefix + '_bid'])
                fraction = costs['stock_spread_fraction' if instrument == 'stock' else 'option_spread_fraction']
                expected_price = mid + (quote - mid) * fraction
                if instrument == 'stock':
                    expected_price *= 1 + np.sign(change) * costs['extra_slippage_bps'] / 10000
                    expected_fee = np.maximum(costs['minimum_stock_commission'], np.abs(change) * costs['stock_commission_per_share']) * costs['fee_multiplier']
                    expected_fee += np.maximum(-change, 0) * expected_price * costs['stock_sell_fee_bps'] / 10000 * costs['fee_multiplier']
                else:
                    abandon = (~buy) & (observed[instrument + '_bid'].to_numpy() == 0)
                    expected_price = np.where(abandon, 0., expected_price)
                    expected_fee = np.where(abandon, 0., np.abs(change) * costs['option_commission_per_contract'] * costs['fee_multiplier'])
                    assert rows.executed.fillna(False).to_numpy().astype(bool).tolist() == (~abandon).tolist()
                assert np.max(np.abs(rows.fill_price.to_numpy() - expected_price), initial=0.) < 1e-7
                assert np.max(np.abs(rows.fee.to_numpy() - expected_fee), initial=0.) < 1e-7
                audited_fills += len(rows)
            orders[(r.data_source, r.hedge_minutes, r.signal_lag_minutes, r.cost_scenario)] = fills[['time', 'instrument', 'quantity']].to_dict('records')
        for (source, frequency, lag), pair in subset.groupby(['data_source', 'hedge_minutes', 'signal_lag_minutes']):
            assert orders[(source, frequency, lag, 'observed_costs')] == orders[(source, frequency, lag, 'no_execution_cost')]
            assert np.ptp(pair.market_pnl) < 1e-6 and np.ptp(pair.abandonment_writeoff) < 1e-6
            cash_pairs += 1
    pd.DataFrame(reproduction).to_csv(OUT / 'archived_cash_reproduction.csv', index=False)
    tests = (ROOT / 'artifacts/long_continuous_tests.log').read_text()
    assert 'Ran 42 tests' in tests and '\nOK\n' in tests
    snapshot = OUT / 'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for name in cfg['source_hashes']:
        shutil.copy2(ROOT / 'volatility_lab' / name, snapshot / name)
    for name in ['long_continuous_report.py', 'long_continuous_validate.py']:
        shutil.copy2(ROOT / 'volatility_lab' / name, snapshot / name)
    shutil.copy2(ROOT / 'long_continuous_plan.json', snapshot / 'long_continuous_plan.json')
    result = {'status': 'passed', 'unit_tests': 42, 'selected_original_long_trades': len(selected), 'completed_scenarios': len(f),
              'original_cash_outcomes_reproduced': len(reproduction), 'resampled_coarse_cash_outcomes_reproduced': len(reproduction),
              'all_original_coarse_quote_prices_reproduced': True, 'source_and_input_hashes_match': True,
              'cost_pairs_with_identical_orders_and_market_pnl': cash_pairs, 'all_entries_and_exits_unchanged': True,
              'causal_non_liquidation_stock_orders_checked': hedge_orders, 'all_long_straddles': True,
              'independently_audited_fill_prices_and_fees': audited_fills,
              'actual_one_minute_stock_and_option_snapshots': True, 'interpolated_subinterval_prices': False,
              'raw_quote_files': len(cfg['raw_hashes']), 'minute_rows_across_cases': sum(q['fine_rows'] for q in quality.values()),
              'max_cash_reconciliation_error': float(f.reconciliation_error.abs().max()),
              'all_report_local_links_exist': True}
    # Write provisionally so the report's verification link can itself be audited.
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2))
    parser = Links(); parser.feed((OUT / 'report.html').read_text())
    assert all((OUT / t).exists() for t in parser.targets if not t.startswith(('http://', 'https://', '#')))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
