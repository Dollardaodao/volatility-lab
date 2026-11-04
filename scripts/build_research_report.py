"""Build readable Markdown reports from the published aggregate results.

This needs only the Python standard library: no credentials or raw data.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/oos_2026'
NAMES = {
    'garch_t': 'GARCH-t', 'ewma': 'EWMA', 'rolling21': 'Rolling21',
    'har_proxy_ridge': 'HAR proxy / Ridge',
    'ml_hist_gradient_boosting': 'Histogram gradient boosting',
}


def read(name, folder=DATA):
    with (folder / name).open(newline='') as source:
        return list(csv.DictReader(source))


def money(value):
    n = float(value)
    return ('−' if n < 0 else '+') + f'{abs(n):,.0f}'


def number(value, digits=2):
    return f'{float(value):,.{digits}f}'


def integer(value):
    return str(int(float(value)))


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
    ])


def main():
    summary = read('summary.csv')
    daily = [r for r in summary if r['protocol_id'] == 'daily_close']
    hourly = [r for r in summary if r['protocol_id'] == 'elapsed_60m']
    scores = sorted(read('forecast_scores.csv'), key=lambda r: float(r['qlike']))
    by_horizon = read('forecast_scores_by_horizon.csv')
    costs = [r for r in read('cost_breakdown.csv') if r['protocol_id'] == 'daily_close']
    development = read('longer_hedge_comparison.csv', ROOT / 'reports/development_2022_2025')

    def trade_rows(records):
        return [[NAMES[r['model']], integer(r['completed']), money(r['before_execution_pnl']),
                 money(r['after_execution_pnl']),
                 integer(r['before_wins']) + ' / ' + integer(r['after_wins']),
                 integer(r['vol_bets_correct_15m']) + ' / ' + integer(r['vol_bets_scorable'])]
                for r in records]

    trade_table = table(['Model', 'Trades', 'Before execution costs ($)', 'After execution costs ($)',
                         'Profitable before / after', 'RV > entry ask-IV'], trade_rows(daily))
    hourly_table = table(['Model', 'Trades', 'Before execution costs ($)', 'After execution costs ($)',
                          'Profitable before / after', 'RV > entry ask-IV'], trade_rows(hourly))
    forecast_table = table(['Model', 'Forecasts', 'QLIKE ↓', 'MAE (vol points) ↓', 'RMSE (vol points) ↓'],
        [[NAMES[r['model']], integer(r['observations']), number(r['qlike'],6),
          number(r['mae_vol_points'],3), number(r['rmse_vol_points'],3)] for r in scores])
    horizon_table = table(['Forecast horizon (sessions)', 'Model', 'QLIKE ↓', 'MAE (vol points) ↓'],
        [[integer(r['horizon_sessions']), NAMES[r['model']], number(r['qlike'],4), number(r['mae_vol_points'],2)]
         for r in sorted(by_horizon, key=lambda r:(int(r['horizon_sessions']),float(r['qlike'])))])
    cost_table = table(['Model', 'Option executed spread ($)', 'Stock spread ($)', 'Slippage ($)',
                        'Option fees ($)', 'Stock fees ($)', 'Total execution cost ($)'],
        [[NAMES[r['model']], *[number(r[k],2) for k in ['option_executed_spread_cost',
           'stock_spread_cost','slippage_cost','option_fees','stock_fees','total_execution_cost']]] for r in costs])
    risk_table = table(['Model', 'Mean trade return before (%)', 'Mean trade return after (%)',
                        'All-opportunity mean return after (%)', 'Mean absolute residual Delta (shares)'],
        [[NAMES[r['model']], number(100*float(r['mean_return_on_capital_before']),4),
          number(100*float(r['mean_return_on_capital_after']),4),
          number(100*float(r['all_opportunity_mean_roc_after']),4),
          number(r['mean_abs_delta_shares'],2)] for r in daily])
    development_table = table(['Hedge schedule', 'Trades', 'Before execution costs ($)',
                              'After execution costs ($)', 'Mean absolute residual Delta (shares)'],
        [[r['protocol_id'],integer(r['trades']),money(r['no_execution_cost']),money(r['observed_costs']),
          number(r['mean_abs_delta_shares'],2)] for r in development])

    report = f'''# Volatility Forecasting and Delta-Hedged Single Options

## Research finding

Five volatility models were evaluated on a common monthly opportunity grid for 20 US stocks. Forecast variance was matched to each option's remaining life. The policy bought a single call or put when forecast volatility exceeded that contract's ask-implied volatility by more than three absolute volatility percentage points, then delta-hedged with stock.

In the January–July 2026 entry cohorts, **all five models lost money after execution costs under both daily-close and hourly hedging**. GARCH and histogram gradient boosting had nearly identical aggregate QLIKE, but gradient boosting selected a worse-performing trade subset. There is no demonstrated durable trading edge or universal model winner from this sample.

![Forecast accuracy and daily-hedge trading results](images/oos_model_comparison.png)

## 1. Sample and research chronology

| Item | Definition |
| --- | --- |
| Development period | 2022–2025; outcomes already inspected during prior research |
| New historical time holdout | First NYSE trading session of each month, January–July 2026 |
| Entry dates | January 2 through July 1, 2026 |
| Actual exits | January 8 through September 17, 2026 |
| Target tenors | 7, 30 and 90 calendar days; actual eligible expiries used |
| Opportunity count | 20 stocks × 7 monthly origins × 3 tenors = 420 per model |
| Primary hedge schedule | Daily, at exchange close minus 15 minutes |
| Fixed comparison schedule | Every 60 elapsed minutes, with orders restricted to regular sessions |
| Entry signal | Forecast vol − decision ask-IV > 0.03 |
| Selection size | One standard option contract, multiplier 100, per selected stock/month/tenor |

The universe is AAPL, MSFT, NVDA, AMZN, JPM, KO, AMD, INTC, ADBE, CRM, QCOM, XOM, CVX, BAC, WFC, PEP, PG, UNH, MRK and DIS. It is a research convenience universe selected in earlier work; it is not a point-in-time reconstruction of an investable stock index.

The five models generated 442 selected entries altogether. Because selections overlap, these correspond to **249 distinct contract opportunities**. Two hedge schedules and two cost books give **996 unique replay scenarios**. There were no rejected entries in this run.

The January–July cutoff was specified to allow all admissible long-tenor contracts to finish within the available history; origins were not included or excluded according to realized profits.

### What “out of sample” means here

The strategy rules, parameters, threshold and two hedge schedules were fixed before this study's 2026 option-chain acquisition and P&L calculation. The 2022–2025 exploratory results were already known. Daily stock history through 2026 had already been cached, so this is a **historical chronological holdout**, not a claim that all 2026 prices had never been viewed, and not a prospective live trial.

Models refit at each monthly origin. Features use the prior actual session close. Supervised rows enter training only when the complete forward-return label block has finished by that cutoff. A 90-day-type label that is still unfolding cannot enter training merely because its anchor date is in the past. Earlier, completed test-period labels may enter later fits: this is walk-forward OOS rather than a model trained once and frozen for the entire year.

Publishing these results makes subsequent use of the same dates a reproduction or further exploration, not another untouched test.

## 2. Forecast models

| Model | Specification |
| --- | --- |
| GARCH-t | Zero-mean Student-t GARCH(1,1), fitted on at most 756 past daily returns |
| EWMA | Squared-return variance with decay 0.94 |
| Rolling21 | Mean of the latest 21 squared returns |
| HAR proxy / Ridge | Pooled regularized regression with daily variance proxies and fixed alpha 10 |
| Histogram gradient boosting | Pooled positive-variance model with Poisson loss; 150 iterations, learning rate 0.05, maximum depth 4, fixed random seed |

The two supervised models pool the 20 stocks, using a rolling 756-anchor training window. Future variance is modeled in blocks covering sessions 1, 2–5, 6–21, 22–63 and 64–84. Feature inputs include past returns, variance at multiple horizons, downside variance, high/low ranges, overnight gaps and relative volume. No option IV is supplied as a model feature. Hyperparameters are fixed rather than selected on holdout P&L.

Daily variance curves are integrated to the **actual option expiry**, using actual exchange sessions and ACT/365 annualization. Partial close-to-close intervals allocate variance in proportion to elapsed wall-clock time, including weekends. This is an approximation; intraday volatility seasonality is not estimated.

The HAR model is explicitly a **daily-proxy version**, not the original high-frequency HAR-RV specification. Labels are future squared daily close-to-close log returns; they are not a continuous realized-variance measure.

## 3. Single-option trading policy

1. On the first exchange session of each month, observe the stock and option chain at 09:45 New York time.
2. For each target tenor, choose the nearest eligible expiry. Tenor tolerances are 5, 12 and 25 calendar days for 7-, 30- and 90-day targets.
3. Find the nearest valid strike separately for the call and put; invert each ask price using the American-option approximation.
4. Choose the individual contract with the largest forecast-vol-minus-ask-IV gap, with a deterministic call tie-break. Buy only if that gap exceeds 0.03. Otherwise remain flat.
5. Enter at 10:00 New York time using valid observed quotes. Do not switch to another leg after a failed entry.
6. Hedge whole shares using one-minute-old **market-IV Delta**. Initialize the hedge independently of the later periodic schedule.
7. Plan to exit 15 minutes before the official close on the session before expiry, subject to quote eligibility and the fixed research capital-reserve rule. Unwind remaining stock at exit.

Within a given stock/month/tenor, a model's forecast is common to the call and put comparison. Maximizing forecast minus ask-IV therefore selects the lower-ask-IV valid candidate. Models primarily differ in whether that winning contract crosses their entry threshold. If several models choose the same opportunity, the exact same market-IV hedge book is reused; forecast values do not otherwise alter its hedge orders or exit rules.

This study does not enter paired straddles. Every model scans all 420 opportunities rather than inheriting an earlier fixed trade list.

## 4. Forecast accuracy

Forecast accuracy is scored independently of whether an option trade is entered. Each model has 420 scored forecasts: 20 stocks × 7 origins × 5/21/63-session horizons. These scoring horizons are separate from the option-trading tenors, whose forecasts are integrated to their actual expiries.

For realized integrated variance `v` and forecast integrated variance `v_hat`, the normalized QLIKE used here is:

```text
r = v / v_hat
QLIKE = r - log(r) - 1
```

Lower QLIKE is better. MAE and RMSE below are annualized volatility percentage-point errors. The realized target is the daily squared-return proxy described above.

{forecast_table}

Gradient boosting and GARCH are effectively tied on aggregate QLIKE; GARCH has slightly smaller aggregate MAE and RMSE. Rankings vary by horizon: gradient boosting has the lowest 5-session QLIKE, while the HAR proxy has the lowest 21- and 63-session QLIKE. Different forecast-loss rankings should not be read as trading-profit rankings.

{horizon_table}

## 5. Trading results

### Daily-close hedging: primary comparison

Dollar amounts are sums of one-contract, independently funded trades. Each model starts with the same 420 opportunities; counts differ because their forecasts trigger different entry decisions.

{trade_table}

“RV > entry ask-IV” compares actual entry-to-exit 15-minute-plus-overnight realized volatility with the option's **actual entry-time ask-IV**. It is a volatility-bet diagnostic, not a profit count or a direct measure of forecast error. Gamma weights, IV repricing, discrete hedges, holding-window differences and costs can make it disagree with P&L.

### Hourly hedging: fixed comparison

{hourly_table}

All five models remain negative after execution costs. Gradient boosting's daily-hedge book is already negative before execution costs; its hourly pre-cost result is approximately flat. The four other models have positive pre-execution-cost totals but negative observed-cost totals.

### Capital normalization and residual risk

Each trade is independently funded at three times its entry 100-share underlying notional. The all-opportunity mean includes flat or failed-entry opportunities as zero and averages over all 420 opportunities. These are diagnostic cohort averages, not annualized returns or the performance of a combined capital-constrained portfolio.

{risk_table}

Daily hedging leaves materially more residual Delta than hourly hedging. For example, GARCH's average absolute residual model Delta is about 9.96 shares under daily-close hedging versus 2.43 under hourly hedging. Lower transaction cost therefore comes with different exposure to the realized price path.

### Exit exception

The INTC February 2, 2026 90-day-target call triggered the fixed research capital-reserve close on May 8, ahead of its planned May 14 exit. EWMA, Rolling21 and HAR selected this opportunity; GARCH and gradient boosting did not. Daily and hourly triggers differed by one minute. Each midpoint comparison follows its corresponding actual exit. This reserve is a research constraint, not a simulated broker-margin requirement. The other 248 distinct opportunities exited at their planned times.

## 6. Execution costs and cash accounting

| Component | Fixed assumption |
| --- | --- |
| Option execution | Buy at observed ask, sell at observed bid |
| Option commission | $1 per contract per executed transaction |
| Stock execution | Buy ask / sell bid, plus 1 bp extra slippage |
| Stock commission | $0.0035 per share; $0.35 minimum per order |
| Stock sell surcharge | 0.1 bp of sale proceeds |
| Cash lending | max(SOFR − 0.5%, 0) |
| Cash borrowing | SOFR + 1.5% |
| Short-stock borrow | 2% annualized |
| Signal timing | At least one-minute-old stock/Delta information |
| Trading hours | Exchange regular sessions only; no overnight hedge orders |
| Sizing | Whole shares; observed sizes, with a stock quote-size multiplier of 100 |

Rate observations use a two-calendar-day availability lag. Cash dividends are booked separately. An unbid long right can be abandoned for zero recovery under the specified eligibility rule; the engine does not invent an executable midpoint sale.

The **before-execution-cost** book keeps the observed book's order times, quantities and exits, substitutes midpoint prices for executed orders, and removes commissions, sell surcharges and slippage. It still retains funding, short-stock borrow, the idle-cash benchmark and abandoned-rights writeoffs. Removing fees changes subsequent cash interest slightly, so the before/after P&L difference need not exactly equal the immediate execution-cost sum.

Daily-close execution-cost attribution:

{cost_table}

Option executed-spread costs exclude abandonment writeoffs. Writeoffs and financing are separate fields in the released cost CSV, avoiding double counting.

The cash bridge is:

```text
net P&L above idle-cash benchmark
  = option midpoint mark change + stock hedge mark P&L + stock dividends
    - execution spreads - slippage - commissions and sell surcharges
    - abandoned-rights writeoffs
    + cash interest - benchmark interest - stock borrow
```

## 7. Development results: context, not a second untouched test

The earlier 2022–2025 monthly GARCH-only single-option study scanned 2,880 opportunities and executed 990 trades. A fixed-entry hedge-schedule comparison produced the following totals:

{development_table}

The daily-close result was close to break-even before the later holdout but remained negative after costs. These outcomes had already been inspected. The 2026 primary daily-close and hourly comparison was fixed subsequently, before the new option P&L evaluation. These development results are not independent confirmation of the 2026 findings.

Calendar-hour schedules count nights and weekends but place orders only during regular sessions. A 12-calendar-hour interval therefore often results in an adjustment near the next session open. Cumulative trading-hour schedules exclude nights, weekends, holidays and early-close downtime.

## 8. Validation and limitations

The original complete local run passed 55 unit tests, including future-price perturbation checks that leave past forecasts unchanged, supervised-label completion/purging tests, expiry variance integration, hedge clocks, and accounting tests. Its independent audit checked **102,570 fill records across observed and midpoint books**, with maximum cash-transition residual below $0.000000001 and zero recorded fill-price/fee discrepancies.

The public repository preserves the numerical sources, fixed plans, aggregate tables and original audit manifests. It excludes raw market data, full per-minute books, fitted models and credentials. The public release check reconciles available aggregates and their provenance; it does not claim to repeat the full historical-quote audit without the original local inputs.

Material limits include seven monthly entry cohorts, overlapping long tenors, cross-stock dependence, a convenience stock universe, independently funded rather than netted portfolio accounts, fixed fee/borrow assumptions, unverified underlying quote-tick freshness, an American-pricing approximation with continuous dividend yield, and no actual early-exercise simulation. Missing IV can be carried forward from past observations for at most four days; missing Delta observations remain disclosed. Discrete regular-session hedging leaves overnight and intraday gap risk.

These results do not establish a profitable strategy. They show how expiry-matched forecasting, signal selection, financing and actual execution assumptions can produce different conclusions from forecast accuracy alone.

## Files

- [All model/hedge summaries](../reports/oos_2026/summary.csv)
- [Forecast scores](../reports/oos_2026/forecast_scores.csv)
- [Monthly trading breakdown](../reports/oos_2026/monthly_breakdown.csv)
- [Stock trading breakdown](../reports/oos_2026/stock_breakdown.csv)
- [Tenor trading breakdown](../reports/oos_2026/tenor_breakdown.csv)
- [Execution-cost attribution](../reports/oos_2026/cost_breakdown.csv)
- [Original complete-run audit manifest](../reports/oos_2026/verification.json)
- [Frozen OOS plan](../oos_single_leg_plan.json)
- [Reproduction guide](REPRODUCIBILITY.md)
'''
    (ROOT/'docs/RESEARCH_REPORT.md').write_text(report,encoding='utf-8')
    print('Built English Markdown report from published aggregate CSV files.')


if __name__ == '__main__':
    main()
