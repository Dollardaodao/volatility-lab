# Trading the Gap Between Realized and Implied Volatility

**Volatility Lab research report · November 8, 2026**

## Motivation and strategy

An option's ask price tells us what we must pay for convex exposure to the stock. We express that price as ask implied volatility (IV). If the stock moves more than the option price implies, a long option with a stock hedge can earn money from those moves. The research question is whether we can identify those entry opportunities **before** the moves occur and retain the gains after trading costs.

We study a monthly scan across 20 US stocks and 7-, 30-, and 90-calendar-day target maturities. The entry decision uses a 09:45 New York snapshot and the actual remaining life of the eligible option. The strategy compares forecast volatility with each near-the-money call and put's ask IV, buys at most one standard contract when the largest gap clears the entry buffer, and hedges the resulting Delta with the stock. The base buffer is 3 volatility percentage points.

## Why Delta hedging can monetize the volatility gap

For intuition, take a European option marked at fixed volatility $\sigma_I$, its entry IV. Let $S$ be the stock price, $V(t,S)$ the option value, $r$ the risk-free rate, and $q$ a continuous dividend yield. With $\Delta=V_S$ and $\Gamma=V_{SS}$, the Black–Scholes–Merton PDE is

$$
V_t+(r-q)S\Delta+\tfrac12\sigma_I^2 S^2\Gamma-rV=0.
$$

Hold one unit of the option, $-\Delta$ shares, and cash $B$, so the portfolio value is $X=V-\Delta S+B$. Rebalancing the hedge is financed from cash. For a continuous stock diffusion with instantaneous volatility $\sigma_t$, the self-financing cash flows and Itô's formula give

$$
dX=dV-\Delta(dS+qS\,dt)+rB\,dt,
\qquad
dX-rX\,dt=\tfrac12\Gamma S^2(\sigma_t^2-\sigma_I^2)\,dt.
$$

The stock hedge offsets the first-order move. With positive Gamma, a price rise increases Delta and calls for selling shares, while a fall calls for buying shares. The PDE balances this convexity against the option's time decay and carry. Discounted excess P&L over a holding horizon $H$ is

$$
e^{-rH}X_H-X_0=\tfrac12\int_0^H e^{-rt}\Gamma_tS_t^2(\sigma_t^2-\sigma_I^2)\,dt.
$$

The long hedge earns excess P&L when realized variance exceeds implied variance on this weighted basis. [Black and Scholes](https://doi.org/10.1086/260062) · [Merton](https://dspace.mit.edu/entities/publication/6e05a2fa-478f-41e6-9e69-a06532002236)

An ordinary holding-period RV averages variation across the window. The hedge earns more from variation when Gamma is high. The replay also rebalances discretely, trades American options at observed prices, and exits before expiry, when changes in market IV can affect the exit price. These differences motivate testing the relationship with actual execution rather than inferring each trade's P&L from the sign of the volatility gap.

## Hindsight test with realized volatility known at entry

We first ask whether the strategy can monetize a positive volatility gap when the selection rule has access to future stock returns. This hindsight experiment uses the original 420 January–July 2026 stock, month, and tenor opportunities. It preserves the two near-the-money candidates and planned exit from the 09:45 snapshot. At 10:00 it recomputes the candidates' ask IVs, selects the cheaper-IV valid leg, and enters when the **subsequent holding-period RV minus that 10:00 ask IV** clears the gate. Holding RV uses 15-minute stock midpoint returns, overnight moves, and the exact entry and planned exit times.

Each entry buys one standard option contract with a 100-share multiplier. The replay initializes the stock hedge at entry, then adjusts it daily at the exchange close minus 15 minutes, using whole shares and market-IV Delta with a one-minute lag. The original planned exit is 15 minutes before the close of the session before expiry. Observed execution uses bid and ask prices, commissions, and the original stock execution and financing rules. The before-execution-cost comparison uses the same orders at midpoints and retains financing, borrow, and the cash benchmark.

| Hindsight gate | Completed trades | Before execution costs | After execution costs | Profitable after costs |
| --- | ---: | ---: | ---: | ---: |
| Actual holding RV > entry ask IV | 195 | +\$40,807 | +\$22,806 | 124 / 195 |
| Actual holding RV > entry ask IV by 3 points | 133 | +\$40,404 | +\$27,346 | 103 / 133 |

The dollar totals sum independently funded one-contract trades. All selected trades exceed entry IV by construction. At the 3-point gate, 12 trades still lose before execution costs and 30 lose after costs. The experiment demonstrates that positive aggregate hedge P&L was possible on these paths when the selection rule knew future RV. It uses hindsight and ranks legs with the 10:00 quotes, while the forecast strategy commits to its signal at 09:45.

## Forecasting the entry before RV is known

The live-information rule replaces future RV with a forecast from data available at the decision. The original models forecast variance to each candidate's actual expiry. Their entries use the 09:45 forecast-minus-ask-IV gap. The diagnostic hit uses actual RV through the planned exit and the observed 10:00 entry ask IV. The separation matters because the option IV can change between signal and fill.

The original January–July 2026 forecast replay tested GARCH, EWMA, 21-day rolling variance, a HAR proxy, and histogram gradient boosting. Daily and hourly hedge schedules were evaluated. The daily GARCH run completed **48 one-contract trades** across the 7-, 30-, and 90-day targets (26, 14, and 8). It hit the volatility direction in **23 of 48** trades and earned **\$3,404 before execution costs**, falling to **−\$1,344 after costs**. All five models lost after costs under both schedules in that fixed-gate study. The [original research report](RESEARCH_REPORT.md) contains the full execution results.

## Historical tests of signal selection

The following experiments reuse the 2022–2025 development data. Supervised models refit by tenor using only completed labels available by the prior close, with up to 36 months of observations. The PDV threshold study groups by entry date, allowing 40 late-2025 positions to resolve in early 2026. All later historical ranking tests require the holding window to end by December 2025. The sample had already informed research decisions, so these are exploratory walk-forward comparisons.

### Path-dependent volatility and thresholds

The PDV adaptation weights historical signed returns and squared returns with fixed power-law kernels from the paper's S&P 500 estimates. Coefficients are fitted by tenor against an expiry-matched daily-return proxy. On 2,029 common opportunities, the >3-point PDV gate hit 277 of 683 signals (40.6%). The rolling 21-day and raw GARCH gates hit 218 of 536 (40.7%) and 263 of 664 (39.6%). These are each model's own threshold counts. [Guyon and Lekeufack](https://doi.org/10.1080/14697688.2023.2221281)

| PDV forecast gap | Hits / signals | Hit rate | Matched random rate |
| --- | ---: | ---: | ---: |
| >3 volatility points | 277 / 683 | 40.6% | 39.4% |
| >5 volatility points | 170 / 400 | 42.5% | 40.5% |
| >8 volatility points | 68 / 169 | 40.2% | 40.7% |
| >10 volatility points | 37 / 103 | 35.9% | 38.3% |
| >15 volatility points | 22 / 46 | 47.8% | 43.4% |

The matched random benchmark preserves the number of selections in each month and tenor. The >5-point result has a one-sided conditional p-value of 0.120. Raising the gate from 8 to 10 points removes 66 signals with 31 hits, which lowers the hit rate on the remaining selections.

### Direct hit probability and intraday volatility

Logistic models estimate the chance that holding RV exceeds entry ask IV. The intraday HAR models use realized variance from the previous 1, 5, and 22 sessions, with a specification separating upside, downside, and overnight variation. Quantile models estimate a distribution for holding RV and compare it with the decision-time ask IV. On the 1,804 opportunities with complete intraday histories, every model receives the same selection quota within each month and tenor, using the counts triggered by PDV's two gates. [Patton and Sheppard](https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf) · [Žikeš and Baruník](https://arxiv.org/abs/1308.4276)

| Model | Hits / 626 | Hits / 367 |
| --- | ---: | ---: |
| 21-day rolling | 269 | 170 |
| PDV | 261 | 165 |
| Raw GARCH | 256 | 158 |
| Logistic with IV and tenor | 259 | 162 |
| Logistic with daily features | 258 | 163 |
| Logistic with daily and intraday features | 258 | 165 |
| Intraday HAR-RV | 260 | 166 |
| Intraday HAR with signed and overnight features | 255 | 159 |
| Daily quantile model | 261 | 156 |
| Intraday quantile model | 262 | 156 |

The intraday quantile median reduced RV mean absolute error to 6.83 volatility points, compared with 8.78 for raw GARCH and 6.94 for the decision-time mid IV. At the 367-signal quota, it had 14 fewer hits than the rolling benchmark. The intraday logistic model's >60% group averaged a 67.8% predicted probability across 143 signals and realized 58 hits (40.6%). Better point estimates and higher model confidence did not deliver better entries in these tests.

### IV term structure and earnings information

The IV curve uses same-strike call and put mid IVs within each maturity at 09:45. Models add cross-tenor IV slopes or forward variance to the rolling volatility gap and decision-time IV. A separate model uses previously public SEC results filings to estimate the next release window and measure past disclosure moves. These features use estimated earnings timing. [Vasquez](https://doi.org/10.1017/S002210901700076X) · [SEC EDGAR submissions](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

| Model on 1,987 common opportunities | Hits / 677 | Hits / 397 |
| --- | ---: | ---: |
| 21-day rolling | 280 | 174 |
| IV slope | 277 | 168 |
| Forward variance | 274 | 169 |
| Earnings cadence and prior disclosure moves | 281 | 167 |
| IV slope and earnings features | 277 | 161 |

For the confirmed-date test, the earnings release must have been announced before the decision. Two same-strike option expiries at most 14 calendar days apart bracket the release. The pre-event expiry's IV sets the background variance rate. Subtracting that rate times the post-event time to expiry from post-event total variance gives an estimate of event variance. The historical comparison uses the root mean square of the close-to-open moves around the previous three to eight confirmed reports. Seventeen opportunities with negative inferred event variance remain unscored. [Dubinsky et al.](https://doi.org/10.1093/rfs/hhy060)

The filter leaves 271 opportunities across 11 stocks and 117 distinct earnings events. Across 150 stock and decision-date estimates, the median implied jump is 3.47% and the median historical estimate is 2.78%.

| Rule setting the matched quota | Earnings jump ranking | 21-day rolling ranking |
| --- | ---: | ---: |
| Historical jump estimate above implied jump | 30 / 72 (41.7%) | 34 / 72 (47.2%) |
| Top half within each month and tenor | 47 / 112 (42.0%) | 50 / 112 (44.6%) |
| Rolling gap above 3 points | 12 / 38 (31.6%) | 17 / 38 (44.7%) |
| Rolling gap above 5 points | 7 / 29 (24.1%) | 10 / 29 (34.5%) |

The positive-gap rule trails rolling volatility by 5.6 percentage points, with a paired four-month block-bootstrap 95% interval of −13.8 to +4.2 points. This comparison applies to the subset with advance notices and usable bracketing quotes.

## Conclusion

The project has not established a stable after-cost trading edge.
