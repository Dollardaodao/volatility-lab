# Historical Tests of Realized Volatility Above Implied Volatility

**Research update · November 8, 2026**

Across the 2022–2025 development sample, the new signals did not reliably improve on ranking by 21-day rolling volatility minus the option's ask implied volatility (IV). A quantile model reduced the average error in forecasting realized volatility (RV), while its top-ranked options hit less often. The earnings jump signal also trailed the rolling benchmark when both received the same number of selections.

## Sample and comparison

The opportunity grid covers 20 US stocks on the first trading session of each month, with 7-, 30-, and 90-calendar-day target maturities. Each opportunity uses the original single call or put selected from the 09:45 New York quote snapshot. A hit means the annualized RV from 10:00 through the planned pre-expiry exit exceeds that option's observed 10:00 entry ask IV. Holding RV uses 15-minute returns and overnight moves.

The historical threshold grid groups trades by entry date, so some late-2025 positions close in early 2026. The subsequent ranking tests require the entire holding window to end by December 2025. Supervised models refit monthly by target tenor. Training uses only labels completed by the prior close, with up to 36 months of eligible observations. Model comparisons use the same eligible opportunities and allocate the same number of selections to each model within each entry month and tenor. The development sample had already informed earlier research decisions, so these are exploratory walk-forward tests. The [original research report](RESEARCH_REPORT.md) covers the separate 2026 execution study and its trading costs.

## Path-dependent volatility and entry thresholds

The path-dependent volatility (PDV) adaptation weights historical signed returns and squared returns with fixed power-law kernels from the paper's S&P 500 estimates. The forecast coefficients are refitted by tenor against an expiry-matched daily-return proxy. On the 2,029 opportunities available to all eight models from March 2023 through December 2025, PDV hit 277 of 683 signals (40.6%) at a 3-point gate. The 21-day rolling benchmark hit 218 of 536 (40.7%) and raw GARCH hit 263 of 664 (39.6%). Each model applies the gate to its own forecast-minus-ask-IV score, so the counts differ. [Guyon and Lekeufack](https://doi.org/10.1080/14697688.2023.2221281)

| PDV gap at 09:45 | Hits / signals | Hit rate | Matched random hit rate |
| --- | ---: | ---: | ---: |
| >3 volatility points | 277 / 683 | 40.6% | 39.4% |
| >5 volatility points | 170 / 400 | 42.5% | 40.5% |
| >8 volatility points | 68 / 169 | 40.2% | 40.7% |
| >10 volatility points | 37 / 103 | 35.9% | 38.3% |
| >15 volatility points | 22 / 46 | 47.8% | 43.4% |

The matched random benchmark preserves the number of selections in each month and tenor. The >5-point result has a one-sided conditional p-value of 0.120. Raising the threshold from 8 to 10 points removes 66 signals with 31 hits (47.0%), leaving a lower hit rate among the retained signals.

## Hit probabilities and intraday RV

The probability models predict the binary hit directly from decision-time inputs. The intraday HAR models use realized variance over the preceding 1, 5 and 22 sessions, with a separate specification for upside, downside and overnight variation. Quantile models estimate a distribution for holding-period RV and evaluate it against the 09:45 ask IV as an approximation to the later execution threshold. [Patton and Sheppard](https://public.econ.duke.edu/~ap172/Patton_Sheppard_REStat_2015.pdf) · [Žikeš and Baruník](https://arxiv.org/abs/1308.4276)

Requiring complete intraday histories leaves 1,804 shared opportunities. PDV's >3- and >5-point gates set the two selection quotas in each month and tenor. All models rank the same candidates.

| Model | Hits / 626 | Hits / 367 |
| --- | ---: | ---: |
| 21-day rolling | 269 | 170 |
| PDV | 261 | 165 |
| Raw GARCH | 256 | 158 |
| Logistic model with IV and tenor | 259 | 162 |
| Logistic model with daily features | 258 | 163 |
| Logistic model with daily and intraday features | 258 | 165 |
| Intraday HAR-RV | 260 | 166 |
| Intraday HAR with signed and overnight features | 255 | 159 |
| Daily quantile model | 261 | 156 |
| Intraday quantile model | 262 | 156 |

At a probability cutoff of 60%, the intraday logistic model averaged 67.8% predicted probability across 143 signals and realized 58 hits (40.6%). Its realized hit rate stayed close to 41% as the cutoff moved from 50% to 55% to 60%. The intraday quantile model's median forecast had a mean absolute error of 6.83 volatility points, versus 8.78 for raw GARCH and 6.94 for the decision-time market mid IV. Its nominal 80% prediction interval covered 66.1% of outcomes. The lower point error came with 14 fewer hits than the 21-day rolling ranking at the 367-signal quota.

## IV term structure and historical earnings information

The IV curve averages same-strike call and put midpoint IVs within each maturity at 09:45. The probability models add the 7-to-30- and 30-to-90-day IV slopes or the forward variances derived from total variance at actual expiries. Both use the rolling-volatility gap and decision-time IV as controls. The earnings model uses previously public SEC results disclosures to estimate the timing of the next report and uses price moves around earlier disclosures. The timing feature is a calendar proxy. [Vasquez](https://doi.org/10.1017/S002210901700076X) · [SEC EDGAR submissions](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

The matched comparison covers 1,987 opportunities.

| Model | Hits / 677 | Hits / 397 |
| --- | ---: | ---: |
| 21-day rolling | 280 | 174 |
| IV slope | 277 | 168 |
| Forward variance | 274 | 169 |
| Earnings cadence and prior disclosure moves | 281 | 167 |
| IV slope and earnings features | 277 | 161 |

On 30-day options alone, ranking by 90-day minus 30-day IV hit 56 of 135 signals, versus 57 for the 21-day rolling score. Paired month-block intervals for the new models' differences versus 21-day rolling include zero.

## Earnings jumps with advance issuer confirmation

Dated issuer notices establish which release dates were public before each decision. The historical jump estimate uses the root mean square of the stock's close-to-open return around its previous three to eight confirmed earnings releases. The option-implied estimate uses a same-strike call and put pair at each of two expiries bracketing the scheduled release, quoted at 09:45. The expiries are at most 14 calendar days apart. Assuming a flat background variance rate equal to the pre-event expiry's IV squared, implied jump variance is the post-event total implied variance less that background rate times the post-event time to expiry. Seventeen opportunities with negative implied jump variance are unscored. [Dubinsky et al.](https://doi.org/10.1093/rfs/hhy060)

The filters leave 271 option opportunities across 11 stocks and 117 distinct stock earnings events. Across 150 stock and decision-date pairs with estimates, the median implied jump magnitude was 3.47% and the median historical estimate was 2.78%.

| Rule setting the matched quota | Earnings jump ranking | 21-day rolling ranking |
| --- | ---: | ---: |
| Historical jump estimate above implied jump | 30 / 72 (41.7%) | 34 / 72 (47.2%) |
| Top half within each month and tenor | 47 / 112 (42.0%) | 50 / 112 (44.6%) |
| 21-day rolling gap above 3 points | 12 / 38 (31.6%) | 17 / 38 (44.7%) |
| 21-day rolling gap above 5 points | 7 / 29 (24.1%) | 10 / 29 (34.5%) |

The positive-gap rule trailed by 5.6 percentage points. A paired four-month block bootstrap puts the 95% interval for the difference at −13.8 to +4.2 points. The comparison reflects the smaller subset with advance notices and usable bracketing quotes. The jump estimate depends on the flat background variance assumption. These figures measure the original single option's holding-period hit rate.
