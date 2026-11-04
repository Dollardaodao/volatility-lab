# Reproducibility and data access

## Three different levels of reproduction

1. **Offline code checks:** run the 55 unit tests with synthetic fixtures and no account or market-data download.
2. **Published-result checks:** verify source/plan hashes, aggregate accounting identities, cross-table totals, model/horizon coverage, and report links using `scripts/verify_release.py`. Rebuild the Markdown reports with `scripts/build_research_report.py`.
3. **Complete historical replay and audit:** restore the original licensed input caches, fitted-run records, and baseline books, then run the archival study and validation modules. These inputs are not included in this public repository.

The original audit manifest in `reports/oos_2026/verification.json` describes level 3 as performed locally. Public CI runs levels 1 and 2. It does not re-download or re-audit the original licensed quotes.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
python scripts/verify_release.py
python scripts/build_research_report.py
```

For the authenticated data adapter and complete research dependencies:

```bash
python -m pip install -e '.[ml,theta]'
cp .env.example .env
# Set THETADATA_API_KEY locally in the ignored .env file.
```

The account needs the historical quote entitlements used by the study. The adapter does not purchase data or change subscriptions. The original runtime used `GRPC_DNS_RESOLVER=native`; that setting may help if the platform's default gRPC DNS resolver fails.

`requirements.lock.txt` records the original research environment. The project metadata provides dependency ranges for installation; a fresh environment is not guaranteed to reproduce the original platform byte-for-byte.

## Local input layout

The archival code resolves paths relative to the project root. It intentionally keeps data and generated artifacts outside version control.

| Path | Contents |
| --- | --- |
| `data/stock_study/{SYMBOL}_daily_actions.csv` | Daily OHLCV, dividends and splits, with a dated first column; original coverage starts in 2018 and ends November 2, 2026 |
| `data/stock_study/sofr.json` | New York Fed reference-rate payload with `refRates`, `effectiveDate` and `percentRate` |
| `data/oos_single_leg/decision_quotes/` | Observed 09:45 New York stock snapshots and full eligible option-chain snapshots |
| `data/near_continuous/quotes/` | Actual one-minute stock and selected-option quote caches and request/hash metadata |
| `artifacts/oos_single_leg/` | Local protocol lock, configuration, model fits, decisions, priced panels, fills, ledgers and audits |
| `artifacts/monthly_single_leg/` | Original 2022–2025 predecessor study and baseline results used by archival integrity checks |

Daily CSV columns consumed by the code are `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends` and `Stock Splits`. Timestamps must represent the same daily sessions as the exchange calendar. `DailyInputs` maps daily observations to their actual exchange-close availability times and reconstructs historical cash-dividend units across splits.

Raw quote files and their normalization are described by `MarketStore` in `volatility_lab/stock_market.py`, the authenticated adapter in `volatility_lab/theta_sdk.py`, and `normalize_quotes` in `volatility_lab/theta.py`. Missing quotes are not replaced with synthetic interpolated trades. Quoted timestamps do not independently establish original tick freshness.

## Original study entry points

With the **complete matching original local inputs and artifacts restored**, the existing stage commands are:

```bash
GRPC_DNS_RESOLVER=native python -m volatility_lab.oos_single_leg forecasts --workers 4
GRPC_DNS_RESOLVER=native python -m volatility_lab.oos_single_leg decisions
python -m volatility_lab.oos_single_leg select
GRPC_DNS_RESOLVER=native python -m volatility_lab.oos_single_leg trade --workers 4
python -m volatility_lab.oos_single_leg_report
python -m volatility_lab.oos_single_leg_validate
```

The study requires a local `protocol_lock.json`; it is not an automatic fresh-clone data bootstrap. Its recorded reference copy is published under `reports/oos_2026/`. Source and input hashes prevent silently mixing changed numerical rules or data with cached original results. Full validation also checks predecessor results and original unit-test logs. The public repository does not include those large predecessor books or authenticate on import.

The original HTML-report generator contains study-specific historical commentary. The public Markdown builder instead reads the released aggregate files and is the entry point for rebuilding this repository's reports.

## New experiments

For new data or rules, create a separate plan and output directory, record a new lock timestamp, and preserve prior results. The dated study drivers are archival research scripts rather than a general production backtesting CLI. Their fixed experiment paths and integrity assertions must be adapted explicitly for a new study; the core forecasting, pricing and replay functions can be reused.

Any rerun of these now-published 2026 dates is a retrospective reproduction. It does not create another untouched holdout. Revised corporate-action data or quote history can change results, and matching aggregate P&L alone does not prove that every original fill was reproduced.

## Audit and source provenance

- `reports/oos_2026/configuration.json`: original plan, numerical-source SHA-256 hashes, daily-input hashes, cost/rate hashes and package versions.
- `reports/oos_2026/protocol_lock.json`: original protocol hash and lock metadata.
- `reports/oos_2026/verification.json`: original complete-run audit findings and result hashes; some referenced private-run files are intentionally absent from the public repository.
- `reports/publication_manifest.json`: SHA-256 hashes of files included in this release.

No API key, session token, raw quote file, per-minute ledger, fitted model or local virtual environment is required to read the reports or run the public checks.
