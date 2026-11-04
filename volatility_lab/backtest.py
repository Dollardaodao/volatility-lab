"""Cash-funded European straddle + spot-hedge pilot.

Option bid/ask are observed. Spot fills are candle-close proxies with configurable
spread/slippage and fees, NOT historical executable spot quotes. Underlying-coin
premiums are converted to/from USD at each option transaction. No coin collateral
is silently treated as USD. No perpetuals or their funding rates are used.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .forecast import YEAR_SECONDS
from .pricing import implied_vol_delta


@dataclass(frozen=True)
class Costs:
    capital_usd: float = 100_000.
    option_units: float = .1  # Underlying-equivalent units, NOT a venue contract count.
    option_min_units: float = .1
    option_unit_step: float = .1
    hedge_every_minutes: int = 60
    spot_half_spread_bps: float = 1.
    spot_slippage_bps: float = 1.
    spot_fee_bps: float = 5.
    option_notional_fee_bps: float = 3.
    option_fee_premium_cap: float = .125
    cash_lend_rate: float = .02
    cash_debit_rate: float = .06
    short_asset_borrow_rate: float = .08
    pricing_rate: float = .04
    short_option_collateral_fraction: float = 1.
    hedge_collateral_fraction: float = .5
    max_gap_seconds: float = 300.
    max_quote_age_seconds: float = 60.


def enrich_panel(panel, pricing_rate=.04):
    panel = panel.copy()
    if not panel.index.is_monotonic_increasing or panel.index.has_duplicates:
        raise ValueError("Execution times must be strictly increasing")
    for kind in ("call", "put"):
        expiry = panel[f"{kind}_expiration"]
        years = (expiry-panel.index).dt.total_seconds()/YEAR_SECONDS
        mid = (panel[f"{kind}_bid_price"]+panel[f"{kind}_ask_price"])/2
        ivs, deltas = [], []
        for price, spot, strike, tau in zip(mid, panel.spot, panel[f"{kind}_strike_price"], years):
            if not np.isfinite([price, spot, strike, tau]).all():
                ivs.append(np.nan); deltas.append(np.nan)
            else:
                iv, delta = implied_vol_delta(price*spot,spot,strike,tau,pricing_rate,kind)
                ivs.append(iv); deltas.append(delta)
        panel[f"{kind}_iv"] = ivs
        panel[f"{kind}_delta"] = deltas
        panel[f"{kind}_mid_coin"] = mid
    panel["straddle_mid_usd"] = (panel.call_mid_coin+panel.put_mid_coin)*panel.spot
    panel["straddle_delta"] = panel.call_delta+panel.put_delta
    return panel


def run_pilot(panel, side, costs=Costs()):
    """Fixed first decision, next-bar entry, last-bar exit; no outcome-based exits.

    side: +1 long gamma, -1 short gamma, 0 no trade. All hedge targets use the
    PREVIOUS row. Available_at/exchange timestamps must precede their fill time,
    and the fill's option quote must postdate the decision. Missing inputs fail
    closed. Margin is a conservative research reserve, not a venue margin engine.
    """
    if side not in (-1, 0, 1) or len(panel) < 4:
        raise ValueError("Need side -1/0/1 and at least four observations")
    if min(costs.capital_usd,costs.option_units,costs.hedge_every_minutes) <= 0:
        raise ValueError("Capital, size, and hedge interval must be positive")
    if costs.option_units < costs.option_min_units or costs.option_unit_step <= 0 or not np.isclose(
            costs.option_units/costs.option_unit_step,round(costs.option_units/costs.option_unit_step)):
        raise ValueError("Option order violates minimum size or lot step")
    if costs.cash_debit_rate < costs.cash_lend_rate:
        raise ValueError("Debit rate cannot be below cash lending rate")
    required = ["spot","straddle_mid_usd","straddle_delta"]
    for k in ("call","put"):
        required += [f"{k}_{x}" for x in ("bid_price","ask_price","bid_amount","ask_amount")]
    if not np.isfinite(panel[required].to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite executable quotes, sizes or model deltas")
    if (panel.spot <= 0).any():
        raise ValueError("Invalid spot")
    for k in ("call","put"):
        if ((panel[f"{k}_bid_price"] <= 0) | (panel[f"{k}_ask_price"] < panel[f"{k}_bid_price"])).any():
            raise ValueError("Invalid/crossed option quote")
        if panel[f"{k}_symbol"].nunique() != 1 or panel[f"{k}_expiration"].nunique() != 1:
            raise ValueError("Cannot change contracts inside an open trade")
        if (panel.index >= panel[f"{k}_expiration"]).any():
            raise ValueError("Pilot must close before expiry; settlement is not implemented")
        for column in (f"{k}_available_at",f"{k}_timestamp"):
            if panel[column].isna().any() or (panel[column] > panel.index).any():
                raise ValueError("Look-ahead in execution inputs")
        if ((panel.index-panel[f"{k}_timestamp"]).dt.total_seconds() > costs.max_quote_age_seconds).any():
            raise ValueError("Stale option quote")
    if not (panel.call_expiration == panel.put_expiration).all() or not (panel.call_strike_price == panel.put_strike_price).all():
        raise ValueError("Call and put must form one same-strike, same-expiry straddle")
    if np.max(np.diff(panel.index.as_unit("ns").asi8)/1e9) > costs.max_gap_seconds:
        raise ValueError("Data gap exceeds permitted execution interval")
    cash, hedge, quantity = costs.capital_usd, 0., 0.
    spread = (costs.spot_half_spread_bps+costs.spot_slippage_bps)/10000
    account = {"option_spread_cost":0., "fx_spread_cost":0., "hedge_spread_cost":0.,
               "option_fees":0., "hedge_fees":0., "cash_interest":0., "borrow_cost":0.,
               "hedge_gross_pnl":0.}
    trades, ledger = [], []
    previous_time, previous_spot = panel.index[1], float(panel.spot.iloc[1])
    entry_mid, exit_mid = float(panel.straddle_mid_usd.iloc[1]), float(panel.straddle_mid_usd.iloc[-1])
    last_hedge_time = None

    def spot_trade(change, spot, time, decision_time):
        nonlocal cash, hedge
        if abs(change) < 1e-12:
            return
        fill = spot * (1+np.sign(change)*spread)
        fee = abs(change)*fill*costs.spot_fee_bps/10000
        cash -= change*fill+fee
        account["hedge_spread_cost"] += abs(change)*spot*spread
        account["hedge_fees"] += fee
        hedge += change
        trades.append({"time":time,"decision_time":decision_time,"instrument":"spot_proxy",
                       "units":change,"fill_usd":fill,"fee_usd":fee})

    def option_trade(change, row, time, decision_time):
        nonlocal cash, quantity
        if change == 0:
            return
        for k in ("call","put"):
            if row[f"{k}_timestamp"] <= decision_time:
                raise ValueError("Fill quote must postdate the decision")
            quote_side = "ask" if change > 0 else "bid"
            if row[f"{k}_{quote_side}_amount"] < abs(change):
                raise ValueError("Order exceeds displayed option quote size")
            price = float(row[f"{k}_{quote_side}_price"])
            midpoint = float(row[f"{k}_mid_coin"])
            # Paying coin requires buying coin at the spot ask; receiving coin
            # is converted by selling at the spot bid. The USD cash ledger
            # therefore includes conversion cost and never ignores coin FX.
            fx = row.spot * (1+np.sign(change)*spread)
            premium = abs(change)*price*row.spot
            fee = min(abs(change)*row.spot*costs.option_notional_fee_bps/10000,
                      costs.option_fee_premium_cap*premium)
            cash -= change*price*fx+fee
            account["option_spread_cost"] += abs(change)*abs(price-midpoint)*row.spot
            account["fx_spread_cost"] += abs(change)*price*row.spot*spread
            account["option_fees"] += fee
            trades.append({"time":time,"decision_time":decision_time,
                           "instrument":row[f"{k}_symbol"],"units":change,
                           "fill_coin":price,"fx_usd":fx,"fee_usd":fee})
        quantity += change

    for i in range(1,len(panel)):
        time, row, previous = panel.index[i], panel.iloc[i], panel.iloc[i-1]
        decision_time = panel.index[i-1]
        if i > 1:
            tau = (time-previous_time).total_seconds()/YEAR_SECONDS
            interest = cash*np.expm1((costs.cash_lend_rate if cash >= 0 else costs.cash_debit_rate)*tau)
            borrow = max(-hedge,0.)*previous_spot*costs.short_asset_borrow_rate*tau
            account["cash_interest"] += interest
            account["borrow_cost"] += borrow
            account["hedge_gross_pnl"] += hedge*(row.spot-previous_spot)
            cash += interest-borrow
        if i == 1:
            option_trade(side*costs.option_units,row,time,decision_time)
        if i == len(panel)-1:
            option_trade(-quantity,row,time,decision_time)
            spot_trade(-hedge,row.spot,time,decision_time)
        elif last_hedge_time is None or (time-last_hedge_time).total_seconds() >= costs.hedge_every_minutes*60:
            target = -quantity*float(previous.straddle_delta)
            spot_trade(target-hedge,row.spot,time,decision_time)
            last_hedge_time = time
        option_value = quantity*float(row.straddle_mid_usd)
        equity = cash+hedge*row.spot+option_value
        reserve = (max(-quantity,0.)*row.spot*costs.short_option_collateral_fraction+
                   abs(hedge)*row.spot*costs.hedge_collateral_fraction)
        if equity < reserve:
            raise ValueError("Research collateral reserve breached; no venue liquidation simulation")
        elapsed = (time-panel.index[1]).total_seconds()/YEAR_SECONDS
        benchmark = costs.capital_usd*np.exp(costs.cash_lend_rate*elapsed)
        ledger.append({"time":time,"cash_usd":cash,"spot_units":hedge,"option_units":quantity,
                       "option_value_usd":option_value,"spot":float(row.spot),"equity_usd":equity,
                       "cash_benchmark_usd":benchmark,"excess_pnl_usd":equity-benchmark,
                       "collateral_reserve_usd":reserve,"delta_units":hedge+quantity*row.straddle_delta})
        previous_time, previous_spot = time, float(row.spot)
    elapsed = (panel.index[-1]-panel.index[1]).total_seconds()/YEAR_SECONDS
    benchmark_interest = costs.capital_usd*np.expm1(costs.cash_lend_rate*elapsed)
    gross_option_pnl = side*costs.option_units*(exit_mid-entry_mid)
    excess = cash-costs.capital_usd-benchmark_interest
    calculated = (gross_option_pnl+account["hedge_gross_pnl"]+account["cash_interest"]-
                  account["borrow_cost"]-benchmark_interest-
                  sum(account[k] for k in ("option_spread_cost","fx_spread_cost","hedge_spread_cost","option_fees","hedge_fees")))
    residual = excess-calculated
    if abs(residual) > 1e-6:
        raise AssertionError(f"Cash ledger does not reconcile: {residual}")
    result = {"side":side,"entry":str(panel.index[1]),"exit":str(panel.index[-1]),
              "gross_option_pnl":gross_option_pnl,**account,"benchmark_interest":benchmark_interest,
              "net_excess_pnl_usd":excess,"pnl_over_committed_capital":excess/costs.capital_usd,
              "reconciliation_error_usd":residual,"trade_fills":len(trades),"cost_assumptions":asdict(costs)}
    return result, pd.DataFrame(ledger), pd.DataFrame(trades)
