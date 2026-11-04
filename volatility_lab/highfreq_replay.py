"""Research fork of the verified stock replay with explicit grid and signal lag.

The original financial engine is preserved. Defaults retain its 15-minute grid.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd
from .stock_replay import StockCosts,exercise_risk,YEAR


def replay(panel,contract,daily,side,frequency,costs=StockCosts(),capital=None,
           grid_minutes=15,signal_lag_minutes=15,allow_exercise_close=True):
    if grid_minutes<1 or signal_lag_minutes<grid_minutes or signal_lag_minutes%grid_minutes:
        raise ValueError('Signal lag must be a positive whole number of observation intervals')
    if frequency<grid_minutes or frequency%grid_minutes:
        raise ValueError('Hedge interval must be a positive whole number of observation intervals')
    if side not in (-1,0,1):raise ValueError("Invalid side")
    decision=pd.Timestamp(contract["decision_time"])
    entry=decision+pd.Timedelta(minutes=15)
    planned_exit=pd.Timestamp(contract["planned_exit"])
    expiry=pd.Timestamp(contract["expiry_time"])
    if decision not in panel.index or entry not in panel.index:raise ValueError("Missing entry grid")
    initial=panel.loc[entry]
    if not bool(initial.all_valid):raise ValueError("Invalid execution quotes at planned entry")
    if not np.isfinite(panel.loc[decision,"delta_per_straddle"]):raise ValueError("No valid decision delta")
    capital=float(capital or 3*100*initial.spot)
    if side==0:
        return {"side":0,"net_pnl":0.,"capital":capital,"return_on_capital":0.,"return_on_notional":0.,
                "fills":0,"max_drawdown_usd":0.,"exit_reason":"no_signal","reconciliation_error":0.},pd.DataFrame(),pd.DataFrame()
    p=panel.loc[decision:expiry].copy()
    signal_indexer=p.index.get_indexer(p.index-pd.Timedelta(minutes=signal_lag_minutes))
    # Only interim marks may carry a past value. Actual entry/exit prices
    # always require observed two-sided quotes and sufficient displayed size.
    p["option_mid_sum"]=(p.call_mid+p.put_mid).where(p.options_valid).ffill()
    cash,hedge,quantity=capital,0,0
    benchmark=capital
    account={k:0. for k in ("option_spread_cost","stock_spread_cost","slippage_cost","option_fees",
                           "stock_fees","cash_interest","benchmark_interest","borrow_cost","dividends","hedge_gross_pnl")}
    fills,ledger=[],[]
    last_hedge=None;exit_pending=None;exit_reason="planned_pre_expiry"
    previous_time=entry;previous_spot=float(initial.spot);previous_rate=float(initial.rate)
    entry_mid=float(initial.call_mid+initial.put_mid);exit_mid=None
    exited=False;abandoned=0

    def can_close(row):
        if bool(row.all_valid):return True
        if not costs.allow_unbid_long_abandonment or not bool(row.stock_passes_snapshot_filters):return False
        for kind in ("call","put"):
            flag=row.get(kind+"_conservative_condition_ok",False)
            if not bool(flag):return False
            bid,ask=float(row[kind+"_bid"]),float(row[kind+"_ask"])
            if not np.isfinite([bid,ask]).all() or bid<0 or ask<=0 or ask<bid:return False
            if quantity>0 and bid>0 and float(row[kind+"_bid_size"])<1:return False
            if quantity<0 and float(row[kind+"_ask_size"])<1:return False
        return True

    def trade_stock(change,row,time,signal_time,liquidation=False):
        nonlocal cash,hedge
        change=int(change)
        if not change:return
        side_name="ask" if change>0 else "bid"
        capacity=int(float(row["stock_"+side_name+"_size"])*costs.stock_quote_size_multiplier)
        if liquidation and abs(change)>capacity:raise ValueError("Insufficient displayed hedge liquidation size")
        change=int(np.sign(change)*min(abs(change),capacity))
        if not change:return
        mid=float(row.spot)
        quote=mid+(float(row["stock_"+side_name])-mid)*costs.stock_spread_fraction
        price=quote*(1+np.sign(change)*costs.extra_slippage_bps/10000)
        commission=max(costs.minimum_stock_commission,abs(change)*costs.stock_commission_per_share)*costs.fee_multiplier
        sell_fee=max(-change,0)*price*costs.stock_sell_fee_bps/10000*costs.fee_multiplier
        cash-=change*price+commission+sell_fee;hedge+=change
        account["stock_spread_cost"]+=abs(change)*abs(quote-mid)
        account["slippage_cost"]+=abs(change)*abs(price-quote)
        account["stock_fees"]+=commission+sell_fee
        fills.append({"time":time,"decision_time":signal_time,"instrument":"stock","quantity":change,
                      "fill_price":price,"fee":commission+sell_fee})

    def trade_options(change,row,time,signal_time):
        nonlocal cash,quantity,abandoned
        for kind in ("call","put"):
            side_name="ask" if change>0 else "bid"
            price=float(row[kind+"_"+side_name]);mid=float(row[kind+"_mid"])
            abandon=change<0 and quantity>0 and price==0 and costs.allow_unbid_long_abandonment
            if not abandon:price=mid+(price-mid)*costs.option_spread_fraction
            if not abandon and float(row[kind+"_"+side_name+"_size"])<abs(change):raise ValueError("Insufficient option displayed size")
            fee=0. if abandon else abs(change)*costs.option_commission_per_contract*costs.fee_multiplier
            if abandon:abandoned+=abs(change)
            cash-=change*100*price+fee
            account["option_spread_cost"]+=abs(change)*100*abs(price-mid)
            account["option_fees"]+=fee
            fills.append({"time":time,"decision_time":signal_time,"instrument":kind,"quantity":change,
                          "fill_price":price,"fee":fee,"executed":not abandon,
                          "action":"abandon_to_expiry_no_exercise" if abandon else "market_order_model"})
        quantity+=change

    for i in range(1,len(p)):
        time,row=p.index[i],p.iloc[i]
        prior_time,prior=p.index[i-1],p.iloc[i-1]
        if time<entry:continue
        if time>entry:
            tau=(time-previous_time).total_seconds()/YEAR
            lend=max(previous_rate-costs.lending_spread,0.)
            rate=lend if cash>=0 else previous_rate+costs.debit_spread
            interest=cash*np.expm1(rate*tau)
            binterest=benchmark*np.expm1(lend*tau)
            borrow=max(-hedge,0)*previous_spot*costs.short_stock_borrow_rate*tau
            account["cash_interest"]+=interest;account["benchmark_interest"]+=binterest;account["borrow_cost"]+=borrow
            account["hedge_gross_pnl"]+=hedge*(float(row.spot)-previous_spot)
            cash+=interest-borrow;benchmark+=binterest
            old_date=previous_time.tz_convert("America/New_York").date()
            new_date=time.tz_convert("America/New_York").date()
            if new_date!=old_date:
                dividend=float(daily.dividends.get(new_date,0.))*hedge
                cash+=dividend;account["dividends"]+=dividend
        signal_index=int(signal_indexer[i])
        lag_ok=(time-prior_time).total_seconds()==grid_minutes*60 and signal_index>=0
        signal_time=p.index[signal_index] if signal_index>=0 else prior_time
        signal=p.iloc[signal_index] if signal_index>=0 else prior
        if time==entry:trade_options(side,row,time,decision)
        if time>=planned_exit and exit_pending is None:exit_pending=planned_exit-pd.Timedelta(minutes=15)
        if allow_exercise_close and quantity<0 and exit_pending is None and exercise_risk(prior,contract["strike"]):
            exit_pending=prior_time;exit_reason="short_exercise_risk_close"
        if exit_pending is not None and can_close(row) and time>exit_pending:
            exit_mid=float(row.call_mid+row.put_mid)
            trade_options(-quantity,row,time,exit_pending)
            trade_stock(-hedge,row,time,exit_pending,liquidation=True)
            exited=True
        elif bool(row.stock_passes_snapshot_filters) and lag_ok and bool(signal.stock_passes_snapshot_filters) and np.isfinite(signal.delta_per_straddle):
            if last_hedge is None or (time-last_hedge).total_seconds()>=frequency*60:
                target=int(np.rint(-quantity*100*float(signal.delta_per_straddle)))
                trade_stock(target-hedge,row,time,signal_time)
                last_hedge=time
        option_value=quantity*100*float(row.option_mid_sum)
        equity=cash+hedge*float(row.spot)+option_value
        # Stated research reserve, not exchange/broker margin. All cohorts
        # start with three times the underlying notional in cash.
        reserve=max(quantity,0)*100*float(row.option_mid_sum)+max(-quantity,0)*100*float(row.spot)+.5*abs(hedge)*float(row.spot)
        if equity<reserve and exit_pending is None:
            exit_pending=time;exit_reason="research_capital_reserve_close"
        ledger.append({"time":time,"equity":equity,"cash":cash,"benchmark":benchmark,"excess_pnl":equity-benchmark,
                       "stock_shares":hedge,"option_contracts_per_leg":quantity,
                       "delta_shares":hedge+quantity*100*float(row.delta_per_straddle),"reserve":reserve})
        previous_time,previous_spot,previous_rate=time,float(row.spot),float(row.rate)
        if exited:break
    if not exited:raise ValueError("No valid observed liquidation before expiry; outcome unresolved")
    gross_option=side*100*(exit_mid-entry_mid)
    net=cash-benchmark
    expected=(gross_option+account["hedge_gross_pnl"]+account["dividends"]+account["cash_interest"]-
              account["benchmark_interest"]-account["borrow_cost"]-
              sum(account[k] for k in ("option_spread_cost","stock_spread_cost","slippage_cost","option_fees","stock_fees")))
    residual=net-expected
    if abs(residual)>1e-5:raise AssertionError("Stock replay cash reconciliation failed")
    path=pd.DataFrame(ledger);drawdown=(path.excess_pnl.cummax().clip(lower=0)-path.excess_pnl).max()
    result={"side":side,"net_pnl":net,"gross_option_pnl":gross_option,**account,"capital":capital,
            "return_on_capital":net/capital,"return_on_notional":net/(100*float(initial.spot)),
            "entry":str(entry),"exit":str(previous_time),"exit_reason":exit_reason,"fills":sum(f.get("executed",True) for f in fills),
            "abandoned_long_legs":abandoned,
            "max_drawdown_usd":float(drawdown),"reconciliation_error":residual,"cost_parameters":asdict(costs),
            "observation_grid_minutes":grid_minutes,"signal_lag_minutes":signal_lag_minutes,
            "exercise_close_enabled":allow_exercise_close}
    return result,path,pd.DataFrame(fills)
