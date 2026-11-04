"""Midpoint cash attribution on the observed book's identical orders and exit."""
import numpy as np
import pandas as pd
from dataclasses import asdict,replace
from .stock_replay import YEAR


def midpoint_book(panel,result,ledger,fills,costs):
    book=fills.copy();costs_at_time=pd.Series(0.,index=pd.DatetimeIndex(ledger.time))
    for instrument,group in fills.groupby('instrument'):
        is_stock=instrument=='stock';rows=panel.loc[pd.DatetimeIndex(group.time)]
        mid=rows.spot.to_numpy() if is_stock else rows.option_mid.to_numpy()
        executed=group.executed.to_numpy(bool)
        fair=np.where(executed,mid,group.fill_price.to_numpy())
        multiplier=1 if is_stock else 100
        saving=group.quantity.to_numpy()*multiplier*(group.fill_price.to_numpy()-fair)+group.fee.to_numpy()
        series=pd.Series(saving,index=pd.DatetimeIndex(group.time)).groupby(level=0).sum()
        costs_at_time=costs_at_time.add(series,fill_value=0.)
        book.loc[group.index,'fill_price']=fair;book.loc[group.index,'fee']=0.
    e=costs_at_time.reindex(pd.DatetimeIndex(ledger.time),fill_value=0.).to_numpy()
    if np.min(e,initial=0.) < -1e-7:raise AssertionError('Midpoint order attribution unexpectedly increases costs')
    if abs(e.sum()-result['execution_cost'])>1e-5:raise AssertionError('Saved order costs do not equal execution-cost bridge')
    cash=ledger.cash.to_numpy();benchmark=ledger.benchmark.to_numpy()
    if (cash>=0).all():
        growth=benchmark/result['capital'];difference=growth*np.cumsum(e/growth)
    else:
        # Only the interest difference needs a loop if the real book borrows.
        times=pd.DatetimeIndex(ledger.time).as_unit('ns');years=np.r_[0.,np.diff(times.asi8)/1e9/YEAR]
        rates=panel.loc[times,'rate'].to_numpy();lend=np.expm1(np.maximum(np.roll(rates,1)-costs.lending_spread,0.)*years)
        debit=np.expm1((np.roll(rates,1)+costs.debit_spread)*years);difference=np.zeros(len(cash));difference[0]=e[0]
        for i in range(1,len(cash)):
            prior=difference[i-1];actual_interest=cash[i-1]*(lend[i] if cash[i-1]>=0 else debit[i])
            ideal_cash=cash[i-1]+prior;ideal_interest=ideal_cash*(lend[i] if ideal_cash>=0 else debit[i])
            difference[i]=prior+ideal_interest-actual_interest+e[i]
    ideal=ledger.copy()
    for column in ['cash','equity','excess_pnl']:ideal[column]=ideal[column].to_numpy()+difference
    r=dict(result);r['net_pnl']=float(ideal.excess_pnl.iloc[-1]);r['cash_interest']+=float(difference[-1]-e.sum())
    r['option_spread_cost']=r['abandonment_writeoff']
    for column in ['stock_spread_cost','slippage_cost','option_fees','stock_fees','execution_cost']:r[column]=0.
    r['net_financing']=r['cash_interest']-r['benchmark_interest']-r['borrow_cost']
    r['return_on_capital']=r['net_pnl']/r['capital']
    r['return_on_notional']=r['net_pnl']/(100*float(panel.loc[pd.Timestamp(result['entry']),'spot']))
    r['cost_parameters']=asdict(replace(costs,fee_multiplier=0,option_spread_fraction=0,stock_spread_fraction=0,extra_slippage_bps=0))
    r['max_drawdown_usd']=float((ideal.excess_pnl.cummax().clip(lower=0)-ideal.excess_pnl).max())
    r['reconciliation_error']=r['net_pnl']-(r['market_pnl']-r['abandonment_writeoff']+r['net_financing'])
    if abs(r['reconciliation_error'])>1e-5:raise AssertionError('Midpoint cash attribution does not reconcile')
    return r,ideal,book
