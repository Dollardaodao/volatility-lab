"""Schedule-controlled fork of the audited single-option cash replay.

The original engine is unchanged. Elapsed-time defaults reproduce its cash;
new schedules distinguish calendar hours, trading hours, and daily clocks.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd
from .stock_replay import StockCosts, YEAR
from .stock_market import sessions


def replay(panel, contract, daily, frequency, costs=StockCosts(), grid_minutes=1,
           signal_lag_minutes=1, capital=None, schedule_kind="elapsed", daily_time="10:00"):
    if schedule_kind not in ("elapsed", "trading_time", "daily_fixed", "daily_close"):
        raise ValueError("Unknown hedge schedule")
    if frequency < grid_minutes or frequency % grid_minutes or signal_lag_minutes < grid_minutes or signal_lag_minutes % grid_minutes:
        raise ValueError('Positive integral grid, hedge interval and past-signal delay required')
    decision = pd.Timestamp(contract['decision_time']); entry = decision + pd.Timedelta(minutes=15)
    planned = pd.Timestamp(contract['planned_exit']); expiry = pd.Timestamp(contract['expiry_time'])
    p = panel.loc[decision:expiry].copy(); times = p.index.as_unit('ns'); p.index = times
    if not times.is_monotonic_increasing or times.has_duplicates: raise ValueError('Non-unique quote grid')
    if entry not in times or decision not in times: raise ValueError('Missing decision/entry grid')
    entry_i = times.get_loc(entry)
    columns = ['spot','stock_bid','stock_ask','stock_bid_size','stock_ask_size','option_bid','option_ask',
               'option_bid_size','option_ask_size','option_mid','option_delta','rate',
               'stock_passes_snapshot_filters','option_passes_snapshot_filters','option_conservative_condition_ok']
    arrays = {c: p[c].to_numpy() for c in columns}
    a = arrays; stock_ok = p.stock_passes_snapshot_filters.fillna(False).to_numpy(bool)
    option_ok = p.option_passes_snapshot_filters.fillna(False).to_numpy(bool)
    conditions = p.option_conservative_condition_ok.fillna(False).to_numpy(bool)
    if not stock_ok[entry_i] or not option_ok[entry_i]: raise ValueError('Invalid observed entry quotes')
    if not np.isfinite(a['spot']).all() or (a['spot'] <= 0).any(): raise ValueError('Invalid stock marks')
    if not np.isfinite(a['option_delta'][times.get_loc(decision)]): raise ValueError('Missing decision Delta')
    option_marks = p.option_mid.where(option_ok).ffill().to_numpy()
    if not np.isfinite(option_marks[entry_i:]).all(): raise ValueError('No past option mark')
    ns = times.asi8; signal_index = times.get_indexer(times - pd.Timedelta(minutes=signal_lag_minutes))
    years = np.r_[0., np.diff(ns)/1e9/YEAR]
    lend = np.maximum(np.roll(a['rate'], 1)-costs.lending_spread, 0.)
    lending_growth = np.expm1(lend*years)
    debit_growth = np.expm1((np.roll(a['rate'],1)+costs.debit_spread)*years)
    local_dates = times.tz_convert('America/New_York').date
    timestamps = times.to_list()
    clock = ns
    if schedule_kind == 'trading_time':
        same_day = np.r_[False, local_dates[1:] == local_dates[:-1]]
        clock = np.cumsum(np.where(same_day, np.r_[0, np.diff(ns)], 0))
    daily_targets = {}
    if schedule_kind == 'daily_fixed':
        daily_targets = {d: pd.Timestamp(str(d)+' '+daily_time, tz='America/New_York').tz_convert('UTC') for d in set(local_dates)}
    elif schedule_kind == 'daily_close':
        calendar = sessions(decision.date(), expiry.date())
        daily_targets = {t.date(): row['close']-pd.Timedelta(minutes=15) for t,row in calendar.iterrows()}
    last_hedge_day = None
    dividend_per_share = np.zeros(len(p))
    for i in range(entry_i+1,len(p)):
        if local_dates[i] != local_dates[i-1]: dividend_per_share[i] = daily.dividends.get(local_dates[i],0.)
    initial_spot = float(a['spot'][entry_i]); capital = float(capital or 3*100*initial_spot)
    cash = capital; benchmark = capital; hedge = 0; quantity = 0
    account = {k: 0. for k in ['option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees',
                                'cash_interest','benchmark_interest','borrow_cost','dividends','hedge_gross_pnl']}
    fills = []; last_hedge_ns = None; pending = None; reason = 'planned_pre_expiry'; abandoned = 0
    ledger = {c: np.empty(len(p)-entry_i) for c in ['equity','cash','benchmark','excess_pnl','stock_shares',
                                                  'option_contracts','delta_shares','reserve']}
    exit_mid = None; exit_i = None

    def stock_trade(change, i, signal_i, liquidation=False):
        nonlocal cash, hedge
        change = int(change)
        if not change: return
        key = 'ask' if change > 0 else 'bid'
        capacity = int(a['stock_'+key+'_size'][i]*costs.stock_quote_size_multiplier)
        if liquidation and abs(change) > capacity: raise ValueError('Insufficient stock liquidation size')
        change = int(np.sign(change)*min(abs(change),capacity))
        if not change: return
        mid = float(a['spot'][i]); quote = mid + (a['stock_'+key][i]-mid)*costs.stock_spread_fraction
        price = quote*(1+np.sign(change)*costs.extra_slippage_bps/10000)
        commission = max(costs.minimum_stock_commission,abs(change)*costs.stock_commission_per_share)*costs.fee_multiplier
        fee = commission + max(-change,0)*price*costs.stock_sell_fee_bps/10000*costs.fee_multiplier
        cash -= change*price+fee; hedge += change
        account['stock_spread_cost'] += abs(change)*abs(quote-mid)
        account['slippage_cost'] += abs(change)*abs(price-quote); account['stock_fees'] += fee
        fills.append({'time':timestamps[i],'decision_time':timestamps[signal_i],'instrument':'stock','quantity':change,
                      'fill_price':price,'fee':fee,'executed':True,'action':'market_order_model'})

    def option_trade(change, i, signal_time):
        nonlocal cash, quantity, abandoned
        key = 'ask' if change > 0 else 'bid'; mid = float(a['option_mid'][i]); price = float(a['option_'+key][i])
        abandon = change < 0 and quantity > 0 and price == 0 and costs.allow_unbid_long_abandonment
        if not abandon:
            if a['option_'+key+'_size'][i] < abs(change): raise ValueError('Insufficient option displayed size')
            price = mid+(price-mid)*costs.option_spread_fraction
        else: abandoned += abs(change)
        fee = 0. if abandon else abs(change)*costs.option_commission_per_contract*costs.fee_multiplier
        cash -= change*100*price+fee; quantity += change
        account['option_spread_cost'] += abs(change)*100*abs(price-mid); account['option_fees'] += fee
        fills.append({'time':timestamps[i],'decision_time':signal_time,'instrument':contract['kind'],'quantity':change,
                      'fill_price':price,'fee':fee,'executed':not abandon,
                      'action':'abandon_to_expiry_no_exercise' if abandon else 'market_order_model'})

    entry_mid = float(a['option_mid'][entry_i])
    for i in range(entry_i,len(p)):
        if i > entry_i:
            interest = cash*(lending_growth[i] if cash >= 0 else debit_growth[i])
            binterest = benchmark*lending_growth[i]
            borrow = max(-hedge,0)*a['spot'][i-1]*costs.short_stock_borrow_rate*years[i]
            dividend = hedge*dividend_per_share[i]
            account['cash_interest'] += interest; account['benchmark_interest'] += binterest; account['borrow_cost'] += borrow
            account['hedge_gross_pnl'] += hedge*(a['spot'][i]-a['spot'][i-1]); account['dividends'] += dividend
            cash += interest-borrow+dividend; benchmark += binterest
        if i == entry_i: option_trade(1,i,decision)
        if timestamps[i] >= planned and pending is None: pending = planned-pd.Timedelta(minutes=15)
        can_close = stock_ok[i] and (option_ok[i] or (
            costs.allow_unbid_long_abandonment and conditions[i] and
            np.isfinite([a['option_bid'][i],a['option_ask'][i]]).all() and
            0 <= a['option_bid'][i] <= a['option_ask'][i] and a['option_ask'][i] > 0 and
            (a['option_bid'][i] == 0 or a['option_bid_size'][i] >= 1)))
        if pending is not None and timestamps[i] > pending and can_close:
            exit_mid = float(a['option_mid'][i]); option_trade(-quantity,i,pending)
            stock_trade(-hedge,i,i,liquidation=True); exit_i = i
        else:
            j = signal_index[i]
            lag_ok = i > 0 and ns[i]-ns[i-1] == grid_minutes*60*1_000_000_000 and j >= 0
            if stock_ok[i] and lag_ok and stock_ok[j] and np.isfinite(a['option_delta'][j]):
                if schedule_kind in ('daily_fixed', 'daily_close'):
                    due = last_hedge_ns is None or (local_dates[i] != last_hedge_day and timestamps[i] >= daily_targets[local_dates[i]])
                else:
                    due = last_hedge_ns is None or clock[i]-last_hedge_ns >= frequency*60*1_000_000_000
                if due:
                    target = int(np.rint(-quantity*100*a['option_delta'][j])); stock_trade(target-hedge,i,j)
                    last_hedge_ns = clock[i]; last_hedge_day = local_dates[i]
        equity = cash+hedge*a['spot'][i]+quantity*100*option_marks[i]
        reserve = max(quantity,0)*100*option_marks[i]+.5*abs(hedge)*a['spot'][i]
        if equity < reserve and pending is None: pending = timestamps[i]; reason = 'research_capital_reserve_close'
        k = i-entry_i
        for key,value in zip(ledger,[equity,cash,benchmark,equity-benchmark,hedge,quantity,
                                     hedge+quantity*100*a['option_delta'][i],reserve]): ledger[key][k] = value
        if exit_i is not None: break
    if exit_i is None: raise ValueError('No observed liquidation before expiry; outcome unresolved')
    path = pd.DataFrame({k:v[:exit_i-entry_i+1] for k,v in ledger.items()})
    path.insert(0,'time',times[entry_i:exit_i+1])
    gross_option = 100*(exit_mid-entry_mid); net = cash-benchmark
    fees = sum(account[k] for k in ['option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees'])
    expected = gross_option+account['hedge_gross_pnl']+account['dividends']+account['cash_interest']-account['benchmark_interest']-account['borrow_cost']-fees
    if abs(net-expected) > 1e-5: raise AssertionError('Single-option cash does not reconcile')
    fills = pd.DataFrame(fills); writeoff = 0.
    for row in fills[(fills.instrument != 'stock') & (~fills.executed)].itertuples():
        writeoff += abs(row.quantity)*100*float(p.loc[row.time,'option_mid'])
    execution = fees-writeoff
    market = gross_option+account['hedge_gross_pnl']+account['dividends']
    financing = account['cash_interest']-account['benchmark_interest']-account['borrow_cost']
    active = path[path.option_contracts != 0]; delta = active.delta_shares.abs().dropna()
    result = {'side':1,'kind':contract['kind'],'schedule_kind':schedule_kind,'daily_time':daily_time,'net_pnl':net,'gross_option_pnl':gross_option,**account,
              'capital':capital,'return_on_capital':net/capital,'return_on_notional':net/(100*initial_spot),
              'entry':str(entry),'exit':str(timestamps[exit_i]),'exit_reason':reason,'abandoned_long_legs':abandoned,
              'market_pnl':market,'execution_cost':execution,'abandonment_writeoff':writeoff,'net_financing':financing,
              'reconciliation_error':net-expected,'cost_parameters':asdict(costs),
              'stock_orders':int((fills.instrument=='stock').sum()),
              'stock_turnover_shares':int(fills.loc[fills.instrument=='stock','quantity'].abs().sum()),
              'mean_abs_delta_shares':float(delta.mean()),'p95_abs_delta_shares':float(delta.quantile(.95)),
              'missing_delta_fraction':float(active.delta_shares.isna().mean()),
              'max_drawdown_usd':float((path.excess_pnl.cummax().clip(lower=0)-path.excess_pnl).max())}
    return result,path,fills
