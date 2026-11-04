import unittest
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
from volatility_lab.highfreq_replay import replay as reference
from volatility_lab.single_leg_replay import replay
from volatility_lab.stock_replay import StockCosts
from volatility_lab.monthly_single_leg_acquire import select_candidates
from volatility_lab.forecast import Fit
from volatility_lab.american import AmericanPricer
from volatility_lab.single_leg_counterfactual import midpoint_book


def fixture():
    times = pd.date_range('2024-11-07 14:45Z', '2024-11-07 15:45Z', freq='min')
    p = pd.DataFrame(index=times); n = len(p)
    p['spot'] = 100 + .4*np.sin(np.arange(n)/3)
    p['stock_bid'] = p.spot-.01; p['stock_ask'] = p.spot+.01
    p['stock_bid_size'] = p['stock_ask_size'] = 10
    p['option_mid'] = 2 + .2*np.sin(np.arange(n)/4)
    p['option_bid'] = p.option_mid-.03; p['option_ask'] = p.option_mid+.03
    p['option_bid_size'] = p['option_ask_size'] = 10
    p['option_delta'] = .5+.1*np.sin(np.arange(n)/3)
    p['stock_passes_snapshot_filters'] = p['option_passes_snapshot_filters'] = p['option_conservative_condition_ok'] = True
    p['rate'] = .04
    c = {'kind':'call','decision_time':str(times[0]),'planned_exit':str(times[-1]),'expiry_time':str(times[-1]+pd.Timedelta(days=1))}
    return p,c,SimpleNamespace(dividends={})


class SingleOptionTests(unittest.TestCase):
    def test_cash_counterfactual_matches_full_replay_with_lending_and_borrowing(self):
        for capital,kind,delta in [(30000.,'call',.5),(100.,'put',-.4)]:
            p,c,d=fixture();c['kind']=kind;p['option_delta']=delta
            costs=StockCosts();r,ledger,fills=replay(p,c,d,1,costs,capital=capital)
            ideal,_,book=midpoint_book(p,r,ledger,fills,costs)
            free=replace(costs,fee_multiplier=0,option_spread_fraction=0,stock_spread_fraction=0,extra_slippage_bps=0)
            same_exit={**c,'planned_exit':r['exit']}
            ref,_,orders=replay(p,same_exit,d,1,free,capital=capital)
            self.assertAlmostEqual(ideal['net_pnl'],ref['net_pnl'],places=7)
            self.assertAlmostEqual(ideal['cash_interest'],ref['cash_interest'],places=7)
            self.assertEqual(book[['time','instrument','quantity']].to_dict('records'),orders[['time','instrument','quantity']].to_dict('records'))

    def test_array_engine_matches_independent_existing_cash_engine(self):
        p,c,d = fixture(); old = p.copy()
        for field in ['bid','ask','mid','bid_size','ask_size']:
            old['call_'+field] = p['option_'+field]
            old['put_'+field] = 10 if 'size' in field else 0.
        old['all_valid'] = old['options_valid'] = True
        old['call_conservative_condition_ok'] = old['put_conservative_condition_ok'] = True
        old['delta_per_straddle'] = p.option_delta
        old['dividend_yield'] = 0.
        for frequency in [1,5,15,60]:
            costs = StockCosts()
            actual,_,fills = replay(p,c,d,frequency,costs)
            # A zero-price/zero-Delta extra bookkeeping leg permits an
            # independent old-engine check; half commissions per leg give
            # the same single-contract total fees. It is not a trade signal.
            expected,_,refills = reference(old,c,d,1,frequency,replace(costs,option_commission_per_contract=.5),
                                          grid_minutes=1,signal_lag_minutes=1,allow_exercise_close=False)
            self.assertAlmostEqual(actual['net_pnl'],expected['net_pnl'],places=7)
            self.assertEqual(fills[fills.instrument=='stock'][['time','quantity']].to_dict('records'),
                             refills[refills.instrument=='stock'][['time','quantity']].to_dict('records'))
            self.assertEqual(len(fills[fills.instrument!='stock']),2)

    def test_put_hedge_has_opposite_sign_and_uses_past_delta(self):
        p,c,d = fixture(); c['kind']='put'; p['option_delta']=-.4
        _,_,fills = replay(p,c,d,1)
        first = fills[fills.instrument=='stock'].iloc[0]
        self.assertEqual(first.quantity,40)
        self.assertEqual(first.time-first.decision_time,pd.Timedelta(minutes=1))
        self.assertEqual(set(fills[fills.instrument!='stock'].instrument),{'put'})

    def test_zero_bid_right_is_written_off_even_in_midpoint_book(self):
        p,c,d = fixture(); p['option_delta']=0.; p['option_mid']=1.; p['option_bid']=.9;p['option_ask']=1.1
        p.loc[p.index[-1],['option_bid','option_ask','option_mid','option_bid_size']]=[0.,.02,.01,0]
        p.loc[p.index[-1],'option_passes_snapshot_filters']=False
        costs=replace(StockCosts(),option_spread_fraction=0,stock_spread_fraction=0,fee_multiplier=0,
                      extra_slippage_bps=0,allow_unbid_long_abandonment=True)
        r,_,fills=replay(p,c,d,1,costs)
        end=fills[fills.instrument!='stock'].iloc[-1]
        self.assertFalse(end.executed);self.assertEqual(end.fill_price,0.)
        self.assertAlmostEqual(r['execution_cost'],0.,places=8)
        self.assertAlmostEqual(r['abandonment_writeoff'],1.,places=8)
        self.assertLess(abs(r['reconciliation_error']),1e-5)

    def test_invalid_past_stock_cannot_generate_a_signal(self):
        p,c,d=fixture();p['option_delta']=0.
        at=pd.Timestamp('2024-11-07 15:01Z');p.loc[p.index>=at,'option_delta']=.2
        p.loc[at,'stock_passes_snapshot_filters']=False
        _,_,fills=replay(p,c,d,1)
        first=fills[fills.instrument=='stock'].iloc[0]
        self.assertEqual(first.time,pd.Timestamp('2024-11-07 15:03Z'))


class SingleSelectionTests(unittest.TestCase):
    def inputs(self):
        decision=pd.Timestamp('2024-11-07 14:45Z');expiry=pd.Timestamp('2024-11-15 21:00Z');rows=[]
        for kind,vol in [('call',.2),('put',.3)]:
            pricer=AmericanPricer(100,kind);pricer.configure(100,(expiry-decision).total_seconds()/(365*86400),.04,0)
            ask=pricer.value(vol)
            rows.append({'sample_time_utc':decision,'expiration':'2024-11-15','right':kind.upper(),
                         'strike':100.,'bid':ask-.01,'ask':ask,'passes_snapshot_filters':True})
        fit=Fit('garch_t',pd.Timestamp('2024-11-06 21:00Z'),np.full(150,.4**2/252),{})
        daily=SimpleNamespace(splits={},yield_at=lambda t,s:0.)
        rates=SimpleNamespace(at=lambda t:.04)
        plan={'model':'garch_t','target_calendar_dte':[7],'dte_tolerance_days':[5],'signal_vol_buffer':.03}
        return pd.DataFrame(rows),fit,daily,rates,plan

    def test_selects_one_larger_gap_and_ignores_future_snapshot(self):
        snapshot,fit,daily,rates,plan=self.inputs()
        future=snapshot.iloc[[1]].copy();future['sample_time_utc']=pd.Timestamp('2024-11-07 15:00Z');future['ask']=.01
        rows=select_candidates(pd.concat([snapshot,future]),100.,'2024-11-07',daily,rates,plan,fit)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['kind'],'call')
        self.assertAlmostEqual(rows[0]['decision_ask_iv'],.2,places=4)
        self.assertEqual(rows[0]['side'],1)

    def test_one_available_leg_does_not_require_a_paired_straddle(self):
        snapshot,fit,daily,rates,plan=self.inputs()
        snapshot.loc[snapshot.right=='CALL','passes_snapshot_filters']=False
        row=select_candidates(snapshot,100.,'2024-11-07',daily,rates,plan,fit)[0]
        self.assertEqual(row['kind'],'put');self.assertEqual(row['side'],1)
        self.assertEqual(len(row['candidates']),1)


if __name__=='__main__':unittest.main()
