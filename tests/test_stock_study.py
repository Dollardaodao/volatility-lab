from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from volatility_lab.american import AmericanPricer,enrich_american
from volatility_lab.pricing import bs_price_delta
from volatility_lab.stock_replay import replay,StockCosts
from volatility_lab.stock_market import RateInputs,select_contracts,local_time


def fixture():
    times=pd.to_datetime(["2024-11-07T14:45Z","2024-11-07T15:00Z","2024-11-07T15:15Z", "2024-11-07T20:30Z", "2024-11-07T20:45Z", "2024-11-07T21:00Z", "2024-11-08T14:30Z", "2024-11-08T14:45Z", "2024-11-08T15:00Z", "2024-11-08T20:30Z", "2024-11-08T20:45Z"])
    p=pd.DataFrame(index=times)
    for col in ["spot","stock_bid","stock_ask"]:p[col]=100.
    for col in ["stock_bid_size","stock_ask_size"]:p[col]=10
    for kind in ["call","put"]:
        for x in ["bid","ask","mid"]:p[kind+"_"+x]=1.
        for x in ["bid_size","ask_size"]:p[kind+"_"+x]=100
        p[kind+"_conservative_condition_ok"]=True
    p["all_valid"]=True;p["options_valid"]=True;p["stock_passes_snapshot_filters"]=True
    p["delta_per_straddle"]=0.;p["rate"]=.005;p["dividend_yield"]=0.
    c={"decision_time":str(times[0]),"planned_exit":str(times[-1]),"expiry_time":"2024-11-15T21:00Z","strike":100.}
    return p,c,SimpleNamespace(dividends={})


def free_costs():
    return StockCosts(option_commission_per_contract=0,stock_commission_per_share=0,minimum_stock_commission=0,
                      stock_sell_fee_bps=0,extra_slippage_bps=0,short_stock_borrow_rate=0)


class AmericanModelTests(unittest.TestCase):
    def test_missing_leg_uses_observed_same_strike_opposite_leg_iv(self):
        times=pd.to_datetime(["2024-01-02T14:45Z","2024-01-02T15:00Z"])
        p=pd.DataFrame(index=times);p["spot"]=100.;p["years_to_expiry"]=[30/365,(30*86400-900)/(365*86400)]
        for kind in ["call","put"]:
            values=[]
            for i,tau in enumerate(p.years_to_expiry):
                pricer=AmericanPricer(100,kind);pricer.configure(100,tau,.04,0);values.append(pricer.value(.3 if i==0 else .5))
            p[kind+"_mid"]=values;p[kind+"_passes_snapshot_filters"]=True
        p.loc[times[1],"call_passes_snapshot_filters"]=False
        result=enrich_american(p,100,SimpleNamespace(yield_at=lambda t,s:0),SimpleNamespace(at=lambda t:.04))
        self.assertTrue(result.iv_cross_leg.iloc[1])
        self.assertAlmostEqual(result.call_iv.iloc[1],result.put_iv.iloc[1],places=8)
        self.assertAlmostEqual(result.call_iv.iloc[1],.5,places=4)

    def test_non_dividend_american_call_matches_european_with_fractional_time(self):
        for years in [7.2604/365,32.111/365,101.21875/365]:
            p=AmericanPricer(100,"call");p.configure(102,years,.04,0)
            expected,delta=bs_price_delta(102,100,years,.3,.04,"call")
            self.assertAlmostEqual(p.value(.3),expected,places=7)
            self.assertAlmostEqual(p.delta(.3),delta,places=7)

    def test_american_put_reference_and_iv_roundtrip(self):
        p=AmericanPricer(40,"put");p.configure(36,1.,.06,0)
        self.assertAlmostEqual(p.value(.2),4.4531,places=3)
        self.assertAlmostEqual(p.implied(p.value(.35)),.35,places=4)


class StockLedgerTests(unittest.TestCase):
    def test_midpoint_counterfactual_removes_spreads_without_changing_orders(self):
        p,c,d=fixture()
        for k in ["call","put"]:p[k+"_bid"]=.9;p[k+"_ask"]=1.1
        p["stock_bid"]=99.;p["stock_ask"]=101.;p["delta_per_straddle"]=.5
        costs=replace(free_costs(),option_spread_fraction=0,stock_spread_fraction=0)
        result,_,fills=replay(p,c,d,1,60,costs)
        self.assertAlmostEqual(result["net_pnl"],0.,places=7)
        self.assertAlmostEqual(result["option_spread_cost"],0.,places=7)
        self.assertAlmostEqual(result["stock_spread_cost"],0.,places=7)
        self.assertEqual(len(fills),6)

    def test_midpoint_counterfactual_does_not_invent_zero_bid_sale(self):
        p,c,d=fixture()
        p.loc[p.index[-1],["call_bid","call_ask","call_mid","call_bid_size"]]=[0.,.02,.01,0]
        p.loc[p.index[-1],["all_valid","options_valid"]]=False
        costs=replace(free_costs(),option_spread_fraction=0,stock_spread_fraction=0,allow_unbid_long_abandonment=True)
        result,_,fills=replay(p,c,d,1,60,costs)
        abandon=fills[fills.action=="abandon_to_expiry_no_exercise"].iloc[0]
        self.assertFalse(abandon.executed)
        self.assertEqual(abandon.fill_price,0.)
        self.assertAlmostEqual(result["net_pnl"],-100.,places=7)

    def test_short_leg_with_zero_bid_closes_at_real_positive_ask(self):
        p,c,d=fixture()
        p.loc[p.index[-1],["call_bid","call_ask","call_mid","call_bid_size"]]=[0.,.02,.01,0]
        p.loc[p.index[-1],["all_valid","options_valid"]]=False
        result,_,fills=replay(p,c,d,-1,15,replace(free_costs(),allow_unbid_long_abandonment=True))
        self.assertEqual(result["abandoned_long_legs"],0)
        self.assertAlmostEqual(result["net_pnl"],98.,places=7)
        call=fills[(fills.instrument=="call")&(fills.quantity>0)].iloc[-1]
        self.assertTrue(call.executed);self.assertEqual(call.fill_price,.02)

    def test_zero_bid_long_abandonment_is_not_logged_as_an_execution(self):
        p,c,d=fixture()
        p.loc[p.index[-1],["call_bid","call_ask","call_mid","call_bid_size"]]=[0.,.02,.01,0]
        p.loc[p.index[-1],["all_valid","options_valid"]]=False
        result,_,fills=replay(p,c,d,1,15,replace(free_costs(),allow_unbid_long_abandonment=True))
        abandon=fills[fills.action=="abandon_to_expiry_no_exercise"].iloc[0]
        self.assertFalse(abandon.executed)
        self.assertEqual(result["abandoned_long_legs"],1)
        self.assertAlmostEqual(result["net_pnl"],-100.,places=7)
        self.assertLess(abs(result["reconciliation_error"]),1e-5)

    def test_bid_ask_roundtrip_costs_both_legs_and_both_directions(self):
        p,c,d=fixture()
        for k in ["call","put"]:p[k+"_bid"]=.9;p[k+"_ask"]=1.1
        for side in [-1,1]:
            result,_,_=replay(p,c,d,side,15,free_costs())
            self.assertAlmostEqual(result["net_pnl"],-40.,places=7)
            self.assertAlmostEqual(result["option_spread_cost"],40.,places=7)

    def test_dividend_obligation_offsets_short_hedge_ex_dividend_gain(self):
        p,c,d=fixture();p["delta_per_straddle"]=.5
        for col in ["spot","stock_bid","stock_ask"]:p.loc[p.index>=pd.Timestamp("2024-11-08T00:00Z"),col]=99.
        d.dividends={pd.Timestamp("2024-11-08").date():1.}
        result,_,_=replay(p,c,d,1,60,free_costs())
        self.assertAlmostEqual(result["hedge_gross_pnl"],50.)
        self.assertAlmostEqual(result["dividends"],-50.)
        self.assertAlmostEqual(result["net_pnl"],0.,places=7)

    def test_orders_use_previous_observation_and_whole_shares(self):
        p,c,d=fixture();p.loc[p.index[1]:,"delta_per_straddle"]=.1234
        result,ledger,fills=replay(p,c,d,1,15,free_costs())
        first=fills[fills.instrument=="stock"].iloc[0]
        self.assertEqual(first.time,p.index[2]);self.assertEqual(first.decision_time,p.index[1])
        self.assertEqual(first.quantity,-12)
        self.assertEqual(ledger.iloc[-1].stock_shares,0)
        self.assertLess(abs(result["reconciliation_error"]),1e-5)

    def test_missing_liquidation_quote_does_not_create_a_model_fill(self):
        p,c,d=fixture();p.loc[p.index[-1],"all_valid"]=False
        with self.assertRaises(ValueError):replay(p,c,d,1,15,free_costs())

    def test_costs_and_variable_cash_benchmark_reconcile(self):
        p,c,d=fixture();p["delta_per_straddle"]=.4
        p["spot"]=np.linspace(100,105,len(p));p["stock_bid"]=p.spot-.01;p["stock_ask"]=p.spot+.01
        p["rate"]=np.linspace(.04,.05,len(p))
        result,_,fills=replay(p,c,d,1,15,StockCosts())
        self.assertGreater(result["stock_fees"],0);self.assertGreater(result["borrow_cost"],0)
        self.assertLess(abs(result["reconciliation_error"]),1e-5)
        self.assertTrue((fills.time>fills.decision_time).all())


class PointInTimeTests(unittest.TestCase):
    def test_rate_publication_lag_does_not_use_a_future_observation(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"rates.json"
            path.write_text(json.dumps({"refRates":[{"effectiveDate":"2024-11-01","percentRate":4},{"effectiveDate":"2024-11-04","percentRate":99}]}))
            r=RateInputs(path)
            self.assertEqual(r.at(pd.Timestamp("2024-11-05T15:00Z")),.04)

    def test_future_quote_does_not_change_contract_selection(self):
        decision=local_time("2024-11-04","09:45")
        stock=pd.DataFrame({"bid":[99.9],"ask":[100.1],"passes_snapshot_filters":[True]},index=[decision])
        rows=[]
        for strike,time in [(101.,decision),(100.,decision+pd.Timedelta(minutes=15))]:
            for kind in ["CALL","PUT"]:rows.append({"sample_time_utc":time,"expiration":"2024-11-08","strike":strike,"right":kind})
        c=select_contracts(pd.DataFrame(rows),stock,"2024-11-04",[7],[5])[0]
        self.assertEqual(c["strike"],101.)


if __name__=="__main__":unittest.main()
