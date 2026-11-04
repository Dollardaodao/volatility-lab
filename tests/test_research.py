import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import pandas as pd
from volatility_lab.forecast import Fit, fit_variance, expiry_forecast, future_closes, MODELS
from volatility_lab.pricing import bs_price_delta, implied_vol_delta
from volatility_lab.backtest import Costs, run_pilot
from volatility_lab.theta import normalize_quotes,ThetaClient


def panel():
    times = pd.date_range("2024-03-01T00:05Z",periods=8,freq="5min").as_unit("ns")
    frame = pd.DataFrame(index=times)
    frame["spot"] = 100.
    frame["straddle_delta"] = 0.
    for kind in ("call","put"):
        frame[f"{kind}_bid_price"] = .009
        frame[f"{kind}_ask_price"] = .011
        frame[f"{kind}_mid_coin"] = .01
        frame[f"{kind}_bid_amount"] = 10.
        frame[f"{kind}_ask_amount"] = 10.
        frame[f"{kind}_timestamp"] = times-pd.Timedelta(seconds=1)
        frame[f"{kind}_available_at"] = times-pd.Timedelta(milliseconds=500)
        frame[f"{kind}_expiration"] = pd.Timestamp("2024-03-29T08:00Z")
        frame[f"{kind}_strike_price"] = 100.
        frame[f"{kind}_symbol"] = kind
    frame["straddle_mid_usd"] = .02*frame.spot
    return frame


def zero_costs(**overrides):
    base = Costs(spot_half_spread_bps=0,spot_slippage_bps=0,spot_fee_bps=0,
                 option_notional_fee_bps=0,cash_lend_rate=0,cash_debit_rate=0,
                 short_asset_borrow_rate=0)
    return replace(base,**overrides)


class ForecastTests(unittest.TestCase):
    def test_variances_are_summed_across_expiry_not_terminal_day(self):
        origin = pd.Timestamp("2024-01-01T00:00Z")
        fit = Fit("fixed",origin,np.array([1.,4.,9.]),{})
        result = expiry_forecast(fit,origin,origin+pd.Timedelta(days=2),"crypto")
        self.assertAlmostEqual(result["integrated_variance"],5.)
        self.assertAlmostEqual(result["annualized_vol"]**2,5*365/2)

    def test_partial_daily_interval(self):
        origin = pd.Timestamp("2024-01-01T00:00Z")
        fit = Fit("fixed",origin,np.array([.01,.04]),{})
        result = expiry_forecast(fit,origin+pd.Timedelta(hours=6),origin+pd.Timedelta(hours=18),"crypto")
        self.assertAlmostEqual(result["integrated_variance"],.005)

    def test_holiday_and_early_close(self):
        closes = future_closes(pd.Timestamp("2024-07-02T20:00Z"),2,"equity")
        self.assertEqual(closes[0],pd.Timestamp("2024-07-03T17:00Z"))
        self.assertEqual(closes[1],pd.Timestamp("2024-07-05T20:00Z"))

    def test_future_price_changes_do_not_change_any_forecast(self):
        rng = np.random.default_rng(42)
        times = pd.date_range("2020-01-01",periods=650,tz="UTC").as_unit("ns")
        prices = pd.Series(100*np.exp(np.cumsum(rng.normal(0,.015,650))),index=times)
        corrupted = prices.copy()
        corrupted.iloc[601:] *= np.linspace(100,10000,49)
        for model in MODELS:
            a = fit_variance(prices,times[600],horizon=10,model=model)
            b = fit_variance(corrupted,times[600],horizon=10,model=model)
            np.testing.assert_allclose(a.daily_variance,b.daily_variance,atol=0,rtol=0)
            self.assertEqual(a.trained_through,times[600])

    def test_timestamp_units_do_not_change_aggregation(self):
        origin = pd.Timestamp("2024-01-01T00:00Z").as_unit("us")
        f = Fit("fixed",origin,np.array([.01,.02]),{})
        result = expiry_forecast(f,origin,origin+pd.Timedelta(days=1),"crypto")
        self.assertAlmostEqual(result["integrated_variance"],.01)


class PricingTests(unittest.TestCase):
    def test_implied_vol_roundtrip_and_put_call_delta(self):
        for kind in ("call","put"):
            price,delta = bs_price_delta(100,105,.25,.4,.04,kind)
            vol,recovered = implied_vol_delta(price,100,105,.25,.04,kind)
            self.assertAlmostEqual(vol,.4,places=9)
            self.assertAlmostEqual(delta,recovered,places=9)
        dc=bs_price_delta(100,105,.25,.4,.04,"call")[1]
        dp=bs_price_delta(100,105,.25,.4,.04,"put")[1]
        self.assertAlmostEqual(dc-dp,1.)


class DataTests(unittest.TestCase):
    def test_cached_quotes_with_mixed_dst_offsets_normalize_to_utc(self):
        frame=pd.DataFrame({"timestamp":["2024-03-08 09:45:00-05:00",pd.Timestamp("2024-03-11 09:45",tz="America/New_York")],
                            "bid":[100,100],"ask":[101,101],"bid_size":[1,1],"ask_size":[1,1],
                            "bid_condition":[0,0],"ask_condition":[0,0]})
        result=normalize_quotes(frame)
        self.assertEqual(result.sample_time_utc.iloc[0],pd.Timestamp("2024-03-08T14:45Z"))
        self.assertEqual(result.sample_time_utc.iloc[1],pd.Timestamp("2024-03-11T13:45Z"))

    def test_resampled_quotes_after_an_early_close_are_flagged(self):
        frame=pd.DataFrame({"timestamp":["2024-11-29T12:45:00","2024-11-29T13:15:00"],
                            "bid":[100,100],"ask":[101,101],"bid_size":[1,1],"ask_size":[1,1],
                            "bid_condition":[0,0],"ask_condition":[0,0]})
        result=normalize_quotes(frame)
        self.assertEqual(result.regular_session.tolist(),[True,False])
        self.assertEqual(result.passes_snapshot_filters.tolist(),[True,False])

    def test_sdk_does_not_attempt_authentication_without_a_key(self):
        from volatility_lab.theta_sdk import connect
        with tempfile.TemporaryDirectory() as directory:
            keyfile=Path(directory)/".env"
            keyfile.write_text("THETADATA_API_KEY=\n")
            with patch.dict("os.environ",{},clear=True),patch("thetadata.ThetaClient") as client:
                with self.assertRaises(ValueError):connect(keyfile)
                client.assert_not_called()

    def test_quote_timestamp_conversion_respects_daylight_saving(self):
        frame=pd.DataFrame({"timestamp":["2024-11-01T09:30:00","2024-11-04T09:30:00"],
                            "bid":[100,100],"ask":[101,99],"bid_size":[1,1],"ask_size":[1,1]})
        result=normalize_quotes(frame)
        self.assertEqual(result.sample_time_utc.iloc[0],pd.Timestamp("2024-11-01T13:30Z"))
        self.assertEqual(result.sample_time_utc.iloc[1],pd.Timestamp("2024-11-04T14:30Z"))
        self.assertEqual(result.valid_two_sided.tolist(),[True,False])
        self.assertFalse(result.original_quote_time_verified.any())

    def test_theta_adapter_rejects_remote_authentication_destinations(self):
        with self.assertRaises(ValueError):ThetaClient("https://example.com/v3")


class AccountingTests(unittest.TestCase):
    def test_option_roundtrip_pays_both_sides_of_both_legs(self):
        for side in (-1,1):
            result,_,_=run_pilot(panel(),side,zero_costs())
            self.assertAlmostEqual(result["net_excess_pnl_usd"],-.04,places=8)
            self.assertAlmostEqual(result["option_spread_cost"],.04)

    def test_coin_premium_marks_and_cashflows_convert_to_usd(self):
        f = panel()
        f["spot"] = np.linspace(100,110,len(f))
        for k in ("call","put"):
            f[f"{k}_bid_price"] = .01
            f[f"{k}_ask_price"] = .01
        f["straddle_mid_usd"] = .02*f.spot
        result,_,_=run_pilot(f,1,zero_costs())
        expected=.1*.02*(f.spot.iloc[-1]-f.spot.iloc[1])
        self.assertAlmostEqual(result["net_excess_pnl_usd"],expected,places=8)

    def test_flat_policy_equals_idle_cash(self):
        result,_,trades=run_pilot(panel(),0,Costs())
        self.assertAlmostEqual(result["net_excess_pnl_usd"],0.,places=8)
        self.assertEqual(len(trades),0)

    def test_hedge_targets_use_previous_observation(self):
        f=panel()
        f.loc[f.index[1]:,"straddle_delta"]=1.
        _,_,trades=run_pilot(f,1,zero_costs(hedge_every_minutes=5))
        hedge=trades[trades.instrument=="spot_proxy"].iloc[0]
        self.assertEqual(hedge.time,f.index[2])
        self.assertEqual(hedge.decision_time,f.index[1])
        self.assertAlmostEqual(hedge.units,-.1)

    def test_financing_spreads_borrow_and_fees_reconcile(self):
        f=panel()
        f["spot"]=np.linspace(100,110,len(f))
        f["straddle_mid_usd"] = .02*f.spot
        f["straddle_delta"] = .6
        result,ledger,trades=run_pilot(f,1,Costs(hedge_every_minutes=5))
        self.assertLess(abs(result["reconciliation_error_usd"]),1e-6)
        self.assertGreater(result["borrow_cost"],0)
        self.assertGreater(result["fx_spread_cost"],0)
        self.assertEqual(ledger.iloc[-1].spot_units,0)
        self.assertEqual(ledger.iloc[-1].option_units,0)
        self.assertTrue((trades.time>trades.decision_time).all())

    def test_reject_missing_crossed_future_and_oversize_quotes(self):
        for problem in ("missing","crossed","future","size","gap"):
            f=panel()
            if problem=="missing":f.loc[f.index[2],"call_bid_price"]=np.nan
            elif problem=="crossed":f.loc[f.index[2],"call_bid_price"]=1.
            elif problem=="future":f.loc[f.index[2],"call_available_at"]=f.index[3]
            elif problem=="size":f["call_ask_amount"]=.001
            elif problem=="gap":f=f.drop(f.index[3])
            with self.subTest(problem=problem),self.assertRaises(ValueError):
                run_pilot(f,1,Costs())


if __name__=="__main__":
    unittest.main()
