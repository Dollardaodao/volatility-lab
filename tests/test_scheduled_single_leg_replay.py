import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from volatility_lab.single_leg_replay import replay as original
from volatility_lab.scheduled_single_leg_replay import replay
from volatility_lab.stock_market import sessions,local_time
from volatility_lab.stock_replay import StockCosts


def fixture(first='2024-11-06',last='2024-11-08'):
    calendar=sessions(first,last)
    times=pd.DatetimeIndex([t for _,r in calendar.iterrows() for t in pd.date_range(r['open'],r['close'],freq='min')]).as_unit('ns')
    p=pd.DataFrame(index=times)
    p['spot']=100.;p['stock_bid']=99.99;p['stock_ask']=100.01
    p['stock_bid_size']=p['stock_ask_size']=10
    p['option_mid']=1.;p['option_bid']=.99;p['option_ask']=1.01
    p['option_bid_size']=p['option_ask_size']=10
    p['option_delta']=.1+np.arange(len(p))*.0003
    p['stock_passes_snapshot_filters']=p['option_passes_snapshot_filters']=p['option_conservative_condition_ok']=True
    p['rate']=.04
    c={'decision_time':str(local_time(first,'09:45')),'planned_exit':str(calendar.iloc[-1]['close']-pd.Timedelta(minutes=1)),
       'expiry_time':str(calendar.iloc[-1]['close']+pd.Timedelta(days=7)),'kind':'call'}
    return p,c,SimpleNamespace(dividends={})


def adjustments(fills,result):
    return fills[(fills.instrument=='stock')&(pd.to_datetime(fills.time,utc=True)!=pd.Timestamp(result['exit']))]


class ScheduleTests(unittest.TestCase):
    def test_elapsed_defaults_reproduce_original_orders_and_cash(self):
        p,c,d=fixture()
        for frequency in [1,60,120,720]:
            a,_,fa=original(p,c,d,frequency,StockCosts())
            b,_,fb=replay(p,c,d,frequency,StockCosts())
            self.assertAlmostEqual(a['net_pnl'],b['net_pnl'],places=8)
            self.assertEqual(fa.to_dict('records'),fb.to_dict('records'))

    def test_twelve_calendar_hours_becomes_next_open_after_initial_hedge(self):
        p,c,d=fixture();r,_,fills=replay(p,c,d,720)
        times=adjustments(fills,r).time.dt.tz_convert('America/New_York')
        self.assertEqual(times.dt.strftime('%H:%M').tolist(),['10:00','09:31','09:31'])

    def test_daily_clock_does_not_follow_next_open_and_is_once_per_day(self):
        p,c,d=fixture();r,_,fills=replay(p,c,d,1440,schedule_kind='daily_fixed')
        stock=adjustments(fills,r);times=stock.time.dt.tz_convert('America/New_York')
        self.assertEqual(times.dt.strftime('%H:%M').tolist(),['10:00','10:00','10:00'])
        self.assertEqual(len(set(times.dt.date)),len(times))

    def test_trading_clock_excludes_overnights_and_respects_early_close(self):
        p,c,d=fixture();r,_,fills=replay(p,c,d,720,schedule_kind='trading_time')
        times=adjustments(fills,r).time.dt.tz_convert('America/New_York')
        self.assertEqual(times.dt.strftime('%Y-%m-%d %H:%M').tolist(),['2024-11-06 10:00','2024-11-07 15:30'])
        p,c,d=fixture('2024-11-27','2024-11-29');r,_,fills=replay(p,c,d,720,schedule_kind='trading_time')
        self.assertEqual(len(adjustments(fills,r)),1)

    def test_daily_close_uses_exchange_calendar_including_early_close(self):
        p,c,d=fixture('2024-11-27','2024-11-29');r,_,fills=replay(p,c,d,1440,schedule_kind='daily_close')
        times=adjustments(fills,r).time.dt.tz_convert('America/New_York')
        self.assertEqual(times.dt.strftime('%Y-%m-%d %H:%M').tolist(),['2024-11-27 10:00','2024-11-29 12:45'])

    def test_invalid_daily_target_waits_for_valid_past_and_current_quotes(self):
        p,c,d=fixture();p.loc[local_time('2024-11-07','10:00'),'stock_passes_snapshot_filters']=False
        r,_,fills=replay(p,c,d,1440,schedule_kind='daily_fixed')
        stock=adjustments(fills,r);times=stock.time.dt.tz_convert('America/New_York')
        self.assertEqual(times.dt.strftime('%H:%M').tolist(),['10:00','10:02','10:00'])
        self.assertTrue(((pd.to_datetime(stock.time,utc=True)-pd.to_datetime(stock.decision_time,utc=True))==pd.Timedelta(minutes=1)).all())


if __name__=='__main__':unittest.main()
