import unittest
import numpy as np
import pandas as pd
from test_stock_study import fixture,free_costs
from volatility_lab.stock_replay import replay as original,StockCosts
from volatility_lab.highfreq_replay import replay


def minute_fixture():
    p,c,d=fixture()
    times=pd.date_range('2024-11-07 14:45Z','2024-11-07 15:10Z',freq='min')
    p=pd.DataFrame([p.iloc[0].to_dict() for _ in times],index=times)
    c={**c,'decision_time':str(times[0]),'planned_exit':str(times[-1])}
    return p,c,d


class MinuteHedgeTests(unittest.TestCase):
    def test_fifteen_minute_default_reproduces_original_cash_and_orders(self):
        p,c,d=fixture();p['delta_per_straddle']=.12;p.loc[p.index[2]:,'delta_per_straddle']=.28
        p['stock_bid']=99.99;p['stock_ask']=100.01
        for k in ('call','put'):p[k+'_bid']=.9;p[k+'_ask']=1.1
        for side in (-1,1):
            a,_,fa=original(p,c,d,side,60,StockCosts())
            b,_,fb=replay(p,c,d,side,60,StockCosts())
            self.assertAlmostEqual(a['net_pnl'],b['net_pnl'],places=8)
            self.assertEqual(fa[['time','instrument','quantity']].to_dict('records'),fb[['time','instrument','quantity']].to_dict('records'))

    def test_minute_hedge_uses_past_minute_delta_and_keeps_delay_separate(self):
        p,c,d=minute_fixture();p.loc[p.index>=pd.Timestamp('2024-11-07 15:00Z'),'delta_per_straddle']=.2
        r,_,fills=replay(p,c,d,-1,1,free_costs(),grid_minutes=1,signal_lag_minutes=1,allow_exercise_close=False)
        stock=fills[fills.instrument=='stock'];first=stock.iloc[0]
        self.assertEqual(first.time,pd.Timestamp('2024-11-07 15:01Z'))
        self.assertEqual(first.decision_time,pd.Timestamp('2024-11-07 15:00Z'))
        self.assertEqual(first.quantity,20)
        _,_,slow=replay(p,c,d,-1,1,free_costs(),grid_minutes=1,signal_lag_minutes=15,allow_exercise_close=False)
        self.assertEqual(len(slow[slow.instrument=='stock']),0)
        self.assertLess(abs(r['reconciliation_error']),1e-5)

    def test_invalid_past_stock_quote_cannot_generate_a_hedge(self):
        p,c,d=minute_fixture()
        change=pd.Timestamp('2024-11-07 15:01Z')
        p.loc[p.index>=change,'delta_per_straddle']=.2
        p.loc[change,['stock_passes_snapshot_filters','all_valid']]=False
        _,_,fills=replay(p,c,d,-1,1,free_costs(),grid_minutes=1,signal_lag_minutes=1,allow_exercise_close=False)
        first=fills[fills.instrument=='stock'].iloc[0]
        self.assertEqual(first.time,pd.Timestamp('2024-11-07 15:03Z'))
        self.assertEqual(first.decision_time,pd.Timestamp('2024-11-07 15:02Z'))


if __name__=='__main__':unittest.main()
