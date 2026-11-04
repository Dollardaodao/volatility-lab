from dataclasses import replace
import unittest
from test_stock_study import fixture,free_costs
from volatility_lab.stock_replay import replay,StockCosts
try:
    from volatility_lab.short_loss_diagnostics import audit_fills,interval_attribution
except ImportError as exc:
    raise unittest.SkipTest('Optional ML diagnostic dependencies are unavailable') from exc


class LossAttributionTests(unittest.TestCase):
    def test_fill_audit_separates_stock_quote_slippage_commission_and_cash(self):
        p,c,d=fixture();p['delta_per_straddle']=.07
        p['stock_bid']=99.;p['stock_ask']=101.
        costs=StockCosts(allow_unbid_long_abandonment=True)
        r,_,fills=replay(p,c,d,-1,60,costs)
        audit,summary=audit_fills(p,fills,r,costs)
        buy=audit[(audit.instrument=='stock')&(audit.quantity>0)].iloc[0]
        self.assertEqual(buy.quantity,7)
        self.assertAlmostEqual(buy.fill_price,101.0101)
        self.assertAlmostEqual(buy.spread_cost,7.)
        self.assertAlmostEqual(buy.added_slippage,.0707)
        self.assertAlmostEqual(buy.commission,.35)
        self.assertAlmostEqual(buy.cash_flow,-707.4207)
        self.assertAlmostEqual(summary['cash_residual'],0.,places=7)
        self.assertAlmostEqual(audit.total_fee.sum(),r['stock_fees']+r['option_fees'])

    def test_interval_stock_pnl_uses_previously_held_shares(self):
        p,c,d=fixture();p['delta_per_straddle']=.07
        p.loc[p.index[1]:,'delta_per_straddle']=.4
        p.loc[p.index[2],'spot']=102.
        p['stock_bid']=p.spot;p['stock_ask']=p.spot
        r,ledger,_=replay(p,c,d,-1,60,free_costs())
        r['market_pnl']=r['gross_option_pnl']+r['hedge_gross_pnl']+r['dividends']
        attribution=interval_attribution(p,c,ledger,d,r)
        self.assertEqual(attribution.shares_before_interval.iloc[1],7)
        self.assertAlmostEqual(attribution.hedge_pnl.iloc[1],14.)
        self.assertAlmostEqual(attribution.hedge_pnl.sum(),r['hedge_gross_pnl'])
        self.assertAlmostEqual(attribution.market_pnl.sum(),r['market_pnl'])


if __name__=='__main__':unittest.main()
