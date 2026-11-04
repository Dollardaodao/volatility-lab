from dataclasses import replace
import unittest
from test_stock_study import fixture,free_costs
from volatility_lab.stock_replay import replay,StockCosts
from volatility_lab.short_vol_study import trade_diagnostics,account


class ShortVolAccountingTests(unittest.TestCase):
    def test_two_leg_bid_ask_roundtrip_is_40_dollars_plus_4_commission(self):
        p,c,d=fixture()
        for kind in ('call','put'):
            p[kind+'_bid']=4.9;p[kind+'_ask']=5.1;p[kind+'_mid']=5.
        actual=replace(StockCosts(),extra_slippage_bps=0,short_stock_borrow_rate=0,
                       allow_unbid_long_abandonment=True)
        ideal=replace(actual,fee_multiplier=0,option_spread_fraction=0,stock_spread_fraction=0)
        r,_,fills=replay(p,c,d,-1,60,actual)
        diag=trade_diagnostics(p,c,r,-1,fills);bridge=account(r,diag)
        self.assertAlmostEqual(r['net_pnl'],-44.)
        self.assertAlmostEqual(r['option_spread_cost'],40.)
        self.assertAlmostEqual(r['option_fees'],4.)
        self.assertAlmostEqual(bridge['execution_cost'],44.)
        self.assertTrue(diag['vol_bet_correct'])
        ri,_,fi=replay(p,c,d,-1,60,ideal)
        self.assertAlmostEqual(ri['net_pnl'],0.)
        self.assertEqual(list(fills.quantity),list(fi.quantity))
        self.assertEqual(list(fills.time),list(fi.time))

    def test_missing_bid_iv_does_not_discard_a_cash_loss(self):
        p,c,d=fixture()
        for col in ('spot','stock_bid','stock_ask'):p[col]=110.
        r,_,fills=replay(p,c,d,-1,60,replace(free_costs(),option_commission_per_contract=1.))
        before=r['net_pnl'];diag=trade_diagnostics(p,c,r,-1,fills)
        self.assertEqual(diag['vol_bet_status'],'entry_iv_unavailable')
        self.assertEqual(r['net_pnl'],before)
        self.assertAlmostEqual(before,-4.)
        self.assertAlmostEqual(account(r,diag)['accounting_residual'],0.)

    def test_unexecuted_rights_writeoff_is_separate_from_transaction_cost(self):
        p,c,d=fixture()
        p.loc[p.index[-1],['call_bid','call_ask','call_mid','call_bid_size']]=[0.,.02,.01,0]
        p.loc[p.index[-1],['all_valid','options_valid']]=False
        costs=replace(free_costs(),allow_unbid_long_abandonment=True,option_spread_fraction=0,stock_spread_fraction=0)
        r,_,fills=replay(p,c,d,1,60,costs)
        diag=trade_diagnostics(p,c,r,1,fills);bridge=account(r,diag)
        self.assertAlmostEqual(diag['abandonment_writeoff'],1.)
        self.assertAlmostEqual(bridge['execution_cost'],0.)
        self.assertAlmostEqual(bridge['market_pnl'],-99.)
        self.assertAlmostEqual(r['net_pnl'],-100.)


if __name__=='__main__':unittest.main()
