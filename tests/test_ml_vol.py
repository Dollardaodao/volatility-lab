import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
try:
    from volatility_lab.ml_vol import StockDataset,fit_models,PooledVarianceModel
except ImportError as exc:
    raise unittest.SkipTest('Optional ML dependencies are not installed') from exc
from volatility_lab.forecast import exchange_closes


class ConstantBlocks:
    def predict(self,x):return np.arange(1,len(x)+1,dtype=float)


class MLTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.index=exchange_closes('2018-01-01','2019-03-01')[:230]
        rng=np.random.default_rng(7317)
        close=100*np.exp(np.cumsum(rng.normal(0,.015,len(self.index))))
        dates=[pd.Timestamp(t.date()).tz_localize('America/New_York') for t in self.index]
        self.frame=pd.DataFrame({'Date':dates,'Open':close*.999,'High':close*1.01,'Low':close*.99,
            'Close':close,'Volume':rng.integers(1000,10000,len(close)),
            'Dividends':0.,'Stock Splits':0.})
        self.blocks=[[1,1],[2,5],[6,21],[22,63],[64,84]]
        self.write(self.frame)
    def tearDown(self):self.temp.cleanup()
    def write(self,frame):frame.to_csv(self.root/'TEST_daily_actions.csv',index=False)

    def test_supervised_block_is_exact_and_purged_by_label_end(self):
        data=StockDataset(self.root,['TEST'],self.blocks)
        f=data.labels['TEST'];anchor=self.index[80]
        row=f[(f.feature_time==anchor)&(f.block_end==5)].iloc[0]
        r=np.diff(np.log(self.frame.Close.to_numpy()))
        self.assertAlmostEqual(row.target_variance,np.mean(r[81:85]**2))
        # r[j] is the return ending at close[j+1], hence future closes 82..85.
        self.assertEqual(row.label_end,self.index[85])
        train,audit=data.training(self.index[190],window=150,min_anchors=100)
        self.assertTrue((train.label_end<=self.index[190]).all())
        self.assertEqual(train[train.block_end==84].feature_time.max(),self.index[106])
        self.assertEqual(pd.Timestamp(audit['TEST']['last_label_end']),self.index[190])

    def test_changing_all_future_prices_cannot_change_fit_or_forecast(self):
        first=StockDataset(self.root,['TEST'],self.blocks);cutoff=self.index[190]
        future=self.frame.copy()
        future.loc[future.index>190,['Open','High','Low','Close']]*=np.linspace(2,30,len(future)-191)[:,None]
        future.loc[future.index>190,'Volume']*=100
        self.write(future);second=StockDataset(self.root,['TEST'],self.blocks)
        a,_=first.training(cutoff,150,100);b,_=second.training(cutoff,150,100)
        pd.testing.assert_frame_equal(a,b)
        plan={'training_anchor_window':150,'min_training_anchors_per_asset':100,'variance_floor':1e-10,
              'har_proxy_ridge':{'alpha':10.},'ml_hist_gradient_boosting':{'loss':'poisson','max_iter':5,
                  'min_samples_leaf':10,'early_stopping':False,'random_state':7317}}
        ma,_=fit_models(first,cutoff,plan);mb,_=fit_models(second,cutoff,plan)
        for name in ma:
            fa=ma[name].curve(first.observed_features('TEST',cutoff),'TEST',cutoff)
            fb=mb[name].curve(second.observed_features('TEST',cutoff),'TEST',cutoff)
            np.testing.assert_allclose(fa.daily_variance,fb.daily_variance,rtol=0,atol=0)
            self.assertTrue((fa.daily_variance>0).all())

    def test_positive_blocks_cover_each_future_day_once(self):
        data=StockDataset(self.root,['TEST'],self.blocks);cutoff=self.index[190]
        model=PooledVarianceModel('ml_hist_gradient_boosting',['TEST'],self.blocks,ConstantBlocks(),1.,cutoff)
        fit=model.curve(data.observed_features('TEST',cutoff),'TEST',cutoff)
        expected=np.concatenate([np.repeat(i/10000,end-start+1) for i,(start,end) in enumerate(self.blocks,1)])
        np.testing.assert_array_equal(fit.daily_variance,expected)
        self.assertEqual(len(fit.daily_variance),84)
        with self.assertRaises(ValueError):model.curve(data.observed_features('TEST',cutoff),'TEST',self.index[189])


if __name__=='__main__':unittest.main()
