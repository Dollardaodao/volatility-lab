"""Past-only, pooled forecasts of positive future daily variance blocks.

Labels are squared close-to-close return proxies, not high-frequency RV.
Each supervised block must have finished before a model can train on it.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .forecast import Fit
from .stock_market import DailyInputs


FEATURES=['log_var_1','log_var_5','log_var_22','log_var_63','return_1','return_5','return_22',
          'log_downside_5','log_downside_22','log_range_5','log_range_22',
          'log_gap_5','log_gap_22','log_relative_volume_5','log_relative_volume_22']
HAR_FEATURES=['log_var_1','log_var_5','log_var_22']
FLOOR=1e-10


def past_features(frame,availability):
    if not availability.is_monotonic_increasing or availability.has_duplicates:
        raise ValueError('Daily availability must be strictly increasing')
    close=pd.Series(frame.Close.to_numpy(dtype=float),index=availability)
    r=np.log(close).diff();sq=r.pow(2)
    high=pd.Series(frame.High.to_numpy(dtype=float),index=availability)
    low=pd.Series(frame.Low.to_numpy(dtype=float),index=availability)
    op=pd.Series(frame.Open.to_numpy(dtype=float),index=availability)
    volume=pd.Series(frame.Volume.to_numpy(dtype=float),index=availability)
    if (low<=0).any() or (high<low).any() or (close<=0).any():raise ValueError('Invalid daily OHLC')
    range_var=np.log(high/low).pow(2)/(4*np.log(2))
    gap=np.log(op/close.shift()).pow(2)
    result=pd.DataFrame(index=availability)
    for h in (1,5,22,63):result[f'log_var_{h}']=np.log(sq.rolling(h).mean().clip(lower=FLOOR))
    for h in (1,5,22):result[f'return_{h}']=np.log(close).diff(h)
    for h in (5,22):
        result[f'log_downside_{h}']=np.log(r.clip(upper=0).pow(2).rolling(h).mean().clip(lower=FLOOR))
        result[f'log_range_{h}']=np.log(range_var.rolling(h).mean().clip(lower=FLOOR))
        result[f'log_gap_{h}']=np.log(gap.rolling(h).mean().clip(lower=FLOOR))
        result[f'log_relative_volume_{h}']=np.log((volume/volume.rolling(h).mean()).clip(lower=1e-6))
    return result.replace([np.inf,-np.inf],np.nan),sq


class StockDataset:
    def __init__(self,root,universe,blocks):
        self.universe=list(universe);self.blocks=[tuple(b) for b in blocks]
        self.daily={};self.features={};self.labels={}
        for symbol in self.universe:
            daily=DailyInputs(root,symbol);self.daily[symbol]=daily
            raw=pd.read_csv(root/f'{symbol}_daily_actions.csv')
            features,sq=past_features(raw,daily.prices.index)
            self.features[symbol]=features
            records=[]
            for start,end in self.blocks:
                target=sq.rolling(end-start+1).mean().shift(-end)
                label_end=pd.Series(sq.index,index=sq.index).shift(-end)
                f=features.copy();f['target_variance']=target
                f['feature_time']=f.index;f['label_end']=label_end
                f['block_start']=start;f['block_end']=end;f['asset']=symbol
                records.append(f.dropna())
            self.labels[symbol]=pd.concat(records,ignore_index=True)

    def training(self,asof,window=756,min_anchors=504):
        asof=pd.Timestamp(asof)
        if asof.tzinfo is None:raise ValueError('asof must be timezone aware')
        rows=[];audits={}
        for symbol in self.universe:
            available=self.daily[symbol].prices.loc[:asof].index
            if len(available)-1<min_anchors:raise ValueError('Insufficient past daily returns')
            first=available[-window:][0]
            f=self.labels[symbol]
            # Row counts alone are not a sufficient purge for multi-day labels.
            f=f[(f.feature_time>=first)&(f.feature_time<asof)&(f.label_end<=asof)].copy()
            if f.empty:raise ValueError('No fully observed supervised labels')
            if (f.label_end>asof).any():raise AssertionError('Future label entered training')
            rows.append(f)
            audits[symbol]={'rows':len(f),'first_anchor':str(f.feature_time.min()),
                'last_anchor':str(f.feature_time.max()),'last_label_end':str(f.label_end.max()),
                'by_block':{str(int(k)):int(v) for k,v in f.groupby('block_end').size().items()}}
        return pd.concat(rows,ignore_index=True),audits

    def observed_features(self,symbol,asof):
        f=self.features[symbol].loc[:pd.Timestamp(asof)].iloc[-1]
        if f.isna().any():raise ValueError('Missing past feature')
        return f


def design(rows,universe,kind):
    names=HAR_FEATURES if kind=='har_proxy_ridge' else FEATURES
    values=rows[names].to_numpy(dtype=float)
    horizons=np.log1p(rows[['block_start','block_end']].to_numpy(dtype=float))
    assets=np.column_stack([(rows.asset==symbol).to_numpy(dtype=float) for symbol in universe])
    return np.column_stack([values,horizons,assets])


@dataclass
class PooledVarianceModel:
    name:str
    universe:list
    blocks:list
    estimator:object
    smearing:float
    trained_asof:pd.Timestamp

    def curve(self,features,symbol,feature_time):
        if pd.Timestamp(feature_time)<self.trained_asof:
            raise ValueError('Cannot apply a fitted model to a prior feature date')
        rows=pd.DataFrame([{**features.to_dict(),'asset':symbol,'block_start':start,'block_end':end}
                           for start,end in self.blocks])
        with threadpool_limits(limits=1):pred=self.estimator.predict(design(rows,self.universe,self.name))
        if self.name=='har_proxy_ridge':pred=np.exp(pred)*self.smearing
        pred=np.maximum(pred/10000,FLOOR)
        if not np.isfinite(pred).all():raise ValueError('Nonfinite ML variance forecast')
        curve=np.concatenate([np.repeat(v,end-start+1) for v,(start,end) in zip(pred,self.blocks)])
        return Fit(self.name,pd.Timestamp(feature_time),curve,
                   {'model_fit_cutoff':str(self.trained_asof),'feature_time':str(feature_time),
                    'daily_variance_blocks':pred.tolist(),'smearing':self.smearing})


def fit_models(dataset,asof,plan):
    rows,audit=dataset.training(asof,plan['training_anchor_window'],plan['min_training_anchors_per_asset'])
    target=rows.target_variance.clip(lower=plan['variance_floor']).to_numpy()*10000
    with threadpool_limits(limits=1):
        ridge=make_pipeline(StandardScaler(),Ridge(alpha=plan['har_proxy_ridge']['alpha']))
        x=design(rows,dataset.universe,'har_proxy_ridge');log_target=np.log(target)
        ridge.fit(x,log_target)
        # Training-only correction from predicted log variance to arithmetic
        # mean variance; no residuals from validation/trading windows are used.
        smear=float(np.mean(np.exp(log_target-ridge.predict(x))))
        tree=HistGradientBoostingRegressor(**plan['ml_hist_gradient_boosting'])
        tree.fit(design(rows,dataset.universe,'ml_hist_gradient_boosting'),target)
    models={name:PooledVarianceModel(name,dataset.universe,dataset.blocks,estimator,smearing,pd.Timestamp(asof))
            for name,estimator,smearing in [('har_proxy_ridge',ridge,smear),('ml_hist_gradient_boosting',tree,1.)]}
    return models,{'fit_cutoff':str(asof),'training_rows':len(rows),'per_asset':audit,
                   'feature_names':FEATURES,'ridge_features':HAR_FEATURES,'ridge_training_smearing':smear}
