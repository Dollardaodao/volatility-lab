"""Compare actual holding-window RV with actual entry ask-IV.

This is a descriptive volatility-gap test, not a gamma-weighted replication
identity or a causal attribution of P&L. It does not change any trading result.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from .american import AmericanPricer
from .theta_sdk import ROOT


def main():
    study=ROOT/'artifacts'/'stock_study_20'
    out=ROOT/'artifacts'/'trade_diagnostics'
    source=study/'completed_results.csv'
    frame=pd.read_csv(source,low_memory=False)
    frame=frame[(frame.policy=='garch_long_flat')&(frame.hedge_minutes==60)&
                (frame.cost_scenario=='zero_bid_writedown')&(frame.side==1)]
    rows=[];hashes={}
    for r in frame.itertuples():
        path=study/f'{r.asset}_{r.origin}_{r.target_dte}d_priced.csv'
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        panel=pd.read_csv(path,index_col=0,parse_dates=True)
        panel.index=pd.to_datetime(panel.index,utc=True)
        start,end=pd.Timestamp(r.entry),pd.Timestamp(r.exit)
        held=panel.loc[start:end]
        assert held.index[0]==start and held.index[-1]==end
        entry=held.iloc[0];ivs={}
        for kind in ('call','put'):
            pricer=AmericanPricer(r.strike,kind)
            pricer.configure(float(entry.spot),float(entry.years_to_expiry),float(entry.rate),float(entry.dividend_yield))
            ivs[kind]=pricer.implied(float(entry[kind+'_ask']))
        years=(end-start).total_seconds()/(365*86400)
        rv=float(np.sqrt(np.square(np.diff(np.log(held.spot.to_numpy()))).sum()/years))
        market=r.gross_option_pnl+r.hedge_gross_pnl+r.dividends
        cost=sum(getattr(r,k) for k in ('option_spread_cost','stock_spread_cost','slippage_cost','option_fees','stock_fees'))
        financing=r.cash_interest-r.benchmark_interest-r.borrow_cost
        assert abs(r.net_pnl-(market-cost+financing))<1e-5
        rows.append({'asset':r.asset,'origin':r.origin,'target_dte':r.target_dte,'entry':r.entry,'exit':r.exit,
            'entry_call_ask_iv':ivs['call'],'entry_put_ask_iv':ivs['put'],'entry_conservative_ask_iv':max(ivs.values()),
            'holding_realized_vol':rv,'holding_rv_above_entry_ask_iv':rv>max(ivs.values()),
            'original_rv_to_expiry_above_signal_ask_iv':r.realized_vol_to_expiry>r.long_execution_iv,
            'market_pnl':market,'trading_cost':cost,'net_financing':financing,'net_pnl':r.net_pnl})
    outcomes=pd.DataFrame(rows)
    outcomes.to_csv(out/'holding_matched_vol_bet_outcomes_92.csv',index=False)
    summaries=[]
    original=outcomes.asset.isin(['AAPL','MSFT','NVDA','AMZN','JPM','KO'])
    for name,f in [('original_22',outcomes[original]),('expanded_92',outcomes)]:
        summaries.append({'sample':name,'trades':len(f),
            'holding_rv_above_entry_ask_iv':int(f.holding_rv_above_entry_ask_iv.sum()),
            'previous_signal_to_expiry_test':int(f.original_rv_to_expiry_above_signal_ask_iv.sum()),
            'positive_market_pnl_before_costs':int((f.market_pnl>0).sum()),'positive_net_pnl':int((f.net_pnl>0).sum()),
            'vol_gap_positive_but_net_loss':int((f.holding_rv_above_entry_ask_iv&(f.net_pnl<0)).sum())})
    pd.DataFrame(summaries).to_csv(out/'vol_bet_count_summary.csv',index=False)
    metadata={'policy':'garch_long_flat','hedge_minutes':60,'cost_scenario':'zero_bid_writedown',
        'realized_vol':'Square root of 15-minute and overnight log-return squared sum divided by actual entry-to-exit ACT/365 years',
        'entry_iv':'Maximum of actual entry call ask-IV and put ask-IV; conservative two-leg comparison, not a unique composite straddle IV',
        'iv_engine':'AmericanPricer with observed entry spot, exact remaining time, past-only rate and dividend inputs',
        'interpretation':'A simple unweighted RV/IV diagnostic. Gamma weighting, IV repricing and discrete hedging can make market P&L disagree with its sign.',
        'previous_metric':'09:45 signal ask-IV compared with RV from signal to expiry; distinct from actual entry-to-exit window',
        'source_results_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_panel_hashes':hashes,'summary':summaries}
    (out/'vol_bet_count_method.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__=='__main__':main()
