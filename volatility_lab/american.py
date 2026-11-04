"""Fast American approximation with exact remaining-time scaling.

QuantLib's Bjerksund-Stensland engine uses a continuous dividend-yield
approximation. Cash dividends are separately booked in the trading ledger.
"""
import math
import numpy as np
import QuantLib as ql
from scipy.optimize import brentq


class AmericanPricer:
    def __init__(self,strike,kind):
        self.base=ql.Date(1,1,2000)
        ql.Settings.instance().evaluationDate=self.base
        self.dc=ql.Actual365Fixed()
        self.spot=ql.SimpleQuote(100);self.vol=ql.SimpleQuote(.3)
        self.rate=ql.SimpleQuote(.04);self.div=ql.SimpleQuote(0)
        self.process=ql.BlackScholesMertonProcess(ql.QuoteHandle(self.spot),
            ql.YieldTermStructureHandle(ql.FlatForward(self.base,ql.QuoteHandle(self.div),self.dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(self.base,ql.QuoteHandle(self.rate),self.dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self.base,ql.NullCalendar(),ql.QuoteHandle(self.vol),self.dc)))
        self.kind=kind;self.strike=strike;self.days=None;self.option=None

    def configure(self,spot,years,rate,dividend_yield):
        if min(spot,years)<=0:raise ValueError("Invalid American option inputs")
        days=max(1,int(math.ceil(years*365)))
        if days!=self.days:
            kind=ql.Option.Call if self.kind=="call" else ql.Option.Put
            self.option=ql.VanillaOption(ql.PlainVanillaPayoff(kind,self.strike),ql.AmericanExercise(self.base,self.base+days))
            self.option.setPricingEngine(ql.BjerksundStenslandApproximationEngine(self.process))
            self.days=days
        # Rescale constant coefficients so the synthetic whole-day QL clock
        # integrates exactly the real fractional remaining time.
        self.scale=years/(days/365)
        self.spot.setValue(float(spot));self.rate.setValue(rate*self.scale);self.div.setValue(dividend_yield*self.scale)

    def value(self,vol):
        self.vol.setValue(max(vol,1e-5)*math.sqrt(self.scale))
        return float(self.option.NPV())

    def delta(self,vol):
        self.value(vol)
        return float(self.option.delta())

    def implied(self,price):
        low,high=self.value(.005),self.value(5.)
        if not low<price<high:raise ValueError("Quote outside American model inversion range")
        return float(brentq(lambda v:self.value(v)-price,.005,5.,xtol=1e-5))


def enrich_american(panel,strike,daily,rates):
    panel=panel.copy();pricers={k:AmericanPricer(strike,k) for k in ("call","put")}
    previous_iv={"call":None,"put":None};last_time={"call":None,"put":None}
    values={c:[] for c in ("call_iv","put_iv","call_delta","put_delta","rate","dividend_yield","iv_carried","iv_cross_leg")}
    for time,row in panel.iterrows():
        r=rates.at(time);q=daily.yield_at(time,float(row.spot));values["rate"].append(r);values["dividend_yield"].append(q)
        carried=False;current={};fresh={}
        for kind in ("call","put"):
            iv,delta=np.nan,np.nan
            if row.years_to_expiry>0:
                p=pricers[kind];p.configure(float(row.spot),row.years_to_expiry,r,q)
                if bool(row[kind+"_passes_snapshot_filters"]):
                    try:iv=p.implied(float(row[kind+"_mid"]));previous_iv[kind]=iv;last_time[kind]=time
                    except (ValueError,RuntimeError):pass
                fresh[kind]=bool(np.isfinite(iv))
                if not np.isfinite(iv) and previous_iv[kind] is not None and (time-last_time[kind]).total_seconds()<=4*86400:
                    iv=previous_iv[kind];carried=True
            current[kind]=iv
        cross_leg=False
        for kind,other in (("call","put"),("put","call")):
            iv=current[kind];delta=np.nan
            # Same-strike/expiry legs share a volatility-process proxy. A
            # fresh opposite-leg IV is preferable to freezing an unbid leg.
            if not fresh.get(kind,False) and fresh.get(other,False):iv=current[other];cross_leg=True
            if row.years_to_expiry>0 and np.isfinite(iv):
                try:delta=pricers[kind].delta(iv)
                except RuntimeError:delta=np.nan
            values[kind+"_iv"].append(iv);values[kind+"_delta"].append(delta)
        values["iv_carried"].append(carried)
        values["iv_cross_leg"].append(cross_leg)
    for k,v in values.items():panel[k]=v
    panel["delta_per_straddle"]=panel.call_delta+panel.put_delta
    return panel
