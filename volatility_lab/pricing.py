"""USD European option value and delta, used only for model-based hedges."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def bs_price_delta(spot, strike, years, vol, rate, kind):
    if kind not in ("call", "put") or min(spot, strike, years, vol) <= 0:
        raise ValueError("Invalid European option inputs")
    d1 = (np.log(spot/strike)+(rate+.5*vol*vol)*years)/(vol*np.sqrt(years))
    d2 = d1-vol*np.sqrt(years)
    sign = 1 if kind == "call" else -1
    price = sign * (spot*ndtr(sign*d1)-strike*np.exp(-rate*years)*ndtr(sign*d2))
    delta = ndtr(d1) if kind == "call" else ndtr(d1)-1
    return float(price), float(delta)


def implied_vol_delta(price, spot, strike, years, rate, kind):
    low = bs_price_delta(spot,strike,years,.0001,rate,kind)[0]
    high = bs_price_delta(spot,strike,years,8.,rate,kind)[0]
    if not low < price < high:
        raise ValueError("Price cannot be inverted under the chosen USD European model")
    vol = brentq(lambda v:bs_price_delta(spot,strike,years,v,rate,kind)[0]-price,.0001,8.)
    return vol, bs_price_delta(spot,strike,years,vol,rate,kind)[1]
