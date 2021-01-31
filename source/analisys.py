import pandas as pd
import numpy as np

def omega_ratio(returns, threshold = .0):
    dailyThreshold = (threshold + 1) ** np.sqrt(1/252) - 1
    excess = returns - dailyThreshold
    positiveSum = np.sum(excess[excess > 0])
    negativeSum = np.sum(excess[excess < 0])
    if negativeSum == 0:
        return np.nan
    omega = positiveSum/(-negativeSum)
    return omega

def tail_ratio(returns):
    if (np.abs(np.percentile(returns, 5))) == 0:
        return np.nan
    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))

def max_drawdown(returns):
    i = (1 + returns).cumprod()
    return np.min(i/i.expanding().max()) - 1

def cum_return(returns):
    return np.cumprod((1 + returns))[-1] -1

def annual_rate_return(returns):
    return returns.mean() * 252

def annualyzed_return (returns):
    n = (returns.index.max() - returns.index.min()).days/366
    return ((1 + cum_return(returns))**(1/n)) - 1

def annualyzed_volatility (returns):
    return returns.std()*np.sqrt(252)
