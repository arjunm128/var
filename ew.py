#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
# function to calculate EWMA VARIANCE COVARIANCE 
# PARAMETER ARE RETURN SERIES AND LAMDA THE SMOOTHENING PARAMETER
def ewma_cov_pd(rets, alpha):
    assets = rets.columns
    n = len(assets)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cov[i, j] = cov[j, i] = ewma_cov_pairwise_pd(
                rets.iloc[:, i], rets.iloc[:, j], alpha=alpha)
    return pd.DataFrame(cov, columns=assets, index=assets)


# In[2]:

# function to calculate EWMA OF TWO SERIES 
# PARAMETER TWO SERIES AND LAMDA THE SMOOTHENING PARAMETER
def ewma_cov_pairwise_pd(x, y, alpha):
    x = x.mask(y.isnull(), np.nan)
    y = y.mask(x.isnull(), np.nan)
    covariation = (x - x.mean()) * (y - y.mean()).dropna()
   
    covariation =covariation.ewm(alpha).mean().iloc[-1]
    return covariation



# function to UPDATE EWMA OF TWO SERIES VARIANCE COVARIANCE 

def update_ewmacov(ewm,ret,alpha):
    updated_ewacov=(alpha*ewm)+(1-alpha)*np.inner(ret,ret)
    return updated_ewacov


# function to CALCULATE EWMA VOLATILITY OF A STOCK  


def ewma_stock(returns,alpha):
    variation=(returns-returns.mean())*(returns-returns.mean())
    return variation.ewm(alpha).mean().iloc[-1]


# function to UPDATE EWMA VOLATILITY OF A STOCK 

def update_ewmastock(ewma,alpha,ret):
    updated_ewa=(alpha*ewma)+(1-alpha)*(ret*ret)
    return updated_ewa


# different methodogys for 
def CalculateEWMAVol (ReturnSeries, Lambda):   
    SampleSize = len(ReturnSeries)
    Average = ReturnSeries.mean()

    e = np.arange(SampleSize-1,-1,-1)
    r = np.repeat(Lambda,SampleSize)
    vecLambda = np.power(r,e)

    sxxewm = (np.power(ReturnSeries-Average,2)*vecLambda).sum()
    Vart = sxxewm/vecLambda.sum()
   

    return (Vart)

def CalculateVol (R, Lambda):
    Vol = pd.Series(index=R.columns)
    for facId in R.columns:
        Vol[facId] = CalculateEWMAVol(R[facId], Lambda)

    return (Vol)

