#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import fix_yahoo_finance as yf
import arch
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
class stock_option():
    def __init__(self, K, r, T, N,No_options,ticker,start_date,end_date,is_call,alpha):
       
        self.K = K
        self.r = r
        self.T = T
        self.alpha=alpha
        self.is_call=is_call
        self.No_options=No_options
        self.N = N
        self.ticker=ticker
        self.start_date=start_date
        self.end_date=end_date
        all_data = pdr.get_data_yahoo(self.ticker, start=self.start_date, end=self.end_date)
        self.stock_data = pd.DataFrame(all_data['Adj Close'], columns=["Adj Close"])
        self.stock_data["log"] = np.log(self.stock_data)-np.log(self.stock_data.shift(1))
#         self.div = prm.get('div', 0)
        self.dt = T/float(N)
#         self.df = math.exp(-(r-self.div)*self.dt)
        self.mean_returns=self.stock_data['log'].mean()
        self.current_price=self.stock_data.iloc[-1,0]
        self.S0 = self.stock_data.iloc[-1,0]
    def mean_sigma(self):
        st = self.stock_data["log"].dropna().ewm(span=252).std()
        self.sigma = st.iloc[-1]
        return self.sigma
    
    
    def garch_sigma(self):
        model = arch.arch_model(self.stock_data["log"].dropna(), mean='Zero', vol='GARCH', p=1, q=1)
        model_fit = model.fit()
        forecast = model_fit.forecast(horizon=1)
        var = forecast.variance.iloc[-1]
        self.sigma = float(np.sqrt(var))
        return self.sigma
    def one_day_var(self):
        mc_sims = 400 
        z=np.random.normal(size=(mc_sims))
        daily_sim_returns=self.mean_returns+self.sigma*z
        price_sim=self.current_price*np.exp(daily_sim_returns)
        p_binomial= np.full(shape=( mc_sims), fill_value=0.0)
        p_black= np.full(shape=( mc_sims), fill_value=0.0)
        current_opt_price_binom = euro_option(self.current_price,self.K, self.r, self.T, self.N,self.sigma,self.is_call).price()
        if self.is_call:
            current_opt_price_black=BS_CALL(self.current_price,self.K,self.T,self.r,self.sigma)
        else:
            current_opt_price_black=BS_PUT(self.current_price,self.K,self.T,self.r,self.sigma)
        for i in range( mc_sims):
            p_binomial[i] = euro_option(price_sim[i], self.K, self.r, self.T, self.N,self.sigma,self.is_call).price()
            if self.is_call:
                p_black[i]=BS_CALL(price_sim[i],self.K,self.T,self.r,self.sigma)
            else:
                p_black[i]=BS_CALL(price_sim[i],self.K,self.T,self.r,self.sigma)
        
        
        pandL_binom= p_binomial-current_opt_price_binom
        pandL_black= p_black-current_opt_price_black
        
        self.t2=pandL_black
        self.var_binom=np.quantile(np.sort(pandL_binom),self.alpha)*self.No_options
        self.var_black=np.quantile(np.sort(pandL_black),self.alpha)*self.No_options
    def n_day_var(self,n):
        mc_sims = 400 
        z=np.random.normal(size=(mc_sims,n))
        daily_sim_returns=self.mean_returns+self.sigma*z
        cummulative_returns=daily_sim_returns.sum(axis=1)
        n_price_sim=self.current_price*np.exp(cummulative_returns)
#         self.t=price_sim
        p_binomial= np.full(shape=( mc_sims), fill_value=0.0)
        p_black= np.full(shape=( mc_sims), fill_value=0.0)
        current_opt_price_binom = euro_option(self.current_price,self.K, self.r, self.T, self.N,self.sigma,self.is_call).price()
        if self.is_call:
            current_opt_price_black=BS_CALL(self.current_price,self.K,self.T,self.r,self.sigma)
        else:
            current_opt_price_black=BS_PUT(self.current_price,self.K,self.T,self.r,self.sigma)
        for i in range( mc_sims):
            p_binomial[i] = euro_option(n_price_sim[i], self.K, self.r, self.T, self.N,self.sigma,self.is_call).price()
            if self.is_call:
                p_black[i]=BS_CALL(n_price_sim[i],self.K,self.T,self.r,self.sigma)
            else:
                p_black[i]=BS_CALL(n_price_sim[i],self.K,self.T,self.r,self.sigma)
        pandL_binom= p_binomial-current_opt_price_binom
        
        pandL_black= p_black-current_opt_price_black
        self.t=pandL_black
        self.nvar_binom=np.quantile(np.sort(pandL_binom),self.alpha)*self.No_options
        self.nvar_black=np.quantile(np.sort(pandL_black),self.alpha)*self.No_options
        
        
        
        
# self.tk = prm.get('tk', None)
# self.start = prm.get('start', None)
# self.end = prm.get('end', None)
    
# 		self.is_calc = prm.get('is_calc', False)
# 		self.use_garch = prm.get('use_garch', False)
		
# 		if self.is_calc:
# 			self.vol = stock_vol(self.tk, self.start, self.end)
# 			if self.use_garch:
# 				self.sigma = self.vol.garch_sigma()
# 			else:
# 				self.sigma = self.vol.mean_sigma()
# 		else:
# 			self.sigma = prm.get('sigma', 0)
# 		self.is_call = prm.get('is_call', True)
# 		self.eu_option = prm.get('eu_option', True)

        
        
# derived values:
# dt = time per step, in years
# df = discount factor

        


# In[2]:
# to calculate oprion prices using binomial tress and black sholes option pricing

class euro_option():
    def __init__(self, S0, K, r, T, N,sigma,is_call):
        self.S0=S0
        self.K=K
        self.r=r
        self.T=T
        self.N=N
        self.div=0
        self.sigma=sigma
        self.is_call=is_call
        self.M = self.N + 1 
        self.dt = T/float(N)
        self.df = math.exp(-(r-self.div)*self.dt)
        self.u = math.exp(self.sigma*math.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (math.exp((self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd = 1-self.qu

                          
    def stocktree(self):
        stocktree = np.zeros([self.M, self.M])
        for i in range(self.M):
            for j in range(self.M):
                stocktree[j, i] = self.S0*(self.u**(i-j))*(self.d**j)
        return stocktree
    def option_price(self, stocktree):
        option = np.zeros([self.M, self.M])
        if self.is_call:
            option[:, self.M-1] = np.maximum(np.zeros(self.M), (stocktree[:, self.N] - self.K))
        else:
            option[:, self.M-1] = np.maximum(np.zeros(self.M), (self.K - stocktree[:, self.N]))
        return option

    def optpricetree(self, option):
        for i in np.arange(self.M-2, -1, -1):
            for j in range(0, i+1):
                option[j, i] = math.exp(-self.r*self.dt) * (self.qu*option[j, i+1]+self.qd*option[j+1, i+1])
        return option
    def begin_tree(self):
        stocktree = self.stocktree()
        payoff = self.option_price(stocktree)
        return self.optpricetree(payoff)
    def price(self):
#         self.__int_prms__()
        self.stocktree()
        payoff = self.begin_tree()
        return payoff[0, 0]


# In[3]:
# black sholes option pricing

from scipy.stats import norm

N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)


