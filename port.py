#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np

import datetime as dt
import math
import ipywidgets as widgets
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
import ew as ew
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro 

# parameter tickers,weights,initial_investment,start_date,end_date

class portfolio():
    
    def __init__(self,tickers,weights,initial_investment,start_date,end_date):
        self.tickers=tickers
        self.weights=weights
        self.initial_investment=initial_investment
        self.start_date=start_date
        self.end_date=end_date
#         fetch data from yahoo
    def get_data(self):
        self.data=pdr.get_data_yahoo(self.tickers, start=self.start_date, end=self.end_date)['Close']
        self.returns=self.data.pct_change().dropna()
        self.weighted_returns=(self.weights * self.returns)
        self.port_weighted_ret = self.weighted_returns.sum(axis=1)
        self.avg_rets = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        self.port_mean = self.avg_rets.dot(self.weights)
        self.port_stdev = np.sqrt(self.weights.T.dot(self.cov_matrix).dot(self.weights))
#    ploting portfolio weighted_returns
    def plot_port_weighted_returns(self):
        plt.plot(self.port_weighted_ret)
        title='daily weighted returns of portfolio'
        plt.title(title)
        plt.show()
#         ploting _each_asset_returns
    def plot_each_asset_returns(self):
        for tic in self.tickers:
            plt.plot(self.returns[tic])
            title='daily_returns of '+tic
            plt.title(title)
            plt.show()
    def norm_test(self):
         for tic in self.tickers:
            result=shapiro(self.returns[tic])
            print(f'norm Statistic for {tic}: {result[0]}')
            print(f'p-value: {result[1]}')
    def ADF_test_port(self):
        result = adfuller(self.port_weighted_ret, autolag='AIC')
        print(f'ADF Statistic for portfolio: {result[0]}')
        print(f'n_lags: {result[1]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
            print('Critial Values:')
            print(f'   {key}, {value}') 
    
    def ADF_test(self):
        for tic in self.tickers:
            result = adfuller(self.returns[tic], autolag='AIC')
            print(f'ADF Statistic for {tic}: {result[0]}')
            print(f'n_lags: {result[1]}')
            print(f'p-value: {result[1]}')
            for key, value in result[4].items():
                print('Critial Values:')
                print(f'   {key}, {value}') 
#                 histogram for each asset
    def plot_each_asset_returns_hist(self):
        for tic in self.tickers:
            self.returns[tic].hist(bins=40,histtype="stepfilled",alpha=0.5)
            x = np.linspace(self.port_mean - 3*self.port_stdev, self.port_mean+3*self.port_stdev,100)
            plt.plot(x,stats.norm.pdf(x, self.port_mean, self.port_stdev), "r")
            plt.title(tic+"  returns (binned) vs. normal distribution")
            plt.show()
#         for i in self.tickers:
#             fig = plt.figure()
#             ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
#             ax1.hist(self.returns.loc[:,i], bins = 60)
#             ax1.set_xlabel(""+str(i)+" returns")
#             ax1.set_ylabel("Freq")
#             ax1.set_title("histogram of returns")
#             plt.show(); 
   
    def plot_portfolio_hist_returns(self):
        fig = plt.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.hist(self.port_weighted_ret, bins = 60)
        ax1.set_xlabel('Portfolio weighted returns')
        ax1.set_ylabel("Freq")
        ax1.set_title("histogram of returns")
        plt.show(); 
    def port_historical_simulation_var(self,alpha):
        self.sorted_ret=self.port_weighted_ret.sort_values()
        self.varhs=self.sorted_ret.quantile(alpha)
        belowVaR = self.sorted_ret<=self.varhs
        self.EShs=self.sorted_ret[belowVaR].mean()
        return [self.varhs*self.initial_investment , self.EShs*self.initial_investment]
    def bootstrap_oneday_var(self,alpha):
        pb = (pd.DataFrame([random.choices(list(self.port_weighted_ret), k=273) for i in range(1000)]).T.shift(1)).dropna()
        t=pb.to_numpy()
        x=np.sort(t,axis=0)
        self.varbs=np.mean([np.quantile(x[:,i],alpha) for i in range(1000)])
        self.bs_oneday_var=self.varbs
        return  self.bs_oneday_var*self.initial_investment
#     def bootstrap_nday_vat(self,alpha,n):
#         varnday=np.zeros(300)
#         for i in range(300):
#             pb = (pd.DataFrame([random.choices(list(self.port_weighted_ret), k=n+1) for i in range(len(self.port_weighted_ret))]).T.shift(1)).dropna().to_numpy()
#             varnday[i]=np.quantile(np.sort(np.sum(pb,axis=0)),alpha)
#         self.n_day_var=np.mean(varnday)
#         return self.n_day_var*self.initial_investment
    def bootstrap_nday_var(self,alpha,n):
        varnday=np.zeros(300)
        returns_1day=np.random.choice(list(self.port_weighted_ret),size=(300,n,len(self.port_weighted_ret)))
        n_day_return=np.sum(returns_1day,axis=1)
        sorted_nday_return=np.sort(n_day_return,axis=1)
        self.boot_allvar=np.quantile(sorted_nday_return,alpha,axis=1)
        self.n_day_var=self.boot_allvar.mean()
        return self.n_day_var*self.initial_investment
    def bootdtrap_conf_interval(self,alpha):
        return [np.quantile(np.sort(self.boot_allvar),(1-(alpha/2)))*self.initial_investment,np.quantile(np.sort(self.boot_allvar),alpha/2)*self.initial_investment]
    def gausian_var(self,alpha,n):
        
#         self.mean_investment = (1+self.port_mean) * self.initial_investment
#         self.stdev_investment = self.initial_investment * self.port_stdev
#         cutoff1 = norm.ppf(alpha, self.mean_investment, self.stdev_investment)
#         g_1d_var = -(self.initial_investment - cutoff1)
#         self.g_nd_var=g_1d_var  * np.sqrt(n)
        self.g_var=norm.ppf(alpha,n*self.port_mean,np.sqrt(n)*self.port_stdev)
        self.g_nd_var=self.initial_investment*self.g_var
        self.ES_g =-self.initial_investment*(alpha**-1 * norm.pdf(norm.ppf(alpha))*(np.sqrt(n)*self.port_stdev) - (n*self.port_mean))
        
        return [self.g_nd_var, self.ES_g]
    def gausian_var_graph(self,alpha,n):
        var_array = np.zeros(n)
        Es_array=np.zeros(n)
        
        for x in range(0, n):    
            var_array[x]=(self.initial_investment*norm.ppf(alpha,(x+1)*self.port_mean,np.sqrt(x+1)*self.port_stdev))
            Es_array[x]=-self.initial_investment*(alpha**-1 * norm.pdf(norm.ppf(alpha))*(np.sqrt(x+1)*self.port_stdev) - ((x+1)*self.port_mean))
            print(str(x+1) + " day VaR and expected shortfall@ "+str(alpha)+"% confidence: " + str(np.round((var_array[x]),2))+"  "+str(np.round((Es_array[x]),2)))
            
        plt.xlabel("Day #")
        plt.ylabel("Max portfolio loss (USD)")
        plt.title("Max portfolio loss over "+str(n)+"-day period")
        plt.plot(-var_array , label="Var ")
        plt.plot(-Es_array,label="Expected shortfall ")
        plt.legend()
        plt.show()
     
    def stock_return_vs_normal(self):
        for tic in self.tickers:
            self.returns[tic].hist(bins=40,histtype="stepfilled",alpha=0.5)
            x = np.linspace(self.port_mean - 3*self.port_stdev, self.port_mean+3*self.port_stdev,100)
            plt.plot(x,stats.norm.pdf(x, self.port_mean, self.port_stdev), "r")
            plt.title(" "+tic+" returns (binned) vs. normal distribution")
            plt.show()
    def Monte_n_var(self,alpha,n):
        mc_sims=2000
        mean_M=np.full(shape=(mc_sims,n, len(self.weights)), fill_value=self.avg_rets)
        self.L = np.linalg.cholesky(self.cov_matrix) 
        z=np.random.normal(size=(mc_sims,n, len(self.weights)))
        daily_returns=np.inner(z,self.L)+mean_M
        cummulative_returns=daily_returns.sum(axis=1)
        self.weighted_mret=np.inner(cummulative_returns,self.weights)
        self.M_n_var=np.quantile(np.sort(self.weighted_mret),alpha)
        belowVaR = self.weighted_mret<=self.M_n_var
        self.ESM=self.weighted_mret[belowVaR].mean()
        
        return [self.M_n_var*self.initial_investment,self.ESM*self.initial_investment]
    def Monte_n_var_graph(self,alpha,n):
        u=[]
        v=[]
        for x in range(n):
            res=self.Monte_n_var(alpha,n)
            u.append(-res[0])
            v.append(-res[1])
            print(str(x+1) + " day VaR and expected shortfall@ "+str(alpha)+"% confidence: " + str(np.round((u[x]),2))+"  "+str(np.round((v[x]),2)))
            
        
        plt.title("Max portfolio loss over "+str(n)+"-day period")
        plt.xlabel("Day #")
        plt.ylabel("Max portfolio loss (USD)")
        plt.plot(u,label="Var ")
        plt.plot(v,label="expexted Shortfall ")
        plt.show()
        
    def ewma_n_day(self,lamda,alpha,n_days):
        mc_sim=1000
        cov_mm=self.returns.cov()
        last_ret=self.returns.iloc[-1,:]
        m_ret=self.avg_rets
        
        t=np.zeros(mc_sim)
        for n in range (mc_sim):
            c_vari=cov_mm
            l_s=last_ret
            cum_ret=0.0
            for day in range(n_days):
                Z = np.random.normal(size=len(self.weights))
                L = np.linalg.cholesky(c_vari)
                t_ret=np.inner(L,Z)+m_ret
                p_ret=np.inner(t_ret,self.weights.T)
                cum_ret=cum_ret+p_ret
                c_vari=ew.update_ewmacov(c_vari,t_ret,lamda)
               
            t[n]=cum_ret
        self.ew_n_var=np.quantile(np.sort(t),alpha)
        belowVaR = t<=self.ew_n_var
        self.ew_n_Es=t[belowVaR].mean()
        return [self.ew_n_var*self.initial_investment,self.ew_n_Es*self.initial_investment]


# In[ ]:




