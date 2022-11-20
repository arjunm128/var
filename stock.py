#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np
import backtest as bk
import datetime as dt
import math
import ipywidgets as widgets
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
import ew as ew
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.optimize as spop
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import backtest as bt
class asset():
    def __init__(self,ticker,initial_investment,start_date,end_date,a_type='stock'):
        self.ticker=ticker
        self.start_date=start_date
        self.end_date=end_date
        self.a_type='stock'
        self.initial_investment=initial_investment
        
    def get_data(self):
        self.data=pdr.get_data_yahoo(self.ticker, start=self.start_date, end=self.end_date)['Close']
        self.returns=self.data.pct_change().dropna()
        self.mean_ret=self.returns.mean()
        self.variance=self.returns.var()
        self.std=self.returns.std()
        self.prices=yf.download(self.ticker, self.start_date, self.end_date)['Close']
        self.ret = np.array(self.prices)[1:]/np.array(self.prices)[:-1] - 1
    def plot_stock_returns(self):
        self.returns.plot(title='daily_returns')
    def historical_simulation_var(self,alpha):
        self.sorted_ret=np.sort(self.returns,axis=None)
        self.varhs=np.quantile(self.sorted_ret,alpha)
        return [self.varhs*self.initial_investment ,self.varhs]
    def gausian_var(self,alpha,n):

        self.g_nd_var=self.initial_investment*norm.ppf(alpha,n*self.mean_ret,np.sqrt(n)*self.std)
        return self.g_nd_var
    def bootstap_oneday_var(self,alpha):
        pb = (pd.DataFrame([random.choices(list((self.returns.iloc[:,0])), k=273) for i in range(1000)]).T.shift(1)).dropna()
        t=pb.to_numpy()
        x=np.sort(t,axis=0)
        self.boot_allvar=[np.quantile(x[:,i],alpha) for i in range(1000)]
        varbs=np.mean(self.boot_allvar)
        self.bs_oneday_var=varbs
        return  self.bs_oneday_var*self.initial_investment
    def bootstrap_nday_var(self,alpha,n):
        varnday=np.zeros(300)
        returns_1day=np.random.choice(list((self.returns.iloc[:,0])),size=(300,n,len(self.returns.iloc[:,0])))
        n_day_return=np.sum(returns_1day,axis=1)
        sorted_nday_return=np.sort(n_day_return,axis=1)
        self.boot_allvar=np.quantile(sorted_nday_return,alpha,axis=1)
        self.n_day_var=self.boot_allvar.mean()
        return self.n_day_var*self.initial_investment
    def bootdtrap_conf_interval(self,alpha):
        return [np.quantile(np.sort(self.boot_allvar),1-(alpha)/2)*self.initial_investment,np.quantile(np.sort(self.boot_allvar),alpha/2)*self.initial_investment]
    def Monte_n_var(self,n,alpha):
        mc_sims=2000
        z=np.random.normal(size=(mc_sims,n))
        daily_returns=z*self.std[0]+self.mean_ret[0]
        cummulative_returns=daily_returns.sum(axis=1)
        self.M_n_var=np.quantile(np.sort(cummulative_returns),alpha)*self.initial_investment
        return self.M_n_var*self.initial_investment
    def ewma_n_day(self,lamda,alpha,n_days):
        mc_sim=1000
        var_mm= ew.CalculateEWMAVol(self.returns.iloc[:,0],lamda)
        last_ret=self.returns.iloc[-1,0]
        m_ret=self.mean_ret[0]
        
        t=np.zeros(mc_sim)
        for n in range (mc_sim):
            c_vari=var_mm
            l_s=last_ret
            cum_ret=0.0
            for day in range(n_days):
                Z = np.random.normal()
                
                t_ret=math.sqrt(c_vari)*Z+m_ret
                cum_ret=cum_ret+t_ret
                c_vari=ew.update_ewmastock(c_vari,lamda,t_ret)
               
            t[n]=cum_ret
        self.ew_n_var=np.quantile(np.sort(t),0.05)
        return self.ew_n_var*self.initial_investment
#     def garch(self,alpha):
#         plt.figure(figsize=(10,4))
#         plt.plot(self.returns)
#         plt.ylabel('Pct Return', fontsize=16)
#         plt.title('DIS Returns', fontsize=20)
        
#         plot_pacf(self.returns**2)
# #         plt.show()
    def garch_mle(self,params):
    #specifying model parameters
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        beta = params[3]
        #calculating long-run volatility
        long_run = (omega/(1 - alpha - beta))**(1/2)
        #calculating realised and conditional volatility
        resid = self.ret - mu
        realised = abs(resid)
        conditional = np.zeros(len(self.ret))
        conditional[0] =  long_run
        for t in range(1,len(self.ret)):
            conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)
        #calculating log-likelihood
        likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-realised**2/(2*conditional**2))
        log_likelihood = np.sum(np.log(likelihood))
        return -log_likelihood
    def ADF_test(self):
        result = adfuller(self.returns, autolag='AIC')
        print(f'ADF Statistic: {result[0]}')
        print(f'n_lags: {result[1]}')
        print(f'p-value: {result[1]}')
        for key, value in result[4].items():
            print('Critial Values:')
            print(f'   {key}, {value}')    
    def plot_ret(self):
        get_ipython().run_line_magic('matplotlib', 'inline')
        fig, axes = plt.subplots(figsize=(10,7))
        plt.plot(self.returns);
        plt.title('returns');
        
    def plot_hist_returns(self):
        fig = plt.figure()
        ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
        ax1.hist(self.returns.iloc[:,0], bins = 60)
        ax1.set_xlabel('asset returns')
        ax1.set_ylabel("Freq")
        ax1.set_title("histogram ")
        plt.show();
    
            
    def garch(self):
        res = spop.minimize(self.garch_mle, [np.average(self.ret), np.std(self.ret)**2, 0, 0], method='Nelder-Mead')
        params = res.x
        mu = res.x[0]
        omega = res.x[1]
        alpha = res.x[2]
        beta = res.x[3]
        log_likelihood = -float(res.fun)
        #calculating realised and conditional volatility for optimal parameters
        long_run = (omega/(1 - alpha - beta))**(1/2)
        resid = self.ret - mu
        
        realised = abs(resid)
        conditional = np.zeros(len(self.returns))
        conditional[0] =  long_run
        for t in range(1,len(self.ret)):
            conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)
         #printing optimal parameters
        print('GARCH model parameters')
        print('')
        print('mu '+str(round(mu, 6)))
        print('omega '+str(round(omega, 6)))
        print('alpha '+str(round(alpha, 4)))
        print('beta '+str(round(beta, 4)))
        print('long-run volatility '+str(round(long_run, 4)))
        print('log-likelihood '+str(round(log_likelihood, 4)))
         #visualising the results
        plt.figure(1)
        plt.rc('xtick', labelsize = 10)
        plt.plot(self.data.index[1:],realised)
        plt.plot(self.data.index[1:],conditional)
        plt.show()
        self.garh_variance=conditional[len(self.ret)-1]

    def garch_var(self,last_obs,start_obs):
        am = arch_model(self.returns, vol="Garch", p=1, o=0, q=1, dist="Normal")
        res = am.fit(disp="off", last_obs=last_obs)
        forecasts = res.forecast(start=start_obs, reindex=False)
        cond_mean = forecasts.mean[start_obs:]
        cond_var = forecasts.variance[start_obs:]
        q = am.distribution.ppf([0.01, 0.05])
        garch_return=(-cond_mean.values - np.sqrt(cond_var).values *q[None, :] )
        self.garch_ret = pd.DataFrame(garch_return, columns=["1%", "5%"], index=cond_var.index)

        value_at_risk = (-cond_mean.values - np.sqrt(cond_var).values * q[None, :])
        self.value_at_risk = pd.DataFrame(value_at_risk, columns=["1%", "5%"], index=cond_var.index)

       
    
        ax = self.value_at_risk.plot(legend=False)
        xl = ax.set_xlim(self.value_at_risk.index[0], self.value_at_risk.index[-1])
        ax.set_title("Parametric VaR")

        

