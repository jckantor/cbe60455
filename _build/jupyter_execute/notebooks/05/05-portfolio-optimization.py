#!/usr/bin/env python
# coding: utf-8

# <!--NOTEBOOK_HEADER-->
# *This notebook contains course material from [CBE40455](https://jckantor.github.io/CBE40455) by
# Jeffrey Kantor (jeff at nd.edu); the content is available [on Github](https://github.com/jckantor/CBE40455.git).
# The text is released under the [CC-BY-NC-ND-4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode),
# and code is released under the [MIT license](https://opensource.org/licenses/MIT).*

# <!--NAVIGATION-->
# < [Real Options](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/07.04-Real-Options.ipynb) | [Contents](toc.ipynb) | [Portfolio Optimization using Mean Absolute Deviation](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/07.06-Portfolio-Optimization-using-Mean-Absolute-Deviation.ipynb) ><p><a href="https://colab.research.google.com/github/jckantor/CBE40455/blob/master/notebooks/07.05-Portfolio-Optimization.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><p><a href="https://raw.githubusercontent.com/jckantor/CBE40455/master/notebooks/07.05-Portfolio-Optimization.ipynb"><img align="left" src="https://img.shields.io/badge/Github-Download-blue.svg" alt="Download" title="Download Notebook"></a>

# # Portfolio Optimization

# This [Jupyter notebook](http://jupyter.org/) demonstrates the formulation and solution of portfolio optimization problems.

# ## Import libraries

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import random

# data wrangling libraries
import pandas_datareader.data as web
import pandas as pd
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Price Data
# 
# Here we create several groups of possible trading symbols, and select one for study.

# In[2]:


# Companies of the Dow Jones Industrial Average
djia = ['mmm','axp','aapl','ba','cat','cvx','csco','ko','dis','dwdp',
        'xom','ge','gs','hd','ibm','intc','jnj','jpm','mcd','mrk',
        'msft','nke','pfe','pg','trv','utx','unh','vz','v','wmt']

# Representative ETF's and ETN's from the energy sector
energy = ['oil','uso','uga','uhn','kol','ung']

# Standard indices
indices = ['^dji','^gspc']

# pick one of these groups for study
syms = djia
print(syms)


# Price data is retrieved using the Pandas DataReader function. The first price data set consists of historical data that will be used to design investment portfolios. The second set will be used for simulation and evaluation of resulting portfolios. The use of 'out-of-sample' data for evaluation is standard practice in these types of applications.

# In[3]:


t2 = datetime.datetime.now()
t1 = t2 - datetime.timedelta(365)
t0 = t1 - datetime.timedelta(2*365)

Prices = pd.DataFrame()
PricesSim = pd.DataFrame()
for s in syms:
    print(s,end='')
    k = 3
    while k > 0:
        try:
            Prices[s] = web.DataReader(s,'yahoo',t0,t2)['Adj Close']
            print(' ok.')
            k = 0
        except:
            k -= 1
            print(' fail. Try again.',end='')


# ## Plot Price and Returns Data

# In[7]:


idx= Prices.index
Prices.loc[idx[1],'axp']


# In[52]:


Prices.plot(figsize=(10,6))

plt.title('Prices for Selected Goods')
plt.ylabel('Price')
plt.legend(loc='upper right',bbox_to_anchor=(1.2, 1.0))
plt.grid()

Returns = Prices.diff()/Prices.shift(1)
Returns.dropna(inplace=True)

Returns.plot(figsize=(10,6))
plt.title('Linear Returns (daily)')
plt.legend(loc='upper right',bbox_to_anchor=(1.2, 1.0))
plt.grid()


# In[11]:


Prices.loc[idx[1],'axp']
Prices.iloc[0,0]


# ## Linear and Log Returns

# In[53]:


rlin = Returns.mean()
stdev = Returns.std()

rlog = np.log(Prices).diff().mean()


# In[54]:


plt.figure(figsize=(10,6))

plt.subplot(2,1,2)
(252*rlin).plot(kind='bar')
plt.title('Annualized Return')
plt.grid()

plt.subplot(2,1,1)
(np.sqrt(252)*stdev).plot(kind='bar')
plt.title('Annualized Volatility')
plt.grid()

plt.tight_layout()


# In[55]:


print( "Annualized Returns and Standard Deviation\n")
print( "Symbol     r_lin       r_log    r_lin-s^2/2    stdev")
for s in Returns.columns.values.tolist():
    print("{0:5s}".format(s),end='')
    print("  {0:9.5f}".format(252.0*rlin[s]),end='')
    print("   {0:9.5f}".format(252.0*rlog[s]),end='')
    print("   {0:9.5f}".format(252.0*(rlin[s] - 0.5*stdev[s]**2)),end='')
    print("   {0:9.5f}".format(np.sqrt(252.0)*stdev[s]))


# ## Covariance and Correlation Matrices
# 
# The covariance matrix is easily computed using the pandas `.cov` and `.corr()` functions.

# In[56]:


sigma = Returns.cov()
rho = Returns.corr()

pd.options.display.float_format = '{:,.6f}'.format

print("\nCovariance Matrix\n")
print(sigma)

print("\nCorrelation Coefficients\n")
print(rho)


# ### Visualizing the correlation coefficients

# In[57]:


syms = Prices.columns
N = len(syms)
plt.figure(figsize=(10,10))
for i in range(0,N):
    for j in range(0,N):
        plt.subplot(N,N,i*N+j+1)
        gca = plt.plot(Returns[syms[i]],Returns[syms[j]],'.',ms=1,alpha=0.5)
        gca[0].axes.spines['top'].set_visible(False)
        gca[0].axes.spines['right'].set_visible(False)
        gca[0].axes.get_xaxis().set_ticks([])
        gca[0].axes.get_yaxis().set_ticks([])
        plt.xlabel(syms[i])
        plt.ylabel(syms[j])
        plt.axis('square')
        plt.title(str(round(rho[syms[i]][syms[j]],3)))
plt.tight_layout()


# In[58]:


plt.imshow(rho,cmap='hot',interpolation='nearest')


# # Return versus Volatility

# In[59]:


plt.figure(figsize=(8,5))
for s in Returns.columns.values.tolist():
    plt.plot(np.sqrt(252.0)*stdev[s],252*rlin[s],'r.',ms=15)
    plt.text(np.sqrt(252.0)*stdev[s],252*rlin[s],"  {0:5<s}".format(s))

plt.xlim(0.0,plt.xlim()[1])
plt.title('Linear Return versus stdev')
plt.xlabel('Annualized Standard Deviation')
plt.ylabel('Annualized Linear Return')
plt.grid()


# ## Creating Portfolios
# 
# A portfolio is created by allocating current wealth among a collection of assets. Let $w_n$ be the fraction of wealth allocated to the $n^{th}$ asset from a set of $N$ assets. Then
# 
# $$\sum_{n=1}^N w_n = 1$$
# 
# It may be possible to borrow assets, in which case the corresponding $w_n$ is negative. A _long only_ portfolio is one such that all of weights $w_n$ are greater than or equal to zero.

# ### Mean Return and Variance of a Portfolio
# 
# Denoting the total value of the portfolio as V, the value invested in asset $n$ is $w_nV$. At a price $S_n$, the number of units of the asset is $\frac{w_nV}{S_n}$ for a total value
# 
# $$ V = \sum_{n=1}^N \frac{w_n V}{S_n}S_n $$
# 
# at the start of the investment period. The value of the investment at the end of the period, $V'$, is given by
# 
# $$ V' = \sum_{n=1}^N \frac{w_n V}{S_n}(1+r^{lin}_n)S_n $$
# 
# Taking differences we find
# 
# $$ V' - V = \sum_{n=1}^N \frac{w_n V}{S_n}r^{lin}_nS_n $$
# 
# which, after division, yields the return on the portfolio
# 
# $$ r^{lin}_p = \frac{V'-V}{V} = \sum_{n=1}^N w_nr^{lin}_n $$
# 
# Taking expectations, the mean return of the portfolio
# 
# $$\boxed{\bar{r}^{lin}_{p} = \sum_{n=1}^N w_n \bar{r}^{lin}_n}$$
# 
# Variance of portfolio return
# 
# $$\boxed{\sigma_{p}^2 = \sum_{m=1}^N\sum_{n=1}^N w_m w_n \sigma_{mn}}$$

# ## Examples of Portfolios with Two Assets

# In[60]:


N_examples = 4

for i in range(0,N_examples):
    a,b = random.sample(Prices.columns.values.tolist(),2)

    plt.figure()
    for w in np.linspace(0.0,1.0,100):
        V = w*Prices[a] + (1-w)*Prices[b]
        returnsV = (V.diff()/V.shift(1))
        rV = returnsV.mean()
        sV = returnsV.std()
        plt.plot(np.sqrt(252.0)*sV,252.0*rV,'g.',ms=5)
            
    for s in (a,b):
        plt.plot(np.sqrt(252.0)*stdev[s],252*rlin[s],'r.',ms=15)
        plt.text(np.sqrt(252.0)*stdev[s],252*rlin[s],"  {0:5<s}".format(s))

    plt.xlim(0.0,np.sqrt(252.0)*stdev.max())
    plt.ylim(252.0*rlin.min(),252.0*rlin.max())

    plt.title('Linear Return versus stdev for selected assets')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Linear Return')
    plt.grid()


# ## Minimum Risk Portfolio
# 
# The minimum variance portfolio is found as a solution to 
# $$\min_{w_1, w_2, \ldots, w_N} \sum_{m=1}^N\sum_{n=1}^N w_m w_n\sigma_{mn}$$
# subject to 
# $$\sum_{n=1}^N w_n = 1$$

# ### Pyomo Model and Solution

# In[81]:


from pyomo.environ import *

# data
N = len(Returns.columns.values.tolist())
STOCKS = Returns.columns

# pyomo model
m = ConcreteModel()
m.w = Var(STOCKS,domain=Reals)

# expression for objective function
portfolio_var = sum(m.w[i]*sigma.loc[i,j]*m.w[j] for i in STOCKS for j in STOCKS)
portfolio_ret = sum(m.w[i]*rlin[i] for i in STOCKS)
m.OBJ = Objective(expr=portfolio_var, sense=minimize)

# constraints
m.cons = ConstraintList()
m.cons.add(sum(m.w[i] for i in STOCKS) == 1)

# solve
SolverFactory('gurobi').solve(m)

# display solutions
w = pd.Series([m.w[i]() for i in STOCKS], STOCKS)
w.plot(kind='bar')
plt.xlabel('Symbol')
plt.title('Weights for a minimum variance portfolio')
plt.grid()


# In[82]:


plt.figure(figsize=(8,5))
for s in STOCKS:
    plt.plot(np.sqrt(252.0)*stdev[s],252*rlin[s],'r.',ms=15)
    plt.text(np.sqrt(252.0)*stdev[s],252*rlin[s],"  {0:5<s}".format(s))

plt.plot(np.sqrt(252.0)*np.sqrt(portfolio_var()),252*portfolio_ret(),'b.',ms=15)

plt.xlim(0.0,plt.xlim()[1])
plt.title('Linear Return versus stdev')
plt.xlabel('Annualized Standard Deviation')
plt.ylabel('Annualized Linear Return')
plt.grid()


# ### Out-of-Sample Simulation

# In[92]:


PricesSim['jpm']


# In[89]:


V = pd.Series(0,index=PricesSim.index)
for s in STOCKS:
    print(s)
    V += (100.0*float(w[s])/PricesSim[s][0])*PricesSim[s]
    
V.plot(lw=5,figsize=(10,6),color='red')
for s in STOCKS:
    S = pd.Series(100.0*PricesSim[s]/PricesSim[s][0])
    S.plot(lw=0.8)

plt.legend(loc='upper right',bbox_to_anchor=(1.2, 1.0))
plt.title('Portfolio Dynamics w/o Rebalancing')
plt.ylabel('Normalized Price/Value')
plt.grid()


# In[49]:


plt.figure(figsize=(12,8))
for s in Returns.columns.values.tolist():
    plt.plot(np.sqrt(252.0)*stdev[s],252*rlin[s],'r.',ms=15)
    plt.text(np.sqrt(252.0)*stdev[s],252*rlin[s],"  {0:5<s}".format(s))

plt.xlim(0.0,plt.xlim()[1])
plt.title('Linear Return versus stdev for selected assets')
plt.xlabel('Standard Deviation')
plt.ylabel('Linear Return')
plt.grid()

ReturnsV = (V.diff()/V.shift(1))
rlinMinVar = ReturnsV.mean()
stdevMinVar = ReturnsV.std()
plt.plot(np.sqrt(252.0)*stdevMinVar,252.0*rlinMinVar,'g.',ms=30)
plt.text(np.sqrt(252.0)*stdevMinVar,252.0*rlinMinVar,'   Min. Var. Portfolio')


# ## Markowitz Portfolio
# 
# The minimum variance portfolio does a good job of handling volatility, but at the expense of relatively low return. The Markowitz portfolio adds an additional constraint to specify mean portfolio return. 
# $$\min_{w_1, w_2, \ldots, w_N} \sum_{m=1}^N\sum_{n=1}^N w_m w_n\sigma_{mn}$$
# subject to 
# $$\sum_{n=1}^N w_n = 1$$
# and
# $$\sum_{n=1}^N w_n \bar{r}^{lin}_n = \bar{r}^{lin}_p$$
# By programming the solution of the optimization problem as a function of $\bar{r}_p$, it is possible to create a risk-return tradeoff curve.

# In[51]:


# cvxpy problem description
w = cvx.Variable(N)
r = cvx.Parameter()

risk = cvx.quad_form(w, np.array(sigma))
prob = cvx.Problem(cvx.Minimize(risk), 
               [cvx.sum_entries(w) == 1, 
                np.array(rlin).T*w == r,
               w >= 0])

# lists to store results of parameter scans
r_data = []
stdev_data = []
w_data = []

# scan solution as function of portfolio return
for rp in np.linspace(rlin.min(),3.0*rlin.max(),100):
    r.value = rp
    s = prob.solve()
    if prob.status == "optimal":
        r_data.append(rp)
        stdev_data.append(np.sqrt(prob.solve()))
        w_data.append([u[0,0] for u in w.value])

plt.figure(figsize=(8,10))
plt.subplot(211)

# plot tradeoff curve
plt.plot(np.sqrt(252.0)*np.array(stdev_data), 252.0*np.array(r_data), lw=3)
for s in syms:
    plt.plot(np.sqrt(252.0)*stdev[s],252.0*rlin[s],'r.',ms=15)
    plt.text(np.sqrt(252.0)*stdev[s],252.0*rlin[s],"  {0:5<s}".format(s))

plt.plot(np.sqrt(252.0)*stdevMinVar,252.0*rlinMinVar,'g.',ms=30)
plt.text(np.sqrt(252.0)*stdevMinVar,252.0*rlinMinVar,'   Min. Var. Portfolio')

plt.xlim(0.0,plt.xlim()[1])
plt.title('Linear Return versus stdev for Markowitz Portfolio')
plt.xlabel('Standard Deviation')
plt.ylabel('Linear Return')
plt.grid()


# In[82]:


# plot portfolio weights
w_data = np.array(w_data)
for i in range(N):
    plt.plot(r_data,w_data.T[i])


# ### Out of Sample Simulation

# In[46]:


end = datetime.datetime.now()
start = end - datetime.timedelta(3*365)

PricesSim = pd.DataFrame()
for s in syms:
    PricesSim[s] = web.DataReader(s,'yahoo',start,end)['Adj Close']


# In[48]:


V = pd.Series((100.0*float(w.value[0])/PricesSim.ix[0,0])*PricesSim.ix[:,0])
for n in range(1,N):
    V += (100.0*float(w.value[n])/PricesSim.ix[0,n])*PricesSim.ix[:,n]

V.plot(lw=5,figsize=(10,6))
plt.hold(True)
for s in PricesSim.columns.values.tolist():
    S = pd.Series(100.0*PricesSim[s]/PricesSim[s][0])
    S.plot(lw=0.5)
V.plot(lw=5,color='red')

plt.title('Portfolio Dynamics w/o Rebalancing')
plt.ylabel('Normalized Price/Value')
plt.grid()


# ## Risk-Free Asset
# 
# The minimum variance portfolio does a good job of handling volatility, but at the expense of relatively low return. The Markowitz portfolio adds an additional constraint to specify mean portfolio return. 
# $$\min_{w_1, w_2, \ldots, w_N} \sum_{m=1}^N\sum_{n=1}^N w_m w_n\sigma_{mn}$$
# subject to 
# $$\sum_{n=1}^N w_n = 1 - w_f$$
# and
# $$\sum_{n=1}^N w_n \bar{r}^{lin}_n + w_f r_f = \bar{r}^{lin}_p$$
# By programming the solution of the optimization problem as a function of $\bar{r}_p$, it is possible to create a risk-return tradeoff curve.

# In[54]:


# cvxpy problem description

m = ConcreteModel()

m.w = Var(range(0,N), domain = Reals)
wf


# In[107]:


w = cvx.Variable(N)
wf = cvx.Variable(1)
r = cvx.Parameter()

rf = 0.01/252.0

risk = cvx.quad_form(w, np.array(sigma))
prob = cvx.Problem(cvx.Minimize(risk), 
               [cvx.sum_entries(w) == 1 - wf, 
                np.array(rlin).T*w + wf*rf == r,
               w >= 0])

# lists to store results of parameter scans
r_data_rf = []
stdev_data_rf = []
w_data_rf = []

# scan solution as function of portfolio return
for rp in np.linspace(np.min(rf,rlin.min()),1.5*rlin.max(),100):
    r.value = rp
    s = prob.solve()
    if prob.status == "optimal":
        r_data_rf.append(rp)
        stdev_data_rf.append(np.sqrt(prob.solve()))
        w_data_rf.append([u[0,0] for u in w.value])

plt.figure(figsize=(10,8))
plt.subplot(211)
plt.plot(np.sqrt(252.0)*np.array(stdev_data_rf), 252.0*np.array(r_data_rf),lw=3)
plt.plot(np.sqrt(252.0)*np.array(stdev_data), 252.0*np.array(r_data),'g--')
for s in syms:
    plt.plot(np.sqrt(252.0)*stdev[s],252.0*rlin[s],'r.',ms=15)
    plt.text(np.sqrt(252.0)*stdev[s],252.0*rlin[s],"  {0:5<s}".format(s))
    
# find max Sharpe ratio
k = np.argmax(np.divide(np.array(r_data)-rf,np.array(stdev_data)))
smax = stdev_data[k]
rmax = r_data[k]
plt.plot(np.sqrt(252.0)*smax,252.0*rmax,'o',ms=10)
print 252.0*rmax
print 252.0*rf
print np.sqrt(252.0)*smax

plt.xlim(0.0,plt.xlim()[1])
plt.grid()

plt.subplot(212)

wval = pd.Series(w_data[k],syms)
wval.plot(kind='bar')

plt.xlabel('Symbol')
plt.title('Weights for a minimum variance portfolio')
plt.grid()


# In[ ]:





# In[85]:


N = len(Prices.columns.values.tolist())

# create price history of the portfolio
V = pd.Series(0,index=Prices.index,name='Market')
for s in Prices.columns.values.tolist():
    V += (100.0*float(wval[s])/Prices[s][0])*Prices[s]
    
V.plot(lw=5,figsize=(10,6),color='red')

# plot components
plt.hold(True)
for s in Prices.columns.values.tolist():
    S = pd.Series(100.0*Prices[s]/Prices[s][0])
    S.plot(lw=0.8)

plt.legend(loc='upper right',bbox_to_anchor=(1.2, 1.0))
plt.title('Portfolio Dynamics w/o Rebalancing')
plt.ylabel('Normalized Price/Value')
plt.grid()


# In[86]:


P['Market'] = V
Sigma = P.cov()
Sigma.ix[:,'Market']/Sigma.ix['Market','Market']


# ## Maximum Log Return (to be developed).

# In[353]:


import cvxpy as cvx

N = len(syms)

w = cvx.Variable(N)
risk = cvx.quad_form(w, np.array(sigma))
prob = cvx.Problem(cvx.Maximize(np.array(rlin).T*w-0.5*risk), 
               [cvx.sum_entries(w) == 1,
                risk <= 0.0005/np.sqrt(252.0),
                w>=0])

prob.solve()
print prob.status

plt.figure(figsize=(12,8))
for s in Returns.columns.values.tolist():
    plt.plot(np.sqrt(252.0)*stdev[s],252*rlin[s],'r.',ms=15)
    plt.text(np.sqrt(252.0)*stdev[s],252*rlin[s],"  {0:5<s}".format(s))

plt.xlim(0.0,plt.xlim()[1])
plt.title('Linear Return versus stdev for selected assets')
plt.xlabel('Standard Deviation')
plt.ylabel('Linear Return')
plt.grid()
plt.hold(True)

plt.plot(np.sqrt(252.0)*stdevMinVar,252.0*rlinMinVar,'g.',ms=30)
plt.text(np.sqrt(252.0)*stdevMinVar,252.0*rlinMinVar,'   Min. Var. Portfolio')

wval = np.array(w.value)[:,0].T
rlog = np.dot(np.array(rlin).T,wval)
stdevlog = np.sqrt(np.dot(np.dot(wval,np.array(sigma)),wval))

plt.plot(np.sqrt(252.0)*stdevlog,252.0*rlog,'g.',ms=30)
plt.text(np.sqrt(252.0)*stdevlog,252.0*rlog,'  Max. Return Portfolio')


# ## Exercises
# 
# 1. Modify the minimum variance portfolio to be 'long-only', and add a diversification constraints that no more than 20% of the portfolio can be invested in any single asset.  How much does this change the annualized return?  the annualized standard deviation?
# 
# 2. Modify the Markowitz portfolio to allow long-only positions and a diversification constraint that no more than 20% of the portfolio can be invested in any single asset. What conclusion to you draw?

# In[ ]:





# In[ ]:





# <!--NAVIGATION-->
# < [Real Options](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/07.04-Real-Options.ipynb) | [Contents](toc.ipynb) | [Portfolio Optimization using Mean Absolute Deviation](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/07.06-Portfolio-Optimization-using-Mean-Absolute-Deviation.ipynb) ><p><a href="https://colab.research.google.com/github/jckantor/CBE40455/blob/master/notebooks/07.05-Portfolio-Optimization.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><p><a href="https://raw.githubusercontent.com/jckantor/CBE40455/master/notebooks/07.05-Portfolio-Optimization.ipynb"><img align="left" src="https://img.shields.io/badge/Github-Download-blue.svg" alt="Download" title="Download Notebook"></a>
