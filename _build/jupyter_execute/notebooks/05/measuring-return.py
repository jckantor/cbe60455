#!/usr/bin/env python
# coding: utf-8

# # Measuring Return

# ## Imports
# 
# The pandas_datareader library provides convenient access to financial data. The library requires separate installation by executing the command
# 
#     pip install pandas-datareader
# 
# in a terminal window, or executign
# 
#     !pip install pandas-datareader --upgrade
# 
# in a Jupyter notebook cell.

# In[34]:


get_ipython().system('pip install pandas-datareader --upgrade')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import datetime
import pandas_datareader as pdr


# The following function provides a convenient stock symbol lookup function.

# ## Where to get Price Data
# 
# Price data is available from a number of sources. Here we demonstrate the process of obtaining price data on financial goods from [Yahoo Finance](http://finance.yahoo.com/). Additional sources are documented [here](https://pandas-datareader.readthedocs.io/en/latest/readers/index.html).

# ### Yahoo Finance
# 
# [Yahoo Finance](http://finance.yahoo.com/) provides historical Open, High, Low, Close, and Volume date for quotes on traded securities. In addition, Yahoo Finance provides historical [Adjusted Close](http://marubozu.blogspot.com/2006/09/how-yahoo-calculates-adjusted-closing.html) price data that corrects for splits and dividend distributions. The Adjusted Close is a useful tool for computing the return on long-term investments.
# 
# The following cell demonstrates how to download historical Adjusted Close price for a selected security into a pandas DataFrame.

# In[ ]:


symbol = 'AAPL'

# end date is today
end = datetime.datetime.today().date()

# start date is three years prior
start = end-datetime.timedelta(3*365)

# get stock price data
S = pdr.data.DataReader(symbol, "yahoo", start, end)['Adj Close']

# plot data
plt.figure(figsize=(10,4))
S.plot(title=symbol, grid=True, ylabel="Adjusted Close")


# ### Quandl
# 
# [Quandl](http://www.quandl.com/) is a searchable source of time-series data on a wide range of commodities, financials, and many other economic and social indicators. Data from Quandl can be downloaded as files in various formats, or accessed directly using the [Quandl API](http://www.quandl.com/help/api) or software-specific package. Here we use demonstrate use of the [Quandl Python package](http://www.quandl.com/help/packages#Python). 
# 
# The first step is execute a system command to check that the Quandl package has been installed.
# 
# Here are examples of energy datasets. These were found by searching Quandl, then identifying the Quandl code used for accessing the dataset, a description, the name of the field containing the desired price information.

# In[ ]:


code = 'CHRIS/MCX_CL1'
description = 'NYMEX Crude Oil Futures, Continuous Contract #1 (CL1) (Front Month)'
field = 'Close'


# In[ ]:


end = datetime.datetime.today().date()
start = end - datetime.timedelta(5*365)

S = pdr.quandl.QuandlReader(code, start, end)

plt.figure(figsize=(10,4))
S.plot()
plt.title(description)
plt.ylabel('Price $/bbl')
plt.grid()


# ## Returns
# 
# The statistical properties of financial series are usually studied in terms of the change in prices. There are several reasons for this, key among them is that the changes can often be closely approximated as stationary random variables whereas prices are generally non-stationary sequences. 
# 
# A common model is 
# 
# $$S_{t} = R_{t} S_{t-1}$$
# 
# so, recursively,
# 
# $$S_{t} = R_{t} R_{t-1} \cdots R_{0} S_{0}$$
# 
# The gross return $R_t$ is simply the ratio of the current price to the previous, i.e.,
# 
# $$R_t = \frac{S_t}{S_{t-1}}$$
# 
# $R_t$ will typically be a number close to one in value, greater than one for an appreciating asset, or less than one for an asset with declining price.

# In[ ]:


symbol = 'AAPL'

# end date is today
end = datetime.datetime.today().date()

# start date is three years prior
start = end-datetime.timedelta(3*365)

# get stock price data
S = pdr.data.DataReader(symbol,"yahoo", start, end)['Adj Close']
R = S/S.shift(1)

# plot data
plt.figure(figsize=(10, 5))
plt.subplot(2,1,1)
S.plot(title=symbol, ylabel="Adjusted Close", grid=True)

plt.subplot(2, 1, 2)
R.plot(ylabel="Returns", grid=True)
plt.tight_layout()


# ### Linear fractional or Arithmetic Returns
# 
# Perhaps the most common way of reporting returns is simply the fractional increase in value of an asset over a period, i.e.,
# 
# $$r^{lin}_t = \frac{S_t - S_{t-1}}{S_{t-1}} = \frac{S_t}{S_{t-1}} - 1 $$
# 
# Obviously
# 
# $$r^{lin}_t = R_t  - 1$$

# In[ ]:


symbol = 'AAPL'

# end date is today
end = datetime.datetime.today().date()

# start date is three years prior
start = end-datetime.timedelta(3*365)

# get stock price data
S = data.DataReader(symbol,"yahoo",start,end)['Adj Close']
rlin = S/S.shift(1) - 1

# plot data
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
S.plot(title=get_symbol(symbol))
plt.ylabel('Adjusted Close')
plt.grid()

plt.subplot(2,1,2)
rlin.plot()
plt.title('Linear Returns (daily)')
plt.grid()
plt.tight_layout()


# ### Linear returns don't tell the whole story.
# 
# Suppose you put money in an asset that returns 10% interest in even numbered years, but loses 10% in odd numbered years. Is this a good investment for the long-haul?
# 
# If we look at mean linear return
# 
# \begin{align}
# \bar{r}^{lin} & = \frac{1}{T}\sum_{t=1}{T} r^{lin}_t  \\
# & = \frac{1}{T} (0.1 - 0.1 + 0.1 - 0.1 + \cdots) \\
# & = 0
# \end{align}
# 
# we would conclude this asset, on average, offers zero return. What does a simulation show?

# In[ ]:


S = 100
log = [[0,S]]
r = 0.10

for k in range(1,101):
    S = S + r*S
    r = -r
    log.append([k,S])
    
df = pd.DataFrame(log,columns = ['k','S'])
plt.plot(df['k'],df['S'])
plt.xlabel('Year')
plt.ylabel('Value')


# Despite an average linear return of zero, what we observe over time is an asset declining in price.  The reason is pretty obvious --- on average, the years in which the asset loses money have higher balances than years where the asset gains value.  Consequently, the losses are somewhat greater than the gains which, over time, leads to a loss of value.
# 
# Here's a real-world example of this phenomenon. For a three year period ending October 24, 2017, United States Steel (stock symbol 'X') offers an annualized linear return of 15.9%. Seems like a terrific investment opportunity, doesn't it?  Would you be surprised to learn that the actual value of the stock fell 18.3% over that three-year period period?
# 
# What we can conclude from these examples is that average linear return, by itself, does not provide us with the information needed for long-term investing.

# In[ ]:


symbol = 'X'

# end date is today
end = datetime.datetime(2017, 10, 24)

# start date is three years prior
start = end-datetime.timedelta(3*365)

# get stock price data
S = pdr.data.DataReader(symbol, "yahoo", start, end)['Adj Close']
rlin = S/S.shift(1) - 1

print('Three year return :', 100*(S[-1]-S[0])/S[0], '%')

# plot data
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
S.plot(title=symbol, ylabel="Adjusted Close", grid=True)


plt.subplot(2, 1, 2)
rlin.plot(title=f'Mean Linear Returns (annualized) = {0:.2f}%'.format(100*252*rlin.mean()))
plt.grid()
plt.tight_layout()


# ### Compounded Log Returns
# 
# Compounded, or log returns, are defined as
# 
# $$r^{log}_{t} = \log R_t = \log \frac{S_{t}}{S_{t-1}}$$
# 
# The log returns have a very useful compounding property for aggregating price changes across time
# 
# $$ \log \frac{S_{t+k}}{S_{t}} = r^{log}_{t+1} + r^{log}_{t+2} + \cdots + r^{log}_{t+k}$$
# 
# If the compounded returns are statistically independent and identically distributed, then this property provides a means to aggregate returns and develop statistical price projections.

# In[ ]:


symbol = 'AAPL'

# end date is today
end = datetime.datetime.today().date()

# start date is three years prior
start = end-datetime.timedelta(3*365)

# get stock price data
S = data.DataReader(symbol,"yahoo",start,end)['Adj Close']
rlog = np.log(S/S.shift(1))

# plot data
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
S.plot(title=get_symbol(symbol))
plt.ylabel('Adjusted Close')
plt.grid()

plt.subplot(2,1,2)
rlin.plot()
plt.title('Log Returns (daily)')
plt.grid()
plt.tight_layout()


# ### Volatility Drag and the Relationship between Linear and Log Returns
# 
# For long-term financial decision making, it's important to understand the relationship between $r_t^{log}$ and $r_t^{lin}$. Algebraically, the relationships are simple.
# 
# $$r^{log}_t = \log \left(1+r^{lin}_t\right)$$
# 
# $$r^{lin}_t = e^{r^{log}_t} - 1$$
# 
# The linear return $r_t^{lin}$ is the fraction of value that is earned from an asset in a single period. It is a direct measure of earnings. The average value $\bar{r}^{lin}$ over many periods this gives the average fractional earnings per period. If you care about consuming the earnings from an asset and not about growth in value, then $\bar{r}^{lin}$ is the quantity of interest to you.
# 
# Log return $r_t^{log}$ is the rate of growth in value of an asset over a single period. When averaged over many periods, $\bar{r}^{log}$ measures the compounded rate of growth of value. If you care about the growth in value of an asset, then $\bar{r}^{log}$ is the quantity of interest to you.
# 
# The compounded rate of growth $r_t^{log}$ is generally smaller than average linear return $\bar{r}^{lin}$ due to the effects of volatility. To see this, consider an asset that has a linear return of -50% in period 1, and +100% in period 2. The average linear return is would be +25%, but the compounded growth in value would be 0%.
# 
# A general formula for the relationship between $\bar{r}^{log}$ and $\bar{r}^{lin}$ is derived as follows:
# 
# $$\begin{align*}
# \bar{r}^{log} & = \frac{1}{T}\sum_{t=1}^{T} r_t^{log} \\
# & = \frac{1}{T}\sum_{t=1}^{T} \log\left(1+r_t^{lin}\right) \\
# & = \frac{1}{T}\sum_{t=1}^{T} \left(\log(1) + r_t^{lin} - \frac{1}{2} (r_t^{lin})^2 + \cdots
# \right) \\
# & = \frac{1}{T}\sum_{t=1}^{T} r_t^{lin} - \frac{1}{2}\frac{1}{T}\sum_{t=1}^{T} (r_t^{lin})^2 + \cdots \\
# & = \bar{r}^{lin} - \frac{1}{2}\left(\frac{1}{T}\sum_{t=1}^{T} (r_t^{lin})^2\right) + \cdots \\
# & = \bar{r}^{lin} - \frac{1}{2}\left((\bar{r}^{lin})^2 + \frac{1}{T}\sum_{t=1}^{T} (r_t^{lin}-\bar{r}^{lin})^2\right) + \cdots
# \end{align*}$$
# 
# For typical values $\bar{r}^{lin}$ of and long horizons $T$, this results in a formula
# 
# $$\begin{align*}
# \bar{r}^{log} & \approx \bar{r}^{lin} - \frac{1}{2} \left(\sigma^{lin}\right)^2
# \end{align*}$$
# 
# where $\sigma^{lin}$ is the standard deviation of linear returns, more commonly called the volatility.
# 
# The difference $- \frac{1}{2} \left(\sigma^{lin}\right)^2$ is the _volatility drag_ imposed on the compounded growth in value of an asset due to volatility in linear returns. This can be significant and a source of confusion for many investors. 
# 
# It's indeed possible to have a positive average linear return, but negative compounded growth.  To see this, consider a \$100 investment which earns 20% on even-numbered years, and loses 18% on odd-numbered years. The average linear return is 1%, and the average log return is -0.81%.
# 
# 

# In[ ]:


symbol = 'AAPL'

# end date is today
end = datetime.datetime.today().date()

# start date is three years prior
start = end-datetime.timedelta(3*365)

# get stock price data
S = pdr.data.DataReader(symbol, "yahoo", start, end)['Adj Close']
rlin = (S - S.shift(1))/S.shift(1)
rlog = np.log(S/S.shift(1))

# plot data
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
S.plot(title=symbol)
plt.ylabel('Adjusted Close')
plt.grid()

plt.subplot(3,1,2)
rlin.plot()
plt.title('Linear Returns (daily)')
plt.grid()
plt.tight_layout()

plt.subplot(3,1,3)
rlog.plot()
plt.title('Log Returns (daily)')
plt.grid()
plt.tight_layout()


# In[ ]:


print("Mean Linear Return (rlin) = {0:.7f}".format(rlin.mean()))
print("Linear Volatility (sigma) = {0:.7f}".format(rlin.std()))
print("Volatility Drag -0.5*sigma**2 = {0:.7f}".format(-0.5*rlin.std()**2))
print("rlin - 0.5*vol = {0:.7f}\n".format(rlin.mean() - 0.5*rlin.std()**2))

print("Mean Log Return = {0:.7f}".format(rlog.mean()))


# In[ ]:


symbols = ['AAPL', 'MSFT', 'F', 'XOM', 'GE', 'X']

# end date is today
end = datetime.datetime.today().date()

# start date is three years prior
start = end - datetime.timedelta(3*365)

rlin = []
rlog = []
sigma = []

for symbol in symbols:

    # get stock price data
    S = pdr.data.DataReader(symbol, "yahoo", start, end)['Adj Close']
    r = (S - S.shift(1))/S.shift(1)
    rlin.append(r.mean()) 
    rlog.append((np.log(S/S.shift(1))).mean())
    sigma.append(r.std())
    


# In[ ]:


import seaborn as sns
N = len(symbols)
idx = np.arange(N)
width = 0.2

p0 = plt.bar(idx - 1.25*width, rlin, width)
p1 = plt.bar(idx, -0.5*np.array(sigma)**2, width, bottom=rlin)
p2 = plt.bar(idx + 1.25*width, rlog, width)

for k in range(0,N):
    plt.plot([k - 1.75*width, k + 0.5*width],[rlin[k],rlin[k]],'k',lw=1)
    plt.plot([k - 0.5*width, k + 1.75*width],[rlog[k],rlog[k]],'k',lw=1)
    
plt.xticks(idx,symbols)
plt.legend((p0[0],p1[0],p2[0]),('rlin','0.5*sigma**2','rlog'))
plt.title('Components of Linear Return')


# In[ ]:




