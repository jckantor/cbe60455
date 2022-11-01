#!/usr/bin/env python
# coding: utf-8

# # Solving the Farmer's problem
# 
# The [Farmer's Problem](https://www.math.uh.edu/~rohop/Spring_15/Chapter1.pdf) is a teaching example presented in the well-known textbook by John Birge and Francois Louveaux.
# 
# * Birge, John R., and Francois Louveaux. Introduction to stochastic programming. Springer Science & Business Media, 2011.

# In[1]:


# install Pyomo and solvers
import requests
import types

url = "https://raw.githubusercontent.com/mobook/MO-book/main/python/helper.py"
helper = types.ModuleType("helper")
exec(requests.get(url).content, helper.__dict__)

helper.install_pyomo()
helper.install_glpk()
helper.install_cbc()
helper.install_ipopt()


# In[2]:


import pyomo.environ as pyo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Problem Statement
# 
# In the [farmer's problem](https://www.math.uh.edu/~rohop/Spring_15/Chapter1.pdf), a European farmer has to allocate 500 acres of land to three different crops (wheat, corn, and sugar beets) aiming to maximize profit. 
# 
# * Planting one acre of wheat, corn and beet costs \\$150, \\$230 and \\$260, respectively.
# 
# * The mean yields are 2.5, 3.0, and 20.0 tons per acre for wheat, corn, and sugar beets, respectively. However, the yields can vary up to 25% from nominal conditions depending on weather.
# 
# * At least 200 tons of wheat and 240 tons of corn are needed for cattle feed. These can be raised on the farm or purchased from a wholesaler. 
# 
# * Over the last decade, mean selling prices have been \\$170 and \\$150 per ton of wheat and corn, respectively. The purchase prices are 40% more due to wholesaler's margins and transportation costs.
# 
# * Sugar beets are a profitable crop expected to sell at \\$36 per ton, but there is a quota on sugar beet production. Any amount in excess of the quota can be sold at only \\$10 per ton. The farmer's  quota for next year is 6,000 tons.
# 
# After collecting this data, the farmer is unsure how to allocate the land among the three crops. So the farmer has hired you as a consultant to develop a model. After interviewing the farmer, you have determined you need to present three solutions for the farmer to consider:
# 
# 1. The first solution should represent the mean solution. How should the farmer allocate land to maximize profits under mean conditions.
# 
# 2. The second solution should consider the potential impact of weather. How should the farmer allocate land to maximize expected profit if the yields could go up or down by 20% due to weather conditions? What is the profit under each scenario?
# 
# 3. During your interview you learned the farmer needs a minimal profit each year to stay in business. How would you allocate land use to maximize the worst case profit?  
# 
# 4. Determine the tradeoff between risk and return by computing the mean expected profit when the minimum required profit is the worst case found in part 3, and \\$58,000, \\$56,000, \\$54,000, \\$52,000, \\$50,000, and \\$48,000. Compare these solutions to part 2 by plotting the expected loss in profit. 
# 
# 5. What would be your advice to the farmer regarding land allocation?

# ## Data Summary
# 
# | Scenario | Yield for wheat <br> (tons/acre)| Yield for corn <br> (tons/acre) | Yield for beets <br> (tons/acre) |
# | :-- | :-: | :-: | :-: |
# | Good weather | 3 | 3.6 | 24 |
# | Average weather | 2.5 | 3 | 20 |
# | Bad weather | 2 | 2.4 | 16 |
# 
# We first consider the case in which all the prices are fixed and not weather-dependent. The following table summarizes the data.
# 
# | Commodity | Sell Price <br> (euro/ton) | Market <br> Demand <br> (tons) | Purchase <br> Price <br> (euro/ton) | Cattle Feed <br> Required <br> (tons) | Planting <br> Cost <br> (euro/acre) |
# | :-- | :--: | :--: | :--: | :--: | :--: |
# | Wheat | 170 | - | 238 | 200 | 150 |
# | Corn | 150 | - | 210 | 240 | 230 |
# | Beets | 36 | 6000 | - | 0 | 260 | 6000 |
# | Beets extra | 10 | - | - | 0 | 260 |
# 
# (a) Implement the extensive form of stochastic LP corresponding to the farmer's problem in Pyomo and solve it.
