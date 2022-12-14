#!/usr/bin/env python
# coding: utf-8

# # Portfolio Optimization using Mean Absolute Deviation
# 

# This [IPython notebook](http://ipython.org/notebook.html) demonstrates portfolio optimization using the Mean Absolute Deviation (MAD) criterion. A portion of these notes is adapted from [GLPK Wikibook tutorial on the subject](http://en.wikibooks.org/wiki/GLPK/Portfolio_Optimization) written by me.

# ## Background

# Konno and Yamazaki (1990) proposed a linear programming model for portfolio optimization in which the risk measure is mean absolute deviation (MAD). This model computes a portfolio minimizing MAD subject to a lower bound on return.
# 
# In contrast to the classical Markowitz portfolio, the MAD criterion requires a data set consisting of returns on the investment assets. The data set may be an historical record or samples from a multivariate statistical model of portfolio returns.  The MAD criterion produces portfolios with properties not shared by the Markowitz portfolio, including second degree stochastic dominance.
# 
# Below we demonstrate portfolio optimization with the MAD criterion where data is generated by sampling a multivariate normal distribution. Given mean return $r$ and the Cholesky decomposition of the covariance matrix $\Sigma$ (i.e., $C$ such that $CC^T = \Sigma$), we compute $r_t = r + C z_t$ where the elements of $z_t$ are zero mean normal variates with unit variance.
# 
# The rest of the formulation is adapted from "Optimization Methods in Finance" by Gerald Curnuejols and Reha Tutuncu (2007) which, in turn, follows an implementation due to Fienstein and Thapa (1993).  
# 
# [A complete tutorial](http://en.wikibooks.org/wiki/GLPK/Portfolio_Optimization) describing the implementation of this model is available on the [GLPK wikibook](http://en.wikibooks.org/wiki/GLPK/).
# 

# ## Mean Absolute Deviation

# Portfolio optimization refers to the process of allocating capital among a set of financial assets to achieve a desired tradeoff between risk and return. The classical Markowitz approach to this problem measures risk using the expected variance of the portfolio return. This criterion yields a quadratic program for the relative weights of assets in the optimal portfolio. 
# 
# In 1991, Konno and Yamazaki <ref>{{Cite journal | last1 = Konno | first1 = Hiroshi | last2 = Yamazaki | first2 = Hiroaki | title = Mean-absolute deviation portfolio optimization model and its applications to Tokyo stock market | journal = Management Science | volume = 37 | pages = 519-531 | date = 1991}}</ref> proposed a linear programming model for portfolio optimization whereby risk is measured by the mean absolute deviation (MAD) from the expected return.  Using MAD as the risk metric produces portfolios with several desirable properties not shared by the Markowitz portfolio, including [[w:Stochastic_dominance|second order stochastic dominance]].
# 
# As originally formulated by Konno and Yamazaki, one starts with a history of returns $R_i(t_n)$ for every asset in a set $S$ of assets. The return at time $t_n$ is determined by the change in price of the asset, 
# 
# $$R_i(t_n)= {(P_i(t_n)-P_i(t_{n-1}))}/{P_i(t_{n-1})}$$
# 
# For each asset, the expected return is estimated by 
# 
# $$\bar{R}_i \approx \frac{1}{N}\sum_{n=1}^NR_i(t_n)$$
# 
# The investor specifies a minimum required return $R_{p}$.  The portfolio optimization problem is to determine the fraction of the total investment allocated to each asset, $w_i$,  that minimizes the mean absolution deviation from the mean
# 
# $$\min_{w_i} \frac{1}{N}\sum_{n=1}^N\lvert\sum_{i \in S} w_i(R_i(t_n)-\bar{R}_i)\rvert$$
# 
# subject to the required return and a fixed total investment:
# 
# $$
# \begin{array}{rcl}
# \sum_{i \in S} w_i\bar{R}_i & \geq & R_{p}  \\
# \quad \\
# \sum_{i \in S} w_i & = & 1
# \end{array}
# $$
# 
# The value of the minimum required return, $R_p$ expresses the investor's risk tolerance. A smaller value for $R_p$ increases the feasible solution space, resulting in portfolios with lower values of the MAD risk metric. Increasing $R_p$ results in portfolios with higher risk. The relationship between risk and return is a fundamental principle of investment theory. 
# 
# This formulation doesn't place individual bounds on the weights $w_i$. In particular, $w_i < 0$ corresponds to short selling of the associated asset.  Constraints can be added if the investor imposes limits on short-selling, on the maximum amount that can be invested in a single asset, or other requirements for portfolio diversification.  Depending on the set of available investment opportunities, additional constraints can lead to infeasible investment problems.

# ## Reformulation of the MAD Objective

# The following formulation of the objective function is adapted from "Optimization Methods in Finance" by Gerald Curnuejols and Reha Tutuncu (2007) <ref>{{Cite book | last1 = Curnuejols | first1 = Gerald | last2 = Tutuncu | first2 = Reha | title = Optimization Methods in Finance | publisher = Cambridge University Press  | date = 2007}}</ref> which, in turn, follows
# Feinstein and Thapa (1993) <ref>{{Cite journal | last1 = Feinstein | first1 = Charles D. | last2 = Thapa | first2 = Mukund N. | title = A Reformulation of a Mean-Absolute Deviation Portfolio Optimization Model | journal = Management Science | volume = 39 | pages = 1552-1553 | date = 1993}}</ref>. 
# 
# The model is streamlined by introducing decision variables $y_n \geq 0$ and $z_n \geq 0$, $n=1,\ldots,N$ so that the objective becomes
# 
# $$\min_{w_i, y_n,z_n} \frac{1}{N}\sum_{n = 1}^{N} (y_n+z_n)$$
# 
# subject to constraints
# 
# $$
# \begin{array}{rcl}
# \sum_{i \in S} w_i\bar{R}_i & \geq & R_{p} \\ \\
# \sum_{i \in S} w_i & = & 1 \\ \\
# y_n-z_n & = & \sum_{i \in S} w_i(R_i(t_n)-\bar{R}_i) \quad \mbox{for}\quad n = 1,\ldots,N
# \end{array}
# $$
# 
# As discussed by Feinstein and Thapa, this version reduces the problem to $N+2$ constraints in $ 2N+\mbox{card}(S)$ decision variables, noting the `card` function.

# ## Seeding the GLPK Pseudo-Random Number Generator

# Unfortunately, MathProg does not provide a method to seed the pseudo-random number generator in GLPK.  Instead, the following GMPL code fragment uses the function gmtime() to find the number of seconds since 1970. Dropping the leading digit avoids subsequent overflow errors, and the square returns a number with big changes every second.  Extracting the lowest digits produces a number between 0 and 100,000 that determines how many times to sample GLPK's pseudo-random number generator prior to subsequent use. This hack comes with no assurances regarding its statistical properties.
# 
# 
#     /* A workaround for the lack of a way to seed the PRNG in GMPL */
#     param utc := prod {1..2} (gmtime()-1000000000);
#     param seed := utc - 100000*floor(utc/100000);
#     check sum{1..seed} Uniform01() > 0;
# 

# ## Simulation of the Historical Returns

# For this implementation, historical returns are simulated assuming knowledge of the means and covariances of asset returns. We begin with a vector of mean returns $\bar{R}$ and covariance matrix $\Sigma$ estimated by
# 
# $$
# \Sigma_{ij}   \approx  \frac{1}{N-1}\sum_{n=1}^N(R_i(t_n)-\bar{R}_i)(R_j(t_n)-\bar{R}_j)
# $$
# 
# Simulation of historical returns requires generation of samples from a multivariable normal distribution.  For this purpose  we  compute the Cholesky factorization where, for a positive semi-definite $\Sigma$, $\Sigma=CC^T$ and $C$ is a lower triangular matrix.  The following MathProg code fragment implements the Cholesky-Crout algorithm.  
# 
# 
#     /* Cholesky Lower Triangular Decomposition of the Covariance Matrix */
#     param C{i in S, j in S : i >= j} :=
#         if i = j then
#             sqrt(Sigma[i,i]-(sum {k in S : k < i} (C[i,k]*C[i,k])))
#         else
#             (Sigma[i,j]-sum{k in S : k < j} C[i,k]*C[j,k])/C[j,j];
# 
# 
# Without error checking, this code fragment fails unless $\Sigma$ is positive definite.  The covariance matrix is normally positive definite for real-world data, so this is generally not an issue.  However, it does become an issue if one attempts to include a risk-free asset, such as a government bond, in the set of investment assets.
# 
# Once the Cholesky factor $C$ has been computed, a vector of simulated returns $R(t_n)$ is given by $R(t_n) = \bar{R} + C Z(t_n)$ where the elements of $Z(t_n)$ are independent samples from a normal distribution with zero mean and unit variance.
# 
#     /* Simulated returns */
#     param N default 5000;
#     set T := 1..N;
#     param R{i in S, t in T} := Rbar[i] + sum {j in S : j <= i} C[i,j]*Normal(0,1);
# 

# ## MathProg Model

# In[1]:


get_ipython().run_cell_magic('script', 'glpsol -m /dev/stdin', '\n# Example: PortfolioMAD.mod  Portfolio Optimization using Mean Absolute Deviation\n\n/* Stock Data */\n\nset S;                                    # Set of stocks\nparam r{S};                               # Means of projected returns\nparam cov{S,S};                           # Covariance of projected returns\nparam r_portfolio\n    default (1/card(S))*sum{i in S} r[i]; # Lower bound on portfolio return\n\n/* Generate sample data */\n\n/* Cholesky Lower Triangular Decomposition of the Covariance Matrix */\nparam c{i in S, j in S : i >= j} := \n    if i = j then\n        sqrt(cov[i,i]-(sum {k in S : k < i} (c[i,k]*c[i,k])))\n    else\n        (cov[i,j]-sum{k in S : k < j} c[i,k]*c[j,k])/c[j,j];\n\n/* Because there is no way to seed the PRNG, a workaround */\nparam utc := prod {1..2} (gmtime()-1000000000);\nparam seed := utc - 100000*floor(utc/100000);\ncheck sum{1..seed} Uniform01() > 0;\n\n/* Normal random variates */\nparam N default 5000;\nset T := 1..N;\nparam zn{j in S, t in T} := Normal(0,1);\nparam rt{i in S, t in T} := r[i] + sum {j in S : j <= i} c[i,j]*zn[j,t];\n\n/* MAD Optimization */\n\nvar w{S} >= 0;                # Portfolio Weights with Bounds\nvar y{T} >= 0;                # Positive deviations (non-negative)\nvar z{T} >= 0;                # Negative deviations (non-negative)\n\nminimize MAD: (1/card(T))*sum {t in T} (y[t] + z[t]);\n\ns.t. C1: sum {s in S} w[s]*r[s] >= r_portfolio;\ns.t. C2: sum {s in S} w[s] = 1;\ns.t. C3 {t in T}: (y[t] - z[t]) = sum{s in S} (rt[s,t]-r[s])*w[s];\n\nsolve;\n\n/* Report */\n\n/* Input Data */\nprintf "Stock Data\\n\\n";\nprintf "         Return   Variance\\n";\nprintf {i in S} "%5s   %7.4f   %7.4f\\n", i, r[i], cov[i,i];\n\nprintf "\\nCovariance Matrix\\n\\n";\nprintf "     ";\nprintf {j in S} " %7s ", j;\nprintf "\\n";\nfor {i in S} {\n    printf "%5s  " ,i;\n    printf {j in S} " %7.4f ", cov[i,j];\n    printf "\\n";\n}\n\n/* MAD Optimal Portfolio */\nprintf "\\nMinimum Absolute Deviation (MAD) Portfolio\\n\\n";\nprintf "  Return   = %7.4f\\n",r_portfolio;\nprintf "  Variance = %7.4f\\n\\n", sum {i in S, j in S} w[i]*w[j]*cov[i,j];\nprintf "         Weight\\n";\nprintf {s in S} "%5s   %7.4f\\n", s, w[s];\nprintf "\\n";\n\ntable tab0 {s in S} OUT "JSON" "Optimal Portfolio" "PieChart": \n    s, w[s]~PortfolioWeight;\n    \ntable tab1 {s in S} OUT "JSON" "Asset Return versus Volatility" "ScatterChart":\n    sqrt(cov[s,s])~StDev, r[s]~Return;\n    \ntable tab2 {s in S} OUT "JSON" "Portfolio Weights" "ColumnChart": \n    s~Stock, w[s]~PortfolioWeight;\n    \ntable tab3 {t in T} OUT "JSON" "Simulated Portfolio Return" "LineChart": \n    t~month, (y[t] - z[t])~PortfolioReturn;\n\n/* Simulated Return data in Matlab Format */\n/*\nprintf "\\nrt = [ ... \\n";\nfor {t in T} {\n   printf {s in S} "%9.4f",rt[s,t];\n   printf "; ...\\n";\n}\nprintf "];\\n\\n";\n*/\n\ndata;\n\n/* Data for monthly returns on four selected stocks for a three\nyear period ending December 4, 2009 */\n\nparam N := 200;\n\nparam r_portfolio := 0.01;\n\nparam : S : r :=\n    AAPL    0.0308\n    GE     -0.0120\n    GS      0.0027\n    XOM     0.0018 ;\n\nparam   cov : \n            AAPL    GE      GS      XOM  :=\n    AAPL    0.0158  0.0062  0.0088  0.0022\n    GE      0.0062  0.0136  0.0064  0.0011\n    GS      0.0088  0.0064  0.0135  0.0008\n    XOM     0.0022  0.0011  0.0008  0.0022 ;\n\nend;')


# In[ ]:




