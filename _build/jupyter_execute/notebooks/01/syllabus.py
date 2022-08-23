#!/usr/bin/env python
# coding: utf-8

# # CBE 40455/60455 Process Operations: Syllabus
# August 23, 2022
# 
# ## Catalog Description
# This course introduces students to methods for the analysis and optimization of process operations. Topics will include process modeling, continuous and discrete optimization, scheduling, supply chains, scenario analysis, and financial  analysis for process operations. Special emphasis will be given to practical implementation of methods for real world applications.
# 
# ## Course Information
# 
# ### Schedule and Locations
# * **Class Meetings:** TTh 9:35-10:50 in 119 DeBartolo Hall
# * **Course Management:** Course materials and assignments will be managed using [Canvas](), Notre Dame's Learning Management System.
# * **Github Repository:** [jckantor.github.io/CBE60455](jckantor.github.io/CBE60455)
# * **Instructor:** Jeffrey Kantor, Department of Chemical & Biomolecular Engineering.
#     * **Office:** 257 Nieuwland Hall
#     * **Office Hours:** Monday and Wednesday afternoons are the regularly scheduled drop-in office hours for the course. Otherwise I'm generally available during regular business hours by appointment.
#     * **Email:** [jeff@nd.edu](jeff@nd.edu) This is the generally the best way to reach me.
#     * **Text Message:** 574-532-4233
#   
# 
# ## Topics
# 
# ### Weeks 1-2: Discrete Event Dynamics and Simulation
# 
# * **Discrete Event Simulation**. Implementing discrete event simulations in Python. Poisson processes, queues, evaluating simulation results.
# 
# Reading: \citep{Beamon1998}  \citep{Christopher2000} 
# \citep{Shah2005} \citep{You2008} \citep{Ferrio2008} 
# 
# Project: Simulation a COVID outbreak.
# 
# ### Weeks 3-4: Linear Optimization
# * Blending Problems. Formulation and solution of `blending'  and `diet' models. History and significance of linear programming. Comparison of single product and multi-product plants. 
# * Modeling Languages. Formulation and solution of linear programming problems using spreadsheets, algebraic modeling languages, and embedded modeling languages.
# * Examples of modeling languages for mathematical programming.
#     * AMPL
#     * APMonitor
#     * GLPK/MathProg
#     * CMPL
#     * [CPLEX/OPL](http://www-03.ibm.com/software/products/en/subcategory/decision-optimization)
# * Examples of modeling languages embedded within programming languages.
#     * Matlab/CVX
#     * Matlab/YALMIP
#     * Matlab|Python/APMonitor
#     * Python/PuLP
#     * Python/Pyomo
# * Mathematical formulation and optimality conditions. Standard formulations of linear programming problems. Necessary conditions for optimality.  Outline of an active set method for solution.
# * Sensitivity Analysis. Weak and strong duality. Slack variables, shadow prices. Applications to process analysis and decision making. 
# * Transportation, Assignment, and Network Flow Problems]
# * Project Management. Critical Path Method, PERT and Gantt charts.
# 
# {Readings:} 
# 
# {Project:} Optimization of a simple model for refinery operations (Example 16.1 from \citep{Edgar2001}).
# 
# ### Weeks 5-6: Scheduling
# * Mixed Integer Linear Programming. Integer and binary variables. Using binary variables to model logical constraints and application to process decisions. Application to agricultural and stock cutting operations.
# * Scheduling. Empirical methods for job scheduling. 
# * Machine and Job Scheduling. Modeling machines and jobs. Disjuctive constraints. Locating bottlenecks, tracking jobs and machine activities with Gannt charts. Practical implications of computational complexity.
# * Short Term Scheduling of Batch Processes. Resource task networks, state task networks.
# 
# Reading: \citep{Mendez2006a} \citep{Floudas2005a}
# 
# Project: Scheduling production for a contract pharmaceutical manufacturer.
# 
# ### Weeks 7-8: Logistics
# * **Inventory Management**. Inventory, reorder levels, economic order quantity. Empirical models for inventory management.
# * **Supply Chains**. [Beer Game Simulation](http://www.beergame.org/). Multi-echelon supply chains, processes, dynamics, and the `bullwhip' effect. The role of information flow in supply chains.
# 
# #### Readings
# 
# Fisher, Marshall L. ["What is the right supply chain for your product?." Harvard business review 75 (1997): 105-117.](https://pdfs.semanticscholar.org/647a/c2ded3d69e41bb09ef5556aa942e01abd14d.pdf)
# 
# 
# ### Weeks 9-10: Optimization under Uncertainty
# 
# * Newsvendor Problem. Optimal order size for uncertain demand \citep{Petruzzi1999}.
# 
# * Scenario Analysis. Plant expansion. Introduction to stochastic programming. Expected values, optimization of the mean scenario, value of perfect information, value of the stochastic solution. `Here and Now' decisions versus`Wait and See'.
# 
# * Stochastic Linear Programming. Two stage decisions, recourse variables. Implementing stochastic linear programs with algebraic modeling languages. Solution by decomposition methods.
# 
# * Process Resiliency. Measures of process flexibility and resiliency to perturbations.
# 
# Readings: \citep{Sen1999a} \citep{Grossmann2014}
# 
# Project: Production planning for a consumer goods company.
# 
# ### Weeks 11-12: Financial Modeling, Risk and Diversification
# 
# * **Time value of money**. Discounted cash flows. Replicating portfolio for a sequence of cash flows. Bonds and duration. Net present value and internal rate of returns. Project valuation.
# * **Stochastic Modeling of Asset Prices**. Statistical properties of prices for financial assets and commodity goods. Statistical distributions, discrete time models, model calibration, approximations with binomial trees. Correlation and copulas.
# * **Commodity Markets**. Futures, forwards, and options, and swaps. Hedging operational costs. Replicating portfolios.
# * **Investment Objectives**. Kelly's criterion, and log optimal growth. Logarithmic utility functions, certainty equivalence principle, coherent risk measures, first and second order stochastic dominance. Practical risk measures.
# * **Portfolio Management**. Risk versus return. Markowitz analysis, diversification. Efficient frontier.  Comparison of Markowitz to Mean Absolute Difference, and optimal portfolios. Effects of adding a risk-free asset, two-fund theorem, market price of risk.
# * **Real Options**. European and American options on a financial asset. Extending the analysis to real options on tradable assets, and to decisions without tradable assets. \citep{Davis20xx}
# 
# Reading: \citep{Adner2004}, \citep{Anderson2013}, 
# 
# Project: (a) Valuation of a natural resource operation, or (b) Pricing and managing an energy swap for a flex-fuel utility
# 
# 
# ### Weeks 13-14: Capstone Project
# The capstone experience will be a team-based, integrative, open-ended project. Student teams will propose projects of their own design or select from a suggested list of projects. Projects will consist of an initial proposal, an oral status report to the class, a final poster presentation and written report, and a Github repository memorializing the project.
# 
# #### Project Ideas
# 
# * **Networks Against Time.** Blood supplies, medical nuclear supplies, pharacueticals, food, and fast fashion are all examples of supply chains for perishable products. The book [Networks Against Time](https://www.springer.com/gp/book/9781461462767) (available from [Hesburgh Library](https://link-springer-com.proxy.library.nd.edu/book/10.1007%2F978-1-4614-6277-4)) describes analytics suitable for supply chains of perishable goods. Using an example from this book, develop an analytical model, sample calculation, and simulation.
# * **Model Predictive Control** is an optimization-based method for feedback control of dynamical systems. Implement and MPC controller for a system with discrete decisions.
# * **Refinery Products Pooling Problem** is the task of finding a cost set of product pools from which final products can be blended. The project is to develop and solve an example problem.
# * **Energy Swap** design and price an energy swap for a flex fuel utility located on a university campus.
# * **Demand Response** evaluate the potential for demand response for an aluminum smelter or a chlorine producer.
# * **Process Resiliency** has been a topic discussed in the process systems engineering literature since the pioneering work of Morari and Grossmann in the early 1980's (Grossmann, 2014). The purpose of this project is to implement a measure of process resiliency and demonstrate its application to a chemical process.
# * **Log Periodic Power Laws** are models that have been used to predict the collapse of speculative `bubbles' in financial and commodity markets. For this project, develop a regression method to fit a log periodic power law model to commodity price data, and compare a recent data series of your choice to the price of gold in the 2007--2009 period.
# 
# Additional project ideas can be found at the following links:
# 
# * [AIMMS Application Examples](http://www.aimms.com/downloads/application-examples)
# 
# #### Project Github Repository
# Your project should be memorialized in the form of a [Github](https://github.com/) repository. The simplest way to manage your repository is to use download the [Github Desktop application](https://desktop.github.com/) and follow the [tutorial and guides](https://help.github.com/en/articles/set-up-git).
# 
# 
# 
# 
# 
# 

# In[ ]:





# In[ ]:




