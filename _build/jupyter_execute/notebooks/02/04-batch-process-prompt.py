#!/usr/bin/env python
# coding: utf-8

# # Batch Chemical Process
# 

# ## Problem Statement
# 
# Schultheisz, Daniel, and Jude T. Sommerfeld. "Discrete-Event Simulation in Chemical Engineering." Chemical Engineering Education 22.2 (1988): 98-102.
# 
# ![Batch Process](figures/BatchProcess.png)
# 
#     "... a small, single-product batch chemical plant has three identical reactors in parallel, followed by a single storage tank and a batch still. Customer orders (batches) to be filled (which begin with processing in the reactor) occur every 115 ± 30 minutes, uniformly distributed. The reaction time in a given reactor is 335 ± 60 minutes, and the distillation time in the still is 110 ± 25 minutes, both times uniformly distributed. The holding capacity of the storage tank is exactly one batch. Hence, the storage tank must be empty for a given reactor to discharge its batch; if not, the reactor cannot begin processing a new batch until the storage tank becomes empty. The simulation is to be run for 100 batches. The model should have the capability to collect waiting line statistics for the queue immediately upstream of the reactor.""
#     
#     
# You have been hired by the client as a consulting engineer. Prepare a SimPy simulation of this process to deliver to the client. The delivery should include functions to report the key performance indicators, visualize the results of the simulation, and to conduct 'what-if' studies to determine ways to improve process performance.

# ## Analysis
# 
# 1. What is the purpose of the simulation? What question needs to be answered? In thinking about this, carefully consider what has been requested, what other questions are relevant to improving system performance.
# 
# 2. What are the key performance indicators?  What data needs to be collected?
# 
# 3. What simulation objects should be created for this application?
# 
# 4. What classes of shared resources will be used in this model?

# In[ ]:




