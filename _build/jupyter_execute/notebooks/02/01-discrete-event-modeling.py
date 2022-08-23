#!/usr/bin/env python
# coding: utf-8

# # Introduction to Discrete Event Simulation

# ## What is discrete-event simulation?
# 
# Simulation is of profound importance to engineering practice. By comparing simulation to experiments, simulation can be used to verify one's understanding of complex systems. Going the next step, simulation can be used to validate the design of new systems prior to their physical realization. 
# 
# Discrete-event simulation addresses the simulation of systems characterized by events. In contrast to systems typically modeled by low order and mathematically smooth functions and differential equations, discrete-event systems generally comprise objects and activities demarcarted by events such as the start and completion of tasks, the assignment of discrete resources to tasks, preemption of tasks. Discrete-event simulation has a wide range of applications ranging from manufacturing and logistics to health care delivery.
# 
# There are a broad range of [software tools that support discrete-event simulation](https://en.wikipedia.org/wiki/List_of_discrete_event_simulation_software). In this unit we will examine two representative tool sets. The first is [SimPy](https://simpy.readthedocs.io/en/latest/), an easy to use Python library that is used to construct custom simulations and can be integrated with a vast array of other Python tools. The second is [AnyLogic](https://www.anylogic.com/), a commercial simulation too representative of current state-of-the-art in discrete-event simulation.

# ## SimPy
# 
# * [SimPy documentation](https://simpy.readthedocs.io/en/latest/) includes a [tutorial](https://simpy.readthedocs.io/en/latest/simpy_intro/index.html#intro)
# * Scherfke, Stefan. [Slides](https://stefan.sofa-rockers.org/downloads/simpy-ep14.pdf) from a [conference talk](https://www.youtube.com/watch?v=Bk91DoAEcjY) given by one of the current developers of SimPy.
# * Heintz, Meghan.[Launching a new warehouse with SimPy at Rent the Runway](https://www.youtube.com/watch?v=693UiPq6mII), a presentation at PyData NYC 2019.
# * Horgan, Juan (2020). [Manufacturing simulation using SimPy](https://towardsdatascience.com/manufacturing-simulation-using-simpy-5b432ba05d98): A complete example of a simulation of a guitar factory using SimPy and class definitions.

# In[ ]:




