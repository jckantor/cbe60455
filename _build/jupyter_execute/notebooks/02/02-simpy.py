#!/usr/bin/env python
# coding: utf-8

# # Introduction to Simpy

# In[1]:


get_ipython().system('pip install simpy')


# ## First Steps

# ### The simplest Simpy model
# 
# Simpy implements an Environment() object that does the heavy work of scheduling processing events that occur when simulating complex models. The simplest possible Simpy model is one that creates and runs an empty simulation environment.

# In[2]:


import simpy

env = simpy.Environment()
env.run()


# ### Adding a process
# 
# The components that make up a Simpy models are implemented using python gnerators. A generator begins with the `def` statement and must have at least one `yield` statement. The yield statement returns an event to the simulation environment. A frequently used events is `simpy.Environment.timeout()` which simulates the passage of time in a model component.
# 
# Once the generator has been defined, an instance of the generator is passed to the simulation environment using the `simpy.Environment.process()` method.
# 
# The following model defines a process model in which "Hello, World" is printed after a simulated 2.0 seconds have elapsed.

# In[3]:


import simpy

def say_hello():
    yield env.timeout(2.0)           # let 2.0 seconds of simulated time elapse
    print("Hello, World")            # print "Hello, World"

env = simpy.Environment()            # create a simulation environment instance
env.process(say_hello())                # register an instance of the process model
env.run()                            # run the simulaton


# ### Accessing time

# The environment variable `simpy.Environment.now` returns the current time in the simulation.

# In[4]:


import simpy

def say_hello():
    print("time =", env.now)             # print current time in the simulation
    yield env.timeout(2.0)               # take a time out of 1 time unit
    print("time =", env.now)             # print current time in the simulation
    print("Hello, World")

env = simpy.Environment()                # create a simulation environment instance
env.process(say_hello())                 # tell the simulation environment which generators to process
env.run()                                # run the simulation


# ### Creating multiple instances of a generator

# The same generator can be used to create multiple instances of a model component. For example, this cell creates three instances of a generator. The generator is modified to add an identifier and variable time delay.
# 
# **Carefully examine the output of this cell. Be sure you explain the order in which the print statements appear in the output.**

# In[5]:


import simpy

def say_hello(id, delay):
    print(id, "time =", env.now)         # print current time in the simulation
    yield env.timeout(delay)             # take a time out of 1 time unit
    print(id, "says Hello, World at time", env.now)

env = simpy.Environment()                # create a simulation environment instance
env.process(say_hello("A", 3.0))         # tell the simulation environment which generators to process
env.process(say_hello("B", 2.0))         # tell the simulation environment which generators to process
env.process(say_hello("C", 4.0))         # tell the simulation environment which generators to process
env.run()                                # run the simulation


# ### Running a simulation for a known period of time
# 
# The simulations presented above finish when all of the components making up the model have finished. In process modeling, however, some model components will simply cycle forever leading to a simulation that would never end. To handle this situation, the `simpy.Environment.run()` has an optional parameter `until` the causes the simulation to end a known point in time.
# 
# The next modifies the generator to include an infinite loop, and controls the simulation period using the `until` parameter.

# In[6]:


import simpy

def say_hello(id, delay):
    print(id, "time =", env.now)          # print current time in the simulation
    while True:
        yield env.timeout(delay)          # take a time out of 1 time unit
        print(id, "says Hello, World at time", env.now)

env = simpy.Environment()                 # create a simulation environment instance
env.process(say_hello("A", 3.0))          # tell the simulation environment which generators to process
env.process(say_hello("B", 2.0))          # tell the simulation environment which generators to process
env.process(say_hello("C", 4.0))          # tell the simulation environment which generators to process
env.run(until=10.0)                       # run the simulation


# ### Logging data
# 
# Discrete-event simulations can create large amounts of data. A good practice is create a data log for the purpose of capturing data generated during the simulation. After the simulation is complete, the data log can be processed to create reports and charts to analyze results of the simulation.
# 
# If not sure what to log, a good practice is log at least three items for each event:
# 
# * **who** what object created this event
# * **what** a description of the event
# * **when** when the event occurred

# In[7]:


import simpy

data_log = []                             # create an empty data log

def say_hello(id, delay):
    while True:
        yield env.timeout(delay)          # take a time out of 1 time unit
        data_log.append([id, "Hello, World", env.now])   # log who (id), what ("Hello, World"), when (env.now)

env = simpy.Environment()                 # create a simulation environment instance
env.process(say_hello("A", 3.0))          # tell the simulation environment which generators to process
env.process(say_hello("B", 2.0))          # tell the simulation environment which generators to process
env.process(say_hello("C", 4.0))          # tell the simulation environment which generators to process
env.run(until=10.0)                       # run the simulation

for data_record in data_log:
    print(data_record)


# ## Example: Simulating a Stirred Tank Reactor

# ### Model development
# 
# Write a Python generator to simulate the response of a differential equation describing concentration in a stirred tank reactor.
# 
# $$\frac{dC}{dt} = -k C + q(t)$$
# 
# where k = 1.0,  C(0) = 1.0 and q(t) has a constant value 0.5. Use can use the Euler approximation
# 
# $$C(t + \Delta t) = C(t) + \Delta t \left[ - k C(t) + q(t) \right]$$
# 
# The definition of the generator should allow specification of the time step, $\Delta t$, and the rate constant $k$.

# In[11]:


# Solution

import simpy

def flow():
    global q
    q = 0.5
    yield env.timeout(100.0)

def reactor(dt, k):
    C = 1.0
    while True:
        print(round(env.now, 2), round(C, 2))
        yield env.timeout(dt)
        C = C - k*dt*C + q*dt
        
env = simpy.Environment()
env.process(flow())
env.process(reactor(0.5, 1.0))
env.run(until=20)


# ### Exercise
# 
# Extend stirred tank model by adding the following features:
# 
# * Modify the flow simulation so that the flowrate initially starts at 0.0, switches to a value of 0.5 at t=1.0, then back to 0.0 five time units later.
# * Add a data log to records time and the values of concentration, $C$, and flowrate $q$. After debugging, remove the print statements.
# * When the simulation is complete, convert the data log to a numpy array, then create plots of $C(t)$ and $q(t)$. Take time to label axes and, if necessary, add a legend. 
# * The whole thing should run in a single cell

# In[10]:


# Your solution goes in this cell

