#!/usr/bin/env python
# coding: utf-8

# <!--NOTEBOOK_HEADER-->
# *This notebook contains course material from [CBE40455](https://jckantor.github.io/CBE40455) by
# Jeffrey Kantor (jeff at nd.edu); the content is available [on Github](https://github.com/jckantor/CBE40455.git).
# The text is released under the [CC-BY-NC-ND-4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode),
# and code is released under the [MIT license](https://opensource.org/licenses/MIT).*

# <!--NAVIGATION-->
# < [Scheduling](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/04.00-Scheduling.ipynb) | [Contents](toc.ipynb) | [Machine Bottleneck](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/04.02-Machine-Bottleneck.ipynb) ><p><a href="https://colab.research.google.com/github/jckantor/CBE40455/blob/master/notebooks/04.01-Critical-Path-Method.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><p><a href="https://raw.githubusercontent.com/jckantor/CBE40455/master/notebooks/04.01-Critical-Path-Method.ipynb"><img align="left" src="https://img.shields.io/badge/Github-Download-blue.svg" alt="Download" title="Download Notebook"></a>

# # Critical Path Method

# This notebook demonstrates the Critical Path Method using GLPK/MathProg.

# ## Background

# The Critical Path Method is a technique for calculating the shortest time span needed to complete a series of tasks. The tasks are represented by nodes, each labelled with the duration. The precedence order of the task is given by a set of arcs.
# 
# Here we demonstrate the representation and calculation of the critical path. Decision variables are introduced for
# 
# * Earliest Start
# * Earliest Finish
# * Latest Start
# * Latest Finish
# * Slack = Earliest Finish - Earliest Start = Latest Finish - Earliest Finish
# 
# Tasks on the Critical Path have zero slack.

# ## MathProg Model

# In[1]:


get_ipython().run_cell_magic('writefile', 'ProjectCPM.mod', "\n# Example: ProjectCPM.mod\n\nset TASKS;\nset ARCS within {TASKS cross TASKS};\n\n/* Parameters are the durations for each task */\nparam dur{TASKS} >= 0;\nparam desc{TASKS} symbolic;\n\n/* Decision Variables associated with each task*/\nvar Tes{TASKS} >= 0;     # Earliest Start\nvar Tef{TASKS} >= 0;     # Earliest Finish\nvar Tls{TASKS} >= 0;     # Latest Start\nvar Tlf{TASKS} >= 0;     # Latest Finish\nvar Tsl{TASKS} >= 0;     # Slacks\n\n/* Global finish time */\nvar Tf >= 0;\n\n/* Minimize the global finish time and, secondarily, maximize slacks */\nminimize ProjectFinish : card(TASKS)*Tf - sum {j in TASKS} Tsl[j];\n\n/* Finish is the least upper bound on the finish time for all tasks */\ns.t. Efnsh {j in TASKS} : Tef[j] <= Tf;\ns.t. Lfnsh {j in TASKS} : Tlf[j] <= Tf;\n\n/* Relationship between start and finish times for each task */\ns.t. Estrt {j in TASKS} : Tef[j] = Tes[j] + dur[j];\ns.t. Lstrt {j in TASKS} : Tlf[j] = Tls[j] + dur[j];\n\n/* Slacks */\ns.t. Slack {j in TASKS} : Tsl[j] = Tls[j] - Tes[j];\n\n/* Task ordering */\ns.t. Eordr {(i,j) in ARCS} : Tef[i] <= Tes[j];\ns.t. Lordr {(j,k) in ARCS} : Tlf[j] <= Tls[k];\n\n/* Compute Solution  */\nsolve;\n\n/* Print Report */\nprintf 'PROJECT LENGTH = %8g\\n',Tf;\n\n/* Critical Tasks are those with zero slack */\n\n/* Rank-order tasks on the critical path by earliest start time */\nparam r{j in TASKS : Tsl[j] = 0} := sum{k in TASKS : Tsl[k] = 0}\n   if (Tes[k] <= Tes[j]) then 1;\n\nprintf '\\nCRITICAL PATH\\n';\nprintf '  TASK  DUR    Start   Finish  Description\\n';\nprintf {k in 1..card(TASKS), j in TASKS : Tsl[j]=0 && k==r[j]}\n   '%6s %4g %8g %8g  %-25s\\n', j, dur[j], Tes[j], Tef[j], desc[j];\n\n/* Noncritical Tasks have positive slack */\n\n/* Rank-order tasks not on the critical path by earliest start time */\nparam s{j in TASKS : Tsl[j] > 0} := sum{k in TASKS : Tsl[k] = 0}\n   if (Tes[k] <= Tes[j]) then 1;\n\nprintf '\\nNON-CRITICAL TASKS\\n';\nprintf '            Earliest Earliest   Latest   Latest \\n';\nprintf '  TASK  DUR    Start   Finish    Start   Finish    Slack  Description\\n';\nprintf {k in 1..card(TASKS), j in TASKS : Tsl[j] > 0 && k==s[j]}\n   '%6s %4g %8g %8g %8g %8g %8g  %-25s\\n', \n   j,dur[j],Tes[j],Tef[j],Tls[j],Tlf[j],Tsl[j],desc[j];\nprintf '\\n';\n\nend;")


# ## Example: Stadium Construction

# Stadium Construction, [Example 7.1.1](http://www.maximalsoftware.com/modellib/modXpressMP.html) from [Christelle Gueret, Christian Prins, Marc Sevaux, "Applications of Optimization with Xpress-MP," Chapter 7, Dash Optimization, 2000](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.69.9634&rep=rep1&type=pdf).

# In[2]:


get_ipython().run_cell_magic('script', 'glpsol -m ProjectCPM.mod -d /dev/stdin -y ProjectCPM.txt --out output', "\nparam : TASKS : dur desc :=\n   T01   2.0  'Installing the contruction site'\n   T02  16.0  'Terracing'\n   T03   9.0  'Constructing the foundations'\n   T04   8.0  'Access roads and other networks'\n   T05  10.0  'Erecting the basement'\n   T06   6.0  'Main floor'\n   T07   2.0  'Dividing up the changing rooms'\n   T08   2.0  'Electrifying the terraces'\n   T09   9.0  'Constructing the roof'\n   T10   5.0  'Lighting the stadium'\n   T11   3.0  'Installing the terraces'\n   T12   2.0  'Sealing the roof'\n   T13   1.0  'Finishing the changing rooms'\n   T14   7.0  'Constructing the ticket office'\n   T15   4.0  'Secondary access roads'\n   T16   3.0  'Means of signaling'\n   T17   9.0  'Lawn and sports accessories'\n   T18   1.0  'Handing over the building' ;\n\nset ARCS := \n   T01  T02\n   T02  T03\n   T02  T04\n   T02  T14\n   T03  T05\n   T04  T07\n   T04  T10\n   T04  T09\n   T04  T06\n   T04  T15\n   T05  T06\n   T06  T09\n   T06  T11\n   T06  T08\n   T07  T13\n   T08  T16\n   T09  T12\n   T11  T16\n   T12  T17\n   T14  T16\n   T14  T15\n   T17  T18 ;\n\nend;")


# In[3]:


f = open('ProjectCPM.txt')
print f.read()
f.close()


# ## Visualization

# In[5]:


import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

G=nx.Graph()
G.add_nodes_from(['T01','T02','T03','T04','T05','T06','T07','T08',    'T09','T10','T11','T12','T13','T14','T15','T16','T18'])

G.add_edge('T01','T02')
G.add_edge('T02','T03')
G.add_edge('T02','T04')
G.add_edge('T02','T14')
G.add_edge('T03','T05')
G.add_edge('T04','T07')
G.add_edge('T04','T10')
G.add_edge('T04','T09')
G.add_edge('T04','T06')
G.add_edge('T04','T15')
G.add_edge('T05','T06')
G.add_edge('T06','T09')
G.add_edge('T06','T11')
G.add_edge('T06','T08')
G.add_edge('T07','T13')
G.add_edge('T08','T16')
G.add_edge('T09','T12')
G.add_edge('T11','T16')
G.add_edge('T12','T17')
G.add_edge('T14','T16')
G.add_edge('T14','T15')
G.add_edge('T17','T18') ;

nx.draw(G)
plt.show()


# In[ ]:





# In[ ]:





# <!--NAVIGATION-->
# < [Scheduling](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/04.00-Scheduling.ipynb) | [Contents](toc.ipynb) | [Machine Bottleneck](http://nbviewer.jupyter.org/github/jckantor/CBE40455/blob/master/notebooks/04.02-Machine-Bottleneck.ipynb) ><p><a href="https://colab.research.google.com/github/jckantor/CBE40455/blob/master/notebooks/04.01-Critical-Path-Method.ipynb"><img align="left" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open in Google Colaboratory"></a><p><a href="https://raw.githubusercontent.com/jckantor/CBE40455/master/notebooks/04.01-Critical-Path-Method.ipynb"><img align="left" src="https://img.shields.io/badge/Github-Download-blue.svg" alt="Download" title="Download Notebook"></a>
