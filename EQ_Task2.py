#!/usr/bin/env python
# coding: utf-8

# In[98]:


import findspark
findspark.init()


# In[99]:


import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from collections import defaultdict


conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)
d = defaultdict(list)


# ### Read the relations file in a dictionary

# In[100]:


with open("relations.txt") as f:
    for line in f:
        (pre_req, task) = line.split('->')
        d[int(task)].append(int(pre_req))
print(d)


# ### Find the optimal path from start to end.
# ##### We start here from end and traverse back to the start. The recursive function 'form_pipeline' needs to be called with only the star_task and end_task. A stack is used to keep track of the pre-requisite jobs for each job. Recursion stops the moment the start_task is reached.
# ##### Assumptions as per directions in the question:
#     1. The pre-requisites of the start task have not been considered
#     2. The pre-requisites of all the intermediate tasks have been considered untill the pre-requisites have no other pre-requisites.
# 

# In[101]:


# check internal dependencies between a set of tasks [a,b,c] who are together prerequisites for  
# another task d. This step is required to set the right order of execution of [a,b,c] before d 
def check_int_dep(req):
    new_req = []
    dep_set = {}
    req_set = set(req)
    for i in req_set:
        dep_set = set(d[i])
        if len(dep_set):
            if (dep_set & req_set):
                new_req.append(i)
    for j in list(req_set - set(new_req)):
        new_req.append(j)
    return list(new_req)
        


# In[102]:


e = []    
def form_pipeline(stack):
    for item in stack:
        start_task = stack.pop()
        if item != start_task:
            r = d[item]
            e.append(item)
            stack.pop(0)

#             check dependency and return right order
            if len(r):
                req = check_int_dep(r)
                for i in req:
                    if i not in stack:
                        stack.append(i)

            if start_task in stack:
                stack.remove(start_task)
            stack.append(start_task)
            
#             recursively call the function
            form_pipeline(stack)
        
#         when the start_task is reached, no need for recursion
        else:
            e.append(item)

    return(e[::-1])


# #### The function form_pipeline is called with start_task and end_task and it returns the pipeline of jobs

# In[103]:


end_task=36
start_task=73
print('---------Printing the pipeline of jobs----------')
print(form_pipeline([end_task, start_task]))


# In[105]:


sc.stop()


# In[ ]:




