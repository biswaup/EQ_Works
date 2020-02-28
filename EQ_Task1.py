#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


import pyspark
import pandas as pd
from haversine import haversine
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import udf, struct
from pyspark.sql.types import StringType, FloatType
import networkx as nx 
import math
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set(style="whitegrid")
conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)


# ### read the data files

# In[144]:


# read requests
req = pd.read_csv('DataSample.csv') 
print('Total number of requests: ', len(req))

# read POIs
poi = pd.read_csv('POIList.csv') 
print('Total number of POIs: ', len(poi))


# ### preprocessing the files

# In[145]:


# remove spaces from column names
req.columns = req.columns.str.strip()
poi.columns = poi.columns.str.strip()

# should you add a sort by those 3 columns here? to make it faster?
req[['Latitude', 'Longitude']] = req[['Latitude', 'Longitude']].apply(pd.to_numeric)
poi[['Latitude', 'Longitude']] = poi[['Latitude', 'Longitude']].apply(pd.to_numeric)

# find and delete duplicates based on timestamp, long and lat
req = req.drop_duplicates(['TimeSt', 'Latitude', 'Longitude'], keep='first')
print('Total number of unique requests: ', len(req))

poi = poi.drop_duplicates(['Latitude', 'Longitude'], keep='first')
print('Total number of unique POIs: ', len(poi))


# ### assign request to the closest POI

# In[146]:


def calc_haversine(a, b):
    loc = poi
    loc['dist'] = loc.apply(lambda x: haversine((x.Latitude, x.Longitude), (a, b)), axis=1)
    loc = loc.sort_values(by = ['dist'], ascending = True)
#     print(loc.head())
    return [loc.iloc[0,0], loc.iloc[0,3]]

req['poi_id'] = req.apply(lambda x: calc_haversine(x.Latitude, x.Longitude), axis=1)

req[['poi', 'distance']] = pd.DataFrame(req['poi_id'].tolist(), index=req.index)
req = req.drop('poi_id', axis=1)
print('----printing assigned requests with their POIs----')
print(req.head())


# ### for each POI calculate average and standard deviation

# In[147]:


req_std = req.groupby('poi')['distance'].std()
req_mean = req.groupby('poi')['distance'].mean()
print('------printing standard deviation of the distances for each POIs------')
print(req_std)
print('------printing mean of the distances for each POIs------')
print(req_mean)


# ### calculate the radius and density (requests/area) for each POI

# In[151]:


req['distance'] = req['distance'].apply(str)
req_grouped = req.groupby('poi').agg({'distance': ','.join}).reset_index()
for row in req_grouped.itertuples():
    G = nx.Graph() 
    edges = []
    nodes = list(map(float, row.distance.split(',')))
    r = max(nodes)
    print('For poi {}, number of requests assigned: {}, radius: {} and density: {}'.format(row.poi, len(nodes), float(max(nodes)), (len(nodes))/(math.pi*r*r)))
#     edges = [(row.poi,float(i)) for i in nodes]
#     edges = [('POI1', 480.73511405378997), ('POI1', 13.296356636993728), ('POI1', 480.73511405378997), ('POI1', 270.38531601182416), ('POI1', 5.416854305705129), ('POI1', 8.050007946427256), ('POI1', 282.6632361831249), ('POI1', 430.1833989764478), ('POI1', 288.88464840712675), ('POI1', 292.1883102120518), ('POI1', 160.92226938108652), ('POI1', 288.88464840712675), ('POI1', 296.2003048781375), ('POI1', 254.44066602204379), ('POI1', 7.9492118154639835), ('POI1', 267.9958549970449), ('POI1', 279.46528625050826), ('POI1', 8.616659648255007), ('POI1', 346.8257332501306), ('POI1', 12.311783177393862), ('POI1', 275.7923827407298), ('POI1', 10.250547549457098), ('POI1', 12.311783177393862), ('POI1', 275.8410453361974), ('POI1', 292.79768876802876), ('POI1', 807.9603672988542), ('POI1', 297.39495556714616), ('POI1', 8.727930365734732), ('POI1', 269.50776756033866), ('POI1', 7.1747147651628245), ('POI1', 26.169728805416444), ('POI1', 268.7592392223315), ('POI1', 197.80990841712156), ('POI1', 9.388938544730971), ('POI1', 31.373258009793243), ('POI1', 269.50776756033866), ('POI1', 1.5994254795639775), ('POI1', 297.0121014055591), ('POI1', 8.727930365734732), ('POI1', 281.8296716792703), ('POI1', 432.5741349353206), ('POI1', 26.28184177728218), ('POI1', 22.417635393036015), ('POI1', 278.6026040623108), ('POI1', 7.454740937025414), ('POI1', 819.7733511574683), ('POI1', 298.3858962534849), ('POI1', 277.80782475654365), ('POI1', 377.94900396198847)]
#     G.add_edges_from(edges) 
#     plt.figure(figsize =(10, 10)) 
#     nx.draw_networkx(G)
#     plt.show()
#     break


# ### To visualize the popularity of each POI, they need to be mapped to a scale that ranges from -10 to 10. Please provide a mathematical model to implement this, taking into consideration of extreme cases and outliers. Aim to be more sensitive around the average and provide as much visual differentiability as possible.
# 
# #### Outliers

# In[152]:


# plot boxplots for each POIs
fig, ax = plt.subplots(figsize=(8,6))
req['distance'] = pd.to_numeric(req['distance'], downcast='float')
ax2 = sns.boxplot(ax=ax, x=req['poi'], y=req['distance'])


# In[153]:


# describing the requests with and without outliers
req['distance'] = req['distance'].apply(float)
print('---with outliers---')
print(req.groupby('poi')['distance'].describe())
req_po1 = req.loc[req['poi'] == 'POI1']
req_po3 = req.loc[req['poi'] == 'POI3']
req_po4 = req.loc[req['poi'] == 'POI4']

print('\n','---without outliers---','\n')
print('----for po1----')
req_po1_out = req_po1.loc[req_po1['distance'] < 10000]
print(req_po1_out['distance'].describe())

print('----for po4----')
req_po4_out = req_po4.loc[req_po4['distance'] < 1000]
print(req_po4_out['distance'].describe())

print('----for po3----')
req_po3_out = req_po3.loc[req_po3['distance'] < 600]
print(req_po3_out['distance'].describe())

print('\n','----boxplot----','\n')
req_no_out = req.loc[req['distance'] <= 2000]
fig, ax = plt.subplots(figsize=(8,6))
ax2 = sns.boxplot(ax=ax, x=req_no_out['poi'], y=req_no_out['distance'])


# ### Scaling

# In[154]:


def apply_scaling(min_x, max_x, x):
    a = -10
    b = 10
    factor = (b - a)/(max_x - min_x)
    scaled_val = ((x - min_x) * factor) + a
    return scaled_val


# In[155]:


req_po1_out_scaled = req_po1_out['distance'].apply(lambda x: apply_scaling(req_po1_out['distance'].min(), req_po1_out['distance'].max(), x))
req_po3_out_scaled = req_po3_out['distance'].apply(lambda x: apply_scaling(req_po3_out['distance'].min(), req_po3_out['distance'].max(), x))
req_po4_out_scaled = req_po4_out['distance'].apply(lambda x: apply_scaling(req_po4_out['distance'].min(), req_po4_out['distance'].max(), x))
print(req_po1_out_scaled.describe())
print(req_po3_out_scaled.describe())
print(req_po4_out_scaled.describe())
# val-mean by std


# In[156]:


sc.stop()


# In[ ]:




