#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[84]:


#read data 
data = pd.read_csv('Unemployment in India.csv')


# In[85]:


print(data.head())


# In[86]:


data.info


# In[87]:


print(data.describe)


# In[88]:


#null values


# In[89]:


print(data.isnull().sum())


# In[90]:


print(data.describe)


# In[91]:


data.columns =["State","Date","Frequency","EstUR","Esti Emp","ELPR","Region"]


# In[92]:


print(data)


# In[93]:


#plotting data

plt.style.use("seaborn-whitegrid")
plt.figure(figsize=(6,5))
sns.heatmap(data.corr())
plt.show()


# In[94]:


#esti no. of emp acc to regions 
data.columns =["State","Date","Frequency","EstUR","Esti Emp","ELPR","Region"]
plt.title("Indian Unemloyment")
sns.histplot(x="Esti Emp",hue="Region",data=data)
plt.show()


# In[74]:


plt.figure(figsize=(6,5))
plt.title("Indian Unemloyment")
sns.histplot(x="EstUR",hue="Region",data=data) 
plt.show()


# In[97]:


plt.figure(figsize=(12,12))
plt.title("Indian Unemloyment")
sns.histplot(x="EstUR",hue="State",data=data) 
plt.show()


# In[77]:


#dashboard(unemployment rate of each indian state by region: sunburst plot)


# In[106]:


Unemployment = data[[ "EstUR","Region","State"]]


# In[107]:


figure = px.sunburst(Unemployment,values="EstUR",path=["Region","State"],width=700,height=700,color_continuous_scale="BrBG",title="Unemployment rate in India")
figure.show()


# In[ ]:




