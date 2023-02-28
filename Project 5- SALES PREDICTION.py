#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Name : Amodi Baakle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Advertising.csv")
print(data.shape)
print(data.head())


# In[9]:


data.info


# In[10]:


data.describe()


# In[6]:


print(data.isnull().sum())


# In[11]:


plt.figure(figsize=(6,6))
sns.histplot(data['TV'])
plt.show()


# In[12]:


plt.figure(figsize=(6,6))
sns.histplot(data['Radio'])
plt.show()


# In[13]:


plt.figure(figsize=(6,6))
sns.distplot(data['Newspaper'])
plt.show()


# In[3]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr())
plt.show()


# In[4]:


x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

data = pd.DataFrame(data={"Predicted Sales": ypred.flatten()})
print(data)


# In[ ]:




