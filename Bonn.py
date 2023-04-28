#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[2]:


#data  
dataset = pd.read_csv('D:/image/CC GENERAL.csv')

dataset.head()


# In[3]:


dataset.shape


# In[4]:


dataset.describe()


# In[5]:


dataset.dtypes


# In[6]:


dataset.isnull().sum()


# In[7]:


dataset['CREDIT_LIMIT'].fillna(value=dataset['CREDIT_LIMIT'].median(), inplace=True)

dataset['MINIMUM_PAYMENTS'].fillna(value=dataset['MINIMUM_PAYMENTS'].median(), inplace=True)

dataset.drop(["CUST_ID"],axis=1, inplace=True)


# In[8]:


correlations = dataset.corr()


# In[9]:


corr = dataset.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[10]:


dataset.plot(x='PURCHASES', y='PAYMENTS', style='*')


# In[11]:


scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(dataset)


# In[12]:


dataset_scaled.shape


# In[13]:


dataset_scaled


# In[15]:


wcss = []
K = range(1, 20)
for k in K:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(dataset_scaled)
    wcss.append(kmeans.inertia_)


    
plt.plot(wcss, 'bx-')
plt.title("finding the optimum number of clusters")
plt.xlabel("number of clusters")
plt.ylabel("WCSS scores")
plt.show()


# In[16]:


kmeans = KMeans(8)
kmeans.fit(dataset_scaled)
labels = kmeans.labels_


# In[17]:


kmeans.cluster_centers_.shape


# In[18]:


cluster_centers_dataset = pd.DataFrame(data = kmeans.cluster_centers_, columns=[dataset.columns])
cluster_centers_dataset


# In[19]:


cluster_centers = scaler.inverse_transform(cluster_centers_dataset)
cluster_centers_dataset = pd.DataFrame(data=cluster_centers, columns=[dataset.columns])
cluster_centers_dataset


# In[26]:


labels.shape


# In[27]:


labels.max()


# In[28]:


y_kmeans = kmeans.fit_predict(dataset_scaled)
y_kmeans


# In[30]:


dataset_clusters = pd.concat([dataset, pd.DataFrame({'cluster': labels})], axis=1)
dataset_clusters.head()


# In[31]:


for col in dataset.columns:
  plt.figure(figsize=(28, 7))
  for i in range(8): # number of clusters
    plt.subplot(1, 8, i+1)
    cluster = dataset_clusters[dataset_clusters['cluster'] == i]
    cluster[col].hist(bins=20)
    plt.title(f"{col} \nCluster {i}")

plt.show()


# In[36]:


pca = PCA(n_components=2)
principal_comp = pca.fit_transform(dataset_scaled)
principal_comp_dataset = pd.DataFrame(principal_comp, columns=['principal_component_1', 'principal_component_2'])
principal_comp_dataset.head()


# In[38]:


pc_dataset_clusters = pd.concat([principal_comp_dataset, pd.DataFrame({'cluster': labels})], axis=1)
pc_dataset_clusters.head()


# In[39]:


plt.figure(figsize=(10, 10))
ax = sns.scatterplot(data=pc_dataset_clusters, hue='cluster', x='principal_component_1', y='principal_component_2')
ax.set(xlabel="Principal Component 1", ylabel="Principal Component 2")


# In[ ]:




