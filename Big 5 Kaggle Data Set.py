#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

data = pd.read_csv('C:/Users/Administrator/Downloads/big5/IPIP-FFM-data-8Nov2018/data-final.csv', delimiter='\t')
data


# In[26]:


big_five_data = data.copy()

big_five_data.drop(big_five_data.columns[50:107], axis=1, inplace=True)
big_five_data.drop(big_five_data.columns[51:], axis=1, inplace=True)
big_five_data.dropna(inplace=True)
big_five_data.head()


# In[27]:


#note: 
#EXT - Extraversion
#EST - Neuroticism
#AGR - Agreeableness
#CSN - Conscientiousness
#OPN - Openness to experience


# In[28]:


#Aggregate Scores


# In[29]:


big_five_data.loc[:, 'EXT'] = 0
big_five_data.loc[:, 'EST'] = 0
big_five_data.loc[:, 'AGR'] = 0
big_five_data.loc[:, 'CSN'] = 0
big_five_data.loc[:, 'OPN'] = 0

big_five_data.loc[:, 'EXT'] = big_five_data.loc[:, 'EXT1'] + big_five_data.loc[:, 'EXT2'] + big_five_data.loc[:, 'EXT3'] + big_five_data.loc[:, 'EXT4']                    + big_five_data.loc[:, 'EXT5'] + big_five_data.loc[:, 'EXT6'] + big_five_data.loc[:, 'EXT7'] + big_five_data.loc[:, 'EXT8']                 + big_five_data.loc[:, 'EXT9'] + big_five_data.loc[:, 'EXT10']
big_five_data.loc[:, 'EST'] = big_five_data.loc[:, 'EST1'] + big_five_data.loc[:, 'EST2'] + big_five_data.loc[:, 'EST3'] + big_five_data.loc[:, 'EST4']                    + big_five_data.loc[:, 'EST5'] + big_five_data.loc[:, 'EST6'] + big_five_data.loc[:, 'EST7'] + big_five_data.loc[:, 'EST8']                 + big_five_data.loc[:, 'EST9'] + big_five_data.loc[:, 'EST10']
big_five_data.loc[:, 'AGR'] = big_five_data.loc[:, 'AGR1'] + big_five_data.loc[:, 'AGR2'] + big_five_data.loc[:, 'AGR3'] + big_five_data.loc[:, 'AGR4']                    + big_five_data.loc[:, 'AGR5'] + big_five_data.loc[:, 'AGR6'] + big_five_data.loc[:, 'AGR7'] + big_five_data.loc[:, 'AGR8']                 + big_five_data.loc[:, 'AGR9'] + big_five_data.loc[:, 'AGR10']
big_five_data.loc[:, 'CSN'] = big_five_data.loc[:, 'CSN1'] + big_five_data.loc[:, 'CSN2'] + big_five_data.loc[:, 'CSN3'] + big_five_data.loc[:, 'CSN4']                    + big_five_data.loc[:, 'CSN5'] + big_five_data.loc[:, 'CSN6'] + big_five_data.loc[:, 'CSN7'] + big_five_data.loc[:, 'CSN8']                 + big_five_data.loc[:, 'CSN9'] + big_five_data.loc[:, 'CSN10']
big_five_data.loc[:, 'OPN'] = big_five_data.loc[:, 'OPN1'] + big_five_data.loc[:, 'OPN2'] + big_five_data.loc[:, 'OPN3'] + big_five_data.loc[:, 'OPN4']                    + big_five_data.loc[:, 'OPN5'] + big_five_data.loc[:, 'OPN6'] + big_five_data.loc[:, 'OPN7'] + big_five_data.loc[:, 'OPN8']                 + big_five_data.loc[:, 'OPN9'] + big_five_data.loc[:, 'OPN10']

big_five_data.head()


# In[30]:


#Extraversion grouped by value
low = 50.0 / 3
len_big_five_data = len(big_five_data)
low_EXT = len(big_five_data[big_five_data.EXT <= low]) / len_big_five_data

medium = 50.0 * 2 / 3

medium_EXT = len(big_five_data[(big_five_data.EXT > low) & (big_five_data.EXT <= medium)]) / len_big_five_data

high_EXT = len(big_five_data[big_five_data.EXT > medium]) / len_big_five_data

plt.figure(figsize=(10,6))
plt.title("Extraversion Value")
sns_diag = sns.barplot(x=['Low', 'Medium', 'High'], y=[low_EXT, medium_EXT, high_EXT])
sns_diag.set(ylim=(0, 1))


# In[31]:


#Neuroticism grouped by value
low_EST = len(big_five_data[big_five_data.EST <= low]) / len_big_five_data
medium_EST = len(big_five_data[(big_five_data.EST > low) & (big_five_data.EST <= medium)]) / len_big_five_data
high_EST = len(big_five_data[big_five_data.EST > medium]) / len_big_five_data
plt.figure(figsize=(10,6))
plt.title("Neuroticism Value")
sns_diag = sns.barplot(x=['Low', 'Medium', 'High'], y=[low_EST, medium_EST, high_EST])
sns_diag.set(ylim=(0, 1))


# In[32]:


#Agreeableness grouped by value
low_AGR = len(big_five_data[big_five_data.AGR <= low]) / len_big_five_data
medium_AGR = len(big_five_data[(big_five_data.AGR > low) & (big_five_data.AGR <= medium)]) / len_big_five_data
high_AGR = len(big_five_data[big_five_data.AGR > medium]) / len_big_five_data
plt.figure(figsize=(10,6))
plt.title("Agreeableness Value")
sns_diag = sns.barplot(x=['Low', 'Medium', 'High'], y=[low_AGR, medium_AGR, high_AGR])
sns_diag.set(ylim=(0, 1))


# In[33]:


#Conscientiousness grouped by value
low_CSN = len(big_five_data[big_five_data.CSN <= low]) / len_big_five_data
medium_CSN = len(big_five_data[(big_five_data.CSN > low) & (big_five_data.CSN <= medium)]) / len_big_five_data
high_CSN = len(big_five_data[big_five_data.CSN > medium]) / len_big_five_data
plt.figure(figsize=(10,6))
plt.title("Conscientiousness Value")
sns_diag = sns.barplot(x=['Low', 'Medium', 'High'], y=[low_CSN, medium_CSN, high_CSN])
sns_diag.set(ylim=(0, 1))


# In[34]:


#Openness to Experience grouped by value
low_OPN = len(big_five_data[big_five_data.OPN <= low]) / len_big_five_data
medium_OPN = len(big_five_data[(big_five_data.OPN > low) & (big_five_data.OPN <= medium)]) / len_big_five_data
high_OPN = len(big_five_data[big_five_data.OPN > medium]) / len_big_five_data
plt.figure(figsize=(10,6))
plt.title("Openness Value")
sns_diag = sns.barplot(x=['Low', 'Medium', 'High'], y=[low_OPN, medium_OPN, high_OPN])
sns_diag.set(ylim=(0, 1))


# In[35]:


main_big5_data = big_five_data.copy()
main_big5_data = main_big5_data.iloc[:, 51:]
main_big5_data.loc[:, 'ACRS'] = ''
name_columns = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
var_degree = ['1', '2', '3']  #  1 - Low, 2 - Medium, 3 - High
main_big5_data.head()


# In[42]:


big_five_data.head(10)


# In[56]:


big_five_data.drop(columns=['country']).head(10)


# In[57]:


kmeans = KMeans(n_clusters=5)
k_fit = kmeans.fit(big_five_data.drop(columns=['country']))


# In[58]:


# Predicting the Clusters
pd.options.display.max_columns = 10

#labels_ is used to identify Labels of each point
predictions = k_fit.labels_
big_five_data['Clusters'] = predictions
big_five_data.head(10)


# In[59]:


big_five_data.groupby('Clusters').mean()


# In[63]:


big_five_data.drop(big_five_data.columns.difference(['EXT','EST', 'AGR','CSN','OPN','Clusters']), 1, inplace=True)

# Visualizing the means for each cluster
data_clusters = big_five_data.groupby('Clusters').mean()
plt.figure(figsize=(22,3))
for i in range(0, 5):
    plt.subplot(1,5,i+1)
    plt.bar(data_clusters.columns, data_clusters.iloc[:, i], color='blue', alpha=0.2)
    plt.plot(data_clusters.columns, data_clusters.iloc[:, i], color='red')
    plt.title('Cluster ' + str(i))
    plt.xticks(rotation=45)
    plt.ylim(0,4);


# In[66]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_fit = pca.fit_transform(big_five_data)

data_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])
data_pca['Clusters'] = predictions
data_pca.head()

plt.figure(figsize=(10,10))
sns.scatterplot(data=data_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.9)
plt.title('Personality Clusters after PCA');


# In[ ]:




