# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:04:07 2023

@author: Dell i7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ds=pd.read_csv(r"C:\Users\Dell i7\Desktop\CLUSTRING\Mall_Customers.csv")
ds

x = ds.iloc[:,[3,4]].values

from sklearn.cluster import KMeans

wcss=[]


for i in range(1,11):
    Kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    Kmeans.fit(x)
    wcss.append(Kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of CLusters')
plt.ylabel('WCSS')
plt.show()    
### kmeans model dataset
Kmeans=KMeans(n_clusters=5,init='k-means++',random_state=42)
y_Kmeans= Kmeans.fit_predict(x)

### visualising the cluster

plt.scatter(x [y_Kmeans==0,0],x[y_Kmeans==0,1],s=100,c='red',label='CLuster 1')
plt.scatter(x[y_Kmeans==1,0],x[y_Kmeans==1,1],s=100,c='blue',label='CLuster 2')
plt.scatter(x[y_Kmeans==2,0],x[y_Kmeans==2,1],s=100,c='green',label='CLuster 3')
plt.scatter(x[y_Kmeans==3,0],x[y_Kmeans==3,1],s=100,c='cyan',label='CLuster 4')
plt.scatter(x[y_Kmeans==4,0],x[y_Kmeans==4,1],s=100,c='magenta',label='CLuster 5')
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Cluster of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.show()