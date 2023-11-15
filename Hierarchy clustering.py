# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 09:15:41 2023

@author: Dell i7
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ds = pd.read_csv(r"C:\Users\Dell i7\Desktop\CLUSTRING\Mall_Customers.csv")
ds

X= ds.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

y_hc=hc.fit_predict(x)
y_hc

plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='CLuster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='blue',label='CLuster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='green',label='CLuster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='cyan',label='CLuster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='magenta',label='CLuster 5')
plt.title('CLuster of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.show()
