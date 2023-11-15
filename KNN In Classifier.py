# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 11:57:20 2023

@author: Dell i7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

ds = pd.read_csv(r"C:\Users\Dell i7\Desktop\ML\Social_Network_Ads.csv")
ds

x = ds.iloc[:,[2,3]].values
x
y = ds.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=6,weights='uniform',algorithm='auto')
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac


bias = classifier.score(x_train,y_train)
bias

variance= classifier.score(x_test,y_test)
variance

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
cr




from matplotlib.colors import ListedColormap
x_set,y_set = x_test, y_test
x1,x2 = np.meshgrid(np.arange(start=x_set[:,0]).min()-1,stop=x_set[:,0].max()+1,step=0.01),
np.arange(start = x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.0)
plt.contour(x1,x2,classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1, shape),
alpha = 0.75,cmap = ListedColormap(('red','green'))(i),label= j)
lt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)



plt.title('K-NN(Training set)')
plt.xlabel('age')
plt.ylabel('Estimated Salary')             
plt.legend()
plt.show()