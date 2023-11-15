# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 20:57:03 2023

@author: Dell i7
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline

ds = pd.read_csv(r"C:\Users\Dell i7\Desktop\ML\Social_Network_Ads.csv")
ds

x = ds.iloc[:, [2, 3]].values
y = ds.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(  x, y, test_size=0.15, random_state=0)
    

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



from sklearn.svm import SVC
classifier = SVC( kernel="sigmoid",gamma="scale",degree=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
ac

bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test, y_test)
variance
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min()-1, stop=x_set[:, 0].max()+1, step=0.01),
                     np.arange(start=x_set[:, 1].min()-1, stop=x_set[:, 1].max()+1, step=0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('SVM(Training set)')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
x_set,y_set = x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],
                c = ListedColormap(('red', 'green'))(i), label = j)

plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()                        
