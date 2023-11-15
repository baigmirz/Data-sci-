# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:04:43 2023

@author: Dell i7
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


ds = pd.read_csv(r"C:\Users\Dell i7\Desktop\ML\Social_Network_Ads.csv")
ds

x=ds.iloc[:,2:4].values
y=ds.iloc[:,-1]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn .model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.svm import SVC
classifier=SVC()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
y_pred

from sklearn.metrics import  confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
ac

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier, X=x_train,y=y_train,cv=5)
print("Accuracy:{:.2f}%".format(accuracies.mean()*100))

# grid search
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
            {'c':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]
grid_search=GridSearchCV(estimator=classifier, param_grid=parameters,
                         scoring='accuracy',
                         cv=5,
                         )

grid_search = grid_search.fit(x_train,y_train)
best_accuracy= grid_search.best_score_
best_parametes
