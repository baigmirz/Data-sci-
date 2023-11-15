# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:56:30 2023

@author: Dell i7
"""


# Artiticial neural network

import numpy as np 
import pandas as pd
import tensorflow as tf
tf.__version__


## part 1 Data preprocessing

ds=pd.read_csv(r"C:\Users\Dell i7\Desktop\Deep learning\Churn_Modelling.csv")
ds

x=ds.iloc[:,3:-1].values
y=ds.iloc[:,-1].values
print(x)
print(y)

## encoding cateroical data
pip install scikit-learn

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct =ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))                      
print(x)

## feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
print(x)

## splitting the dataset into training set & test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

## part2
## intializing the ANN
ann=tf.keras.models.Sequential()

## Adding the inputlayer the first hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=5,activation='relu'))
ann.add(tf.keras.layers.Dense(units=4,activation='relu'))

## adding the output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

### parts 3 training the ann
### compiling the ANN

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

### training the Ann on the training set
ann.fit(x_train,y_train,batch_size=32,epochs=250)

## parts 4making the predictions evalutig the modle
## predicting the test set result
y_pred=ann.predict(x_test)
y_pred=(y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

## making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
