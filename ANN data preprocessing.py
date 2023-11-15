# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:01:04 2023

@author: Dell i7
"""
pip install matplotlib
## Aritficial neural network
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds=pd.read_csv(r"C:\Users\Dell i7\Desktop\Deep learning\Churn_Modelling.csv")
ds

x=ds.iloc[:,3:13]
y=ds.iloc[:,13]

## create dummy varibles

geography=pd.get_dummies(x["Geography"],drop_first=True)
gender=pd.get_dummies(x["Gender"],drop_first=True)

## concatenate the dataframe

x=pd.concat([x,geography,gender],axis=1)

### drop unnecessary columns
x=x.drop(['Geography','Gender'],axis=1)

##splitting te dataset into training and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

## part 2 now lets make the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

## intialising thw ANN
classifier=Sequential()

##adding the input and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))
#adding the second hdden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

## compliling te ANN
classifier.compile(optimizer='Adagrad',loss='binary_crossentropy',metrics=['accuracy'])
# Fitting the ANN to the Training set
model_history = classifier.fit(x_train, y_train, validation_split=0.33, batch_size=10, epochs=100)

## summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

## summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Part 3 - Making the predictions and evaluating the model

y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

##making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

## calculate the accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac
