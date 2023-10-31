# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:07:56 2023

@author: Dell i7
"""

from IPython.display import Image
url = 'https://img.etimg.com/thumb/msid-71806721,width-650,imgsize-807917,,resizemode-4,quality-100/avocados.jpg'
Image(url,height=300,width=400)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

ds=pd.read_csv(r"C:\Users\Dell i7\Downloads\data\avocado.csv")

ds.info()
ds.head(3)
sns.distplot(ds['AveragePrice']);

sns.countplot(x='year',data=ds,hue='type')
ds.year.value_counts()

sns.boxplot(y="type",x="AveragePrice",data=ds)

ds.year=ds.year.apply(str)
sns.boxenplot(x="year",y="AveragePrice",data=ds)

ds['type']=ds['type'].map({'conventional':0,'organic':1})

ds.Date = ds.Date.apply(pd.to_datetime)
ds['Month']=ds['Date'].apply(lambda x:x.month)
ds.drop('Date',axis=1,inplace=True)
ds.Month = ds.Month.map({1:'JAN',2:'FEB',3:'MARCH',4:'APRIL',5:'MAY',6:'JUNE',7:'JULY',8:'AUG',9:'SEPT',10:'OCT',11:'NOV',12:'DEC'})

plt.figure(figsize=(9,5))
sns.countplot(x=ds['Month'])
plt.title('Monthwise Distribution of sales',fontdict={'fontsize':25})


dummies = pd.get_dummies(ds[['year','region','Month',]],drop_first=True)
df_dummies =pd.concat([ds[['Total Volume','4046','4225','4770','Total Bags',
                           'Small Bags','Large Bags','XLarge Bags','type']],dummies],axis=1)
target = ds ['AveragePrice']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_dummies,target,test_size=0.30)

cols_to_std = ['Total Volume', '4046', '4225', '4770', 'Total Bags', 'Small Bags','Large Bags', 'XLarge Bags']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train[cols_to_std])
X_train[cols_to_std] = scaler.transform(X_train[cols_to_std])
X_test[cols_to_std] = scaler.transform(X_test[cols_to_std])



from  sklearn.linear_model import LinearRegression
from sklearn .tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


regressor={
    'Linear Regression': LinearRegression(),
    'Descision Tree': DecisionTreeRegressor(),
    'Random Forest':RandomForestRegressor(),
    'Support Vector Machine':SVR(gamma=1),
    'K-nearest Neighbors': KNeighborsRegressor(n_neighbors=1),
    'XGBoost':XGBRegressor()
    }

results=pd.DataFrame(columns=['MAE','MSE','R2-score'])
for method,func in regressor.items():
    model = func.fit(X_train,y_train)
    pred = model.predict(X_test)
    results.loc[method]= [np.round(mean_absolute_error(y_test,pred),3),
                          np.round(mean_squared_error(y_test,pred),3),
                          np.round(r2_score(y_test,pred),3)
                         ]