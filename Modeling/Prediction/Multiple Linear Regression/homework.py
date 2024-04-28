# -*- coding: utf-8 -*-
"""
Spyder Editor
Homework1:Tenis oynamkla ilgili model kümseinden çoklu linear regresyon ve geri eleme yapmak
This is a temporary script file.
"""
import pandas as pd
import numpy as np
data=pd.read_csv("play.csv")
print(data)

outlook=data.iloc[:,0:1].values
windy=data.iloc[:,-2].values
play=data.iloc[:,-1].values
from sklearn import preprocessing

le=preprocessing.LabelEncoder()

outlook[:,0]=le.fit_transform(data.iloc[:,0])
ohe=preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()

windy=le.fit_transform(data.iloc[:,-2]).reshape(-1,1)
ohe=preprocessing.OneHotEncoder()
windy=ohe.fit_transform(windy).toarray()

play=le.fit_transform(data.iloc[:,-1]).reshape(-1,1)
ohe=preprocessing.OneHotEncoder()
play=ohe.fit_transform(play).toarray()

result= pd.DataFrame(data=outlook,index=range(14),columns=["sunny","overcast","rainy"])
print(result)
result2=pd.DataFrame(data=data.iloc[:,2:3],index=range(14),columns=["humidity"])
print(result2)
result3=pd.DataFrame(data=windy[:,:1],index=range(14),columns=["windy"])
print(result3)
result4=pd.DataFrame(data=play[:,:1],index=range(14),columns=["play"])

sum1=pd.concat((result,result2),axis=1)
print(sum1)
sum2=pd.concat((sum1,result3),axis=1)
print(sum2)
sum3=pd.concat((sum2,result4),axis=1)
print(sum3)

temperature=data.iloc[:,2:3]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sum2,temperature,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test) #tahmin bulma

# model success
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int), values=data,axis=1)


X_list=data.iloc[:,[0,1,2,3,4]].values
X_list=np.array(X_list,dtype=float)

model=sm.OLS(temperature,X_list).fit()
