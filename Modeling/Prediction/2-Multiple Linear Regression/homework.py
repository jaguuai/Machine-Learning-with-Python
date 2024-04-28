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
# 1.yol
# outlook=data.iloc[:,0:1].values
# windy=data.iloc[:,-2].values
# play=data.iloc[:,-1].values
# from sklearn import preprocessing

# le=preprocessing.LabelEncoder()

# outlook[:,0]=le.fit_transform(data.iloc[:,0])
# ohe=preprocessing.OneHotEncoder()
# outlook=ohe.fit_transform(outlook).toarray()

# windy=le.fit_transform(data.iloc[:,-2]).reshape(-1,1)
# ohe=preprocessing.OneHotEncoder()
# windy=ohe.fit_transform(windy).toarray()

# play=le.fit_transform(data.iloc[:,-1]).reshape(-1,1)
# ohe=preprocessing.OneHotEncoder()
# play=ohe.fit_transform(play).toarray()
from sklearn import preprocessing
data2=data.apply(preprocessing.LabelEncoder().fit_transform)
c=data2.iloc[:,:1]
ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

weather= pd.DataFrame(data=c,index=range(14),columns=["sunny","overcast","rainy"])
print(weather,"resulthh")
data_end=pd.concat([weather,data.iloc[:,1:3]],axis=1)
data_end=pd.concat([data2.iloc[:,-2:],data_end],axis=1)
print(data_end)



from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(data_end.iloc[:,:-1],data_end.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)
y_pred=regressor.predict(x_test) #tahmin bulma

# model success
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int), values=data_end.iloc[:,:-1],axis=1)
X_list=data_end.iloc[:,[0,1,2,3,4,5]].values
X_list=np.array(X_list,dtype=float)
model=sm.OLS(data_end.iloc[:,-1:],X_list).fit()
print(model.summary())
back_elimination=data_end.iloc[:,1:]

X_list=back_elimination.iloc[:,[0,1,2,3,4]].values
X_list=np.array(X_list,dtype=float)
model=sm.OLS(data_end.iloc[:,-1:],X_list).fit()
print(model.summary())

x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
regressor.fit(x_train, y_train)
y_pred=regressor.predict(x_test)