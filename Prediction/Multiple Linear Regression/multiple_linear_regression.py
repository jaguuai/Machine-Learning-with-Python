# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:53:30 2024
The aim of the project is to predict gender from country, height and weight variables
@author: Ayse Yilmaz
"""
#1-libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#2-data preprocessing
#2.1.data upload
data=pd.read_csv("data.csv")
print(data)


#2.2.Encoder:Converting categorical data to numerical data

country=data.iloc[:,0:1].values

from sklearn import preprocessing

le=preprocessing.LabelEncoder()
country[:,0]=le.fit_transform(data.iloc[:,0])
print(country)

ohe=preprocessing.OneHotEncoder()
country=ohe.fit_transform(country).toarray()
print(country)

# Gender categoric to numeric
gender=data.iloc[:,-1:].values

from sklearn import preprocessing


gender[:,-1]=le.fit_transform(data.iloc[:,-1])
print(gender)

ohe=preprocessing.OneHotEncoder()
gender=ohe.fit_transform(gender).toarray()
print(gender)

#2.3.data merging

result= pd.DataFrame(data=country,index=range(22),columns=["fr","tr","us"])
print(result)

age = data.iloc[:,1:4].values
result2=pd.DataFrame(data=age ,index=range(22),columns=["height","weight","age"])
print(result2)

result3=pd.DataFrame(data=gender[:,:1],index=range(22),columns=["gender"])
print(result3)

s=pd.concat([result,result2],axis=1)
print(s)
s2=pd.concat([s,result3],axis=1)
print(s2)

# 2.4. Dividing data for training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,result3,test_size=0.33,random_state=0)


# #2.5.Attribute scaling


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

# 3.Modeling
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test) #tahmin bulma


#*********************** boy bulma

height=s2.iloc[:,3:4].values
print(height)

left=s2.iloc[:,:3]
right=s2.iloc[:,4:]

new_data=pd.concat([left,right],axis=1)

x_train,x_test,y_train,y_test=train_test_split(new_data,height,test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression
r2=LinearRegression()
r2.fit(x_train, y_train)

y_pred=r2.predict(x_test) #tahmin bulma

# model success
import statsmodels.api as sm
X=np.append(arr=np.ones((22,1)).astype(int), values=new_data,axis=1)
# Bir sutun içine dizi ekliyoruz 1 lerden oluşan yani bu dizi aslında denklemdeki beta0 dır

X_list=new_data.iloc[:,[0,1,2,3,4,5,]].values
X_list=np.array(X_list,dtype=float)
# En başta tamamını alıyoruz Çünkü hepsinin en bastakı p-values hesaplamak istiyoruz
model=sm.OLS(height,X_list).fit()
# istatistiksel değelerimizi çıkarır Boya bakarak bağımsız değişkenleri içeren dizinin ne kaadr etkisini olduğunu ölçmeye yarar 
print(model.summary())
# sonuca bakınca p-values en yüksek değer x 5 tir yani 4. eleman Bunu eleriz

X_list=new_data.iloc[:,[0,1,2,3,5]].values
X_list=np.array(X_list,dtype=float)
model=sm.OLS(height,X_list).fit()
print(model.summary())

X_list=new_data.iloc[:,[0,1,2,3]].values
X_list=np.array(X_list,dtype=float)
model=sm.OLS(height,X_list).fit()
print(model.summary())