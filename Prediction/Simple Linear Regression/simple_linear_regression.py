# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:53:30 2024

@author: Ayse Yilmaz
"""
#1-libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#2-data preprocessing
#2.1.data upload
data=pd.read_csv("sales.csv")

print(data)

mounths=data[["Aylar"]] #bağımsız değişken
print(mounths)
sales=data[["Satislar"]] #bağımli değişken
print(sales)
# 2.4. Dividing data for training and testing
# Verilerin eğitim ve test için bölünmesi

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(mounths,sales,test_size=0.33,random_state=0)
"""
#2.5.Attribute scaling


from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)
"""
#model building
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train) 

prediction=lr.predict(x_test)

#Burada bize dönen prediction values Y test le kıyasladığımızda yaklaşık değerler olduğunu görürüz

#visualization
#önce sıralama yapmalıyız yoksa gorssellestırme hatalı olacaktır
x_train=x_train.sort_index()
y_train=y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title("Sales by Mounths")
plt.xlabel("Mounths")
plt.ylabel("Sales")




