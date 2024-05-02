# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:53:30 2024
Classification-> gender prediction from age,height and weight
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

x=data.iloc[5:,1:4].values # bağımsız değişkenler
y=data.iloc[5:,4:].values # bağımlı değişsken


# 2.4. Dividing data for training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#2.5.Attribute scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)

y_pred=log_reg.predict(X_test)
print(y_pred)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
# error rate:7/8

# K-NN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric="minkoxski")
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
# n_neighbors=1 olsa en dogru cözüm olurdu





