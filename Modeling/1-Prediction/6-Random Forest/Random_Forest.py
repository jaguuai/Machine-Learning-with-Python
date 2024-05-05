# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:20:19 2024
Polynomal regression templete
@author: ADMIN
"""
# 1-Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 2-Data preprocessing
# 2.1-Data Upload
# 2.2-Data Frame Slice (Dilimleme)
data=pd.read_csv("salaries.csv")
x=data.iloc[:,1:2]
y=data.iloc[:,2:]
# 2.3-Numpy array conversion
X=x.values
Y=y.values

# linear regression
# linear regresyon 1. derecelidir
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

# polynomal regression(nonlinear model)
# Aslında çok değişkenli çok dereceli linear regresyon gibi düşünülebilir
# denkleminde gördüğümüz üzere

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
#polinom objesi oluşturur
x_poly=poly_reg.fit_transform(X)
# linear dünyadaki X i polinomal dünyayaa çevirme
print(x_poly)

from sklearn.linear_model import LinearRegression
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
# x x^2 x^3 al beta kat sayılarını bul demektir

# *************************
# polynomial equation with more degrees
poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(X)


lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)

# data visualition
# plt.scatter(X,Y)
# plt.plot(x,lin_reg.predict(X),color="red")
# plt.show()


# plt.scatter(X,Y,color="red")
# # x,y data point
# plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
# # lin_reg i predict etmek ve gelen değeri polinomal domain e cevirmek
# plt.show()


# plt.scatter(X,Y,color="green")
# plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(X)),color="black")
# plt.show()

# predictions
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

# Data Scaling
# SVR Regression da veriler scal edilmek zorundadır
# Karar ağaçlarında scale edilmez
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_scale=sc1.fit_transform(X)
sc2=StandardScaler()
y_scale=np.ravel(sc1.fit_transform(Y.reshape(-1,1)))

from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_scale,y_scale)

# plt.scatter(x_scale,y_scale)
# plt.plot(x_scale,svr_reg.predict(x_scale))

print(svr_reg.predict([[6.6]]))

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(random_state=0)
# train etmek
dt_reg.fit(X, Y)
Z=X+0.5
K=X-0.4

# plt.scatter(X,Y,color="red")
# plt.plot(x,dt_reg.predict(X),color="yellow")

# plt.plot(x,dt_reg.predict(Z),color="green")
# plt.plot(x,dt_reg.predict(K),color="blue")
print(dt_reg.predict([[6.6]]))
# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
# kaç decision tree çizileceği
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

Z=X+0.5
K=X-0.4

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="yellow")

plt.plot(X,rf_reg.predict(Z),color="green")
plt.plot(X,rf_reg.predict(K),color="blue")








