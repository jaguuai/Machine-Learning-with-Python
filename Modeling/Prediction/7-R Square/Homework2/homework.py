# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:04:37 2024

@author: Ayşe Yılmaz
-Download the dataset
-Find the necessary/unnecessary arguments
-Create a regression model using 5 different methods
   MLR,PR,SVR,DT,RR
-Compare their methods and their success
-A CE0 with 10 years of experience and 100 points and the same qualifications
Estimate the salaries of a Manager with 5 methods and interpret the results.
**********************
tr:
-Veri kümesini indiriniz
-Gerekli/Gereksiz bağımsız değişkeneri bulunuz
-5 farklı yöntemle regresyon modeli çıkarınız
  MLR,PR,SVR,DT,RR
-Yöntemlerini başarılarını karşılaştırınız
-10 yıl tecribeli ve 100 puan almış bir CE0 ve aynı özelliklere sahip
bir Müdürün maaşlarını 5 yöntemle tahmin edip sonuçları yorumlayınız

"""
# 1-Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm
# 2-Data preprocessing
# 2.1-Data Upload
# 2.2-Data Frame Slice (Dilimleme)
data=pd.read_csv("new_salaries.csv")
print(data)
x=data.iloc[:,2:3]
y=data.iloc[:,5:]

# 2.3-Numpy array conversion
X=x.values
Y=y.values

corr_data=pd.concat([x, y], axis=1)
print(corr_data)
print(corr_data.corr())
# Bu matris kösegen diagon her zaman 1 

# linear regression

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

# Find the necessary/unnecessary arguments
print("linear OLS")
model1=sm.OLS(lin_reg.predict(X),X)
print(model1.fit().summary())
print("Linear Regression R2")
print(r2_score(Y,lin_reg.predict(X)))

# polynomal regression(nonlinear model)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
print(x_poly)

from sklearn.linear_model import LinearRegression
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)





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
# print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print("Poly OLS")
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())
print("Polynomal R2")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))


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
# print(svr_reg.predict([[6.6]]))

print("SVR OLS")
model3=sm.OLS(svr_reg.predict(x_scale),x_scale)
print(model3.fit().summary())
print("SVR R2")
print(r2_score(y_scale,svr_reg.predict(x_scale)))

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(random_state=0)
# train etmek
dt_reg.fit(X, Y)
Z=X+0.5
K=X-0.4

print("DT OLS")
model4=sm.OLS(dt_reg.predict(X),X)
print(model4.fit().summary())
print("Decision Tree R2")
print(r2_score(Y,dt_reg.predict(X)))

# plt.scatter(X,Y,color="red")
# plt.plot(x,dt_reg.predict(X),color="yellow")

# plt.plot(x,dt_reg.predict(Z),color="green")
# plt.plot(x,dt_reg.predict(K),color="blue")
# print(dt_reg.predict([[6.6]]))

# Random Forest Rehression
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
# kaç decision tree çizileceği
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

Z=X+0.5
K=X-0.4

# plt.scatter(X,Y,color="red")
# plt.plot(X,rf_reg.predict(X),color="yellow")

# plt.plot(X,rf_reg.predict(Z),color="green")
# plt.plot(X,rf_reg.predict(K),color="blue")

# R Square
# from sklearn.metrics import r2_score
# tahmin değeri ve gerçek değer arsında ki bağkantıyı bulur

print("RF OLS")
model5=sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())
print("Randomm Forest R2")
print(r2_score(Y,rf_reg.predict(X)))

print(r2_score(Y,rf_reg.predict(K)))
print(r2_score(Y,rf_reg.predict(Z)))

# Summary
print("*********************************")
print("Linear Regression R2")
print(r2_score(Y,lin_reg.predict(X)))
print("*********************************")
print("Polynomal Forest R2")
print(r2_score(Y,lin_reg3.predict(poly_reg3.fit_transform(X))))
print("*********************************")
print("SVR R2")
print(r2_score(y_scale,svr_reg.predict(x_scale)))
print("*********************************")
print("Decision Tree R2")
print(r2_score(Y,dt_reg.predict(X)))
print("*********************************")
print("Random Forest R2")
print(r2_score(Y,rf_reg.predict(X)))





































