# 1-Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 2-Data preprocessing
# 2.1-Data Upload
data=pd.read_csv("salaries.csv")
x=data.iloc[:,1:2]
X=x.values
print(x)
y=data.iloc[:,2:]
Y=y.values
# linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y)
plt.plot(x,lin_reg.predict(X),color="red")
plt.show()
# linear regresyon 1. derecelidir
# *****************************************
# polynomal regression
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
plt.scatter(X,Y,color="red")
# x,y data point
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
# lin_reg i predict etmek ve gelen değeri polinomal domain e cevirmek
plt.show()

# *************************

poly_reg=PolynomialFeatures(degree=4)
#polinom objesi oluşturur
x_poly=poly_reg.fit_transform(X)
# linear dünyadaki X i polinomal dünyayaa çevirme
print(x_poly)


lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
# x x^2 x^3 al beta kat sayılarını bul demektir
plt.scatter(X,Y,color="green")
# x,y data point
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)),color="black")
# lin_reg i predict etmek ve gelen değeri polinomal domain e cevirmek
plt.show()
# Eğer dereceyi arttırırsak  tahmin daha da iyileşiyor
# predictions
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))


