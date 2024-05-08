# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:07:28 2024

@author: Customer Churn A.
Müşteriyi kaybetmeden anlayabilir miyiz?
Act fonk:Girişve gizli katmanda linear, çıkış katmanda ise sigmoid kullanmak tavsiye edilir , kural değildir
Girişte katman kaç nörün olcağının bir foormulu olmasada giriş katman artı cıkıs /2 seklınde yaklasık hesap yapılabilir

"""
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme

X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values



#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]




#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Yapay Sinir Ağları
import keras
from keras.models import Sequential # yapay sinir ağı oluşturma
from keras.layers import Dense# yapay sinir ağı katmanı olusturmak için
classifier=Sequential()# İlk yapay sinir ağı
# Katman ve nörünları yerleştirmek
classifier.add(Dense(6,kernel_initializer="uniform",activation="relu",input_dim=11))
classifier.add(Dense(6,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(1,kernel_initializer="uniform",activation="sigmoid"))
# optimizer=Stokastik radyan alçalım değerinin bir versiyonu Sinapsis değerlerin nasıl optimize edileceği
# loss, bir derin öğrenme modelinin eğitimi sırasında belirlenen hedefle gerçek çıktılar arasındaki farkı ölçen bir ölçüdür. 
# metrics parametresi, modelin performansını ölçmek için kullanılan metriklerin bir listesini belirtir
# Örneğin, ["accuracy"] metriği, modelin doğruluğunu ölçer. 
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
# epochs=kaç turda öğrensin?
classifier.fit(X_train,y_train,epochs=50)
y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)


