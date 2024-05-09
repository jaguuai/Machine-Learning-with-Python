# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:03:34 2024

@author:Ayşe Yılmaz
13 kolonu daha az sayıda kolona indirebilir miyiz ?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("wine.csv")
print(data)
X=data.iloc[:,0:13].values
y=data.iloc[:,13].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

# PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)

X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

# PCA dönüşümünden önce gelen logistic regrssion
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
# PCA dönüşümünden gelen logistic regrssion
from sklearn.linear_model import LogisticRegression
classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train2, y_train)

# tahminler
y_pred=classifier.predict(X_test)
y_pred2=classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
# actual/PCA olmadan sonuç
cm=confusion_matrix(y_test,y_pred)
print(cm)
# actual/PCA olduğunda sonuç
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)
# PCA sonrası ve öncesi karşılaştırma
cm3=confusion_matrix(y_test,y_pred2)
print(cm3)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)

X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)

# LDA dönüşümünden sonra
classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)
# LDA tahmini
y_pred_lda=classifier_lda.predict(X_test_lda)

# LDA sonrası ve orj karşılaştırma
cm4=confusion_matrix(y_pred,y_pred_lda)
print(cm4)





