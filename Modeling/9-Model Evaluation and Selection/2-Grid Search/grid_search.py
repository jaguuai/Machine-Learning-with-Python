# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:17:09 2024

@author: AYSE YILMAZ
SVM Algoritması en iyi parametrelerini bulma
Gris Search
svm
-kernel rbf ? or linear ?
rbf=> gamma?
c error turn ?

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
X=data.iloc[:,[2,3]].values
y=data.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test==sc.transform(X_test)

from sklearn.svm import SVC
classifier=SVC(kernel="rbf",random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#k-fold cross validation
from sklearn.model_selection import cross_val_score
# estimator:hangi alg (bu örnekte classifier)
# 2.X
# 3.y
# 4.cv:kaça katlamalı
cross_val=cross_val_score(estimator=classifier, X=X_train,y=y_train,cv=4)
print(cross_val.mean())
print(cross_val.std()) #standart sapma düşükse iyidi

# Grid SearchCV-Parametre optimizasyon ve algoritma seçimi
from sklearn.model_selection import GridSearchCV
p=[{"C":[1,2,3,4,5],"kernel":["linear"]},
   {"C":[1,10,100,1000],"kernel":["rbf"],
    "gamma":[1,0.5,0.1,0.01,0.001]}]
"""
GS Parametreleri
estimator:sınıflandırma algoritması-optimize etmek istediğimiz algoritma
param_grid:parametreler/tanımlı dizi
scoring:eye göre skorlanacak -> exp: accuracy
cv=kaç katmalı olacak
n_jobs=aynı anda calıscak iş/paraleleştirme
"""
gs=GridSearchCV(estimator=classifier, #Bu örnekte SVM, 
                param_grid=p,
                scoring="accuracy",
                cv=10,
                n_jobs=-1)

grid_search=gs.fit(X_train,y_train)
best_result=grid_search.best_score_
best_param=grid_search.best_params_

print(best_result)
print(best_param)













