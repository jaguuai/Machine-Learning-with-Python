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
print(cross_val.std()) #standart sapma düşükse iyidir