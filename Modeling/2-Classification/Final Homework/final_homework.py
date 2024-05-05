""" 4 özellikten yaprağın hangi sınıfa ait oldupunu bulmaya çalışmak"""
#1-libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#2-data preprocessing
#2.1.data upload
data=pd.read_excel("iris.xls")
print(data)

x=data.iloc[0:,0:4].values # bağımsız değişkenler
y=data.iloc[0:,4:].values # bağımlı değişsken


# 2.4. Dividing data for training and testing

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#2.5.Attribute scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

# Logistic Reg
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)


# confusion matrix
from sklearn.metrics import confusion_matrix
print("logistic Regression")
cm=confusion_matrix(y_test, y_pred)
print(cm)
# error rate:7/8

# K-NN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
print("K-NN")
cm=confusion_matrix(y_test, y_pred)
print(cm)
# n_neighbors=1 olsa en dogru cözüm olurdu

# SVM
from sklearn.svm import SVC
svm=SVC(kernel="linear")
svm.fit(X_train, y_train)
y_pred=svm.predict(X_test)
print("SVM")
cm=confusion_matrix(y_test, y_pred)
print(cm)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive_bayes=GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred=naive_bayes.predict(X_test)
print("Naive Bayes")
cm=confusion_matrix(y_test, y_pred)
print(cm)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
print("Decision Tree")
cm=confusion_matrix(y_test, y_pred)
print(cm)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rand_forest=RandomForestClassifier(n_estimators=10,criterion="entropy")
rand_forest.fit(X_train,y_train)
y_pred=rand_forest.predict(X_test)

# Olassılığın yüzde kaçlık şekilde alındığı
print("Random Forest")
cm=confusion_matrix(y_test, y_pred)
print(cm)
# ROC, FPR ,TPR
y_proba=rand_forest.predict_proba(X_test)
print(y_proba[:,0])

from sklearn import metrics
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:,0], pos_label="iris")
print(fpr)
print(tpr)

