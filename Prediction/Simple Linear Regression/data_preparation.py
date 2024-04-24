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
missing_data=pd.read_csv("missingValues.csv")

#2.1.missing values
# sci-kit learn

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
age=missing_data.iloc[:,1:4].values
imputer=imputer.fit(age[:,1:4])
# fit fonksiyonu öğrenme yapmasını istediğimiz yapıyı verdik
# eğitmek için kullanılır 
# Eğitim staratejimiz ise ortalama almaktı bunu öğrenecek
age[:,1:4]=imputer.transform(age[:,1:4])
# transform fonksiyonu ise nan değerlerimizi bu ortalamayı koyacak


#2.2.Encoder:Converting categorical data to numerical data
# Burada 2 encoder vardı 1)label 2)OneHot
# Label her değere numeric atama yapar OneHot cloumn
# başlıklarına etiketleri tasımak ve ve her etıket 
# altına 1 /0 ile ait ait değil bilgisi yerleştirmek
country=missing_data.iloc[:,0:1].values
# Burada kategori datamızı çekiyoruz
from sklearn import preprocessing
# Sonra bu kategori datayı numeric değer atıyoruz
le=preprocessing.LabelEncoder()
country[:,0]=le.fit_transform(missing_data.iloc[:,0])
# print(country)
# En sonda kategori veri sayısına göre(county sayısı)
# numeric değerler karşılastırma için anlamlandırılır
ohe=preprocessing.OneHotEncoder()
country=ohe.fit_transform(country).toarray()
print(country)


#2.3.data merging
# numpy dizileri data frame dönüştürme
# ve concat ile dataframe birleştirme
result= pd.DataFrame(data=country,index=range(22),columns=["fr","tr","us"])
print(result)
result2=pd.DataFrame(data=age ,index=range(22),columns=["height","weight","age"])
print(result2)

gender= missing_data.iloc[:,-1].values
print(gender)

result3=pd.DataFrame(data=gender,index=range(22),columns=["gender"])
print(result3)

s=pd.concat([result,result2],axis=1)
print(s)
s2=pd.concat([s,result3],axis=1)
print(s2)

# 2.4. Dividing data for training and testing
# Verilerin eğitim ve test için bölünmesi
# ülke,boy,kilo ve yaştan cinsiyet tahmin etmek istiyoruz
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,result3,test_size=0.33,random_state=0)
# Burada x yetiştidiğimiz sınıf(bağımsız değişken)
#  y ise test sınıfıdır (sonuc değişkeni)
# test_size: Test alt kümesinin oranını belirtir. Bu değer 0 ile 1 arasında olmalıdır ve
#  genellikle 0.2 veya 0.3 gibi bir değer alır. Bu örnekte, test alt kümesinin oranı %33 olarak
#  belirlenmiştir.
# random_state: Veri kümesini rastgele bölerken kullanılan rasgele sayı 
# üreteci için bir tohum değerdir. Belirli bir tohum değeri kullanıldığında, 
# her zaman aynı rastgele bölme elde edilir. Bu, sonuçların tekrarlanabilirliğini sağlar.

#2.5.Attribute scaling
# Verilerin ölçeklenmesi

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
 # Bu işlem, özelliklerin ortalamasını 0'a ve standart sapmasını 1'e dönüştürür.
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

# Eğitim Verisi (Training Data):
# Eğitim verisi, modelin öğrenme sürecinde kullanılan veri kümesidir.
# Model, eğitim verisini kullanarak özellikleri ve hedef değişkenler arasındaki ilişkiyi anlamaya çalışır.
# Model, eğitim verisindeki desenleri tespit eder ve bu desenleri temsil etmek için bir tahmin modeli oluşturur.
# Test Verisi (Test Data):
# Test verisi, modelin performansını değerlendirmek için kullanılan ayrı bir veri kümesidir.
# Model, test verisini kullanarak eğitim sırasında öğrendiği desenleri test eder.
# Test verisi, modelin gerçek dünya verileri üzerinde ne kadar iyi performans gösterdiğini belirlemek için kullanılır.






