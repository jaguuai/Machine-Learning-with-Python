"""
Created on Sun May  5 03:40:05 2024

@author: ADMIN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("customers.csv")
print(data)

x=data.iloc[:,3:].values

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=4,init="k-means++")
kmeans.fit(x)

print(kmeans.cluster_centers_)
results=[]
for i in range(1,10):
     kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
     # amaç farklı cluster sayısı denemek
     kmeans.fit(x)
     results.append(kmeans.inertia_)
     # inertia wcss değridir
plt.plot((range(1,10)), results)
plt.show()

kmeans=KMeans(n_clusters=4,init="k-means++",random_state=123)
y_pred=kmeans.fit_predict(x)
print(y_pred)
plt.scatter(x[y_pred==0,0], x[y_pred==0,1] ,s=100,c="red")
plt.scatter(x[y_pred==1,0], x[y_pred==1,1] ,s=100,c="green")
plt.scatter(x[y_pred==2,0], x[y_pred==2,1] ,s=100,c="blue")
plt.scatter(x[y_pred==3,0], x[y_pred==3,1] ,s=100,c="yellow")
plt.show()

# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
agg_clus=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
y_pred=agg_clus.fit_predict(x)
print(y_pred)
# Yukarıda 3 cluster ayarladık ve burada y_pred bze hangi data hangi cluster da gösterdi
plt.scatter(x[y_pred==0,0], x[y_pred==0,1] ,s=100,c="red")
plt.scatter(x[y_pred==1,0], x[y_pred==1,1] ,s=100,c="green")
plt.scatter(x[y_pred==2,0], x[y_pred==2,1] ,s=100,c="blue")
plt.show()
# Bu çizimle yukarıda ki 3 clusterı görselleştirdik
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.show()
