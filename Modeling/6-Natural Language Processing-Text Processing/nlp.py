import pandas as pd
import numpy as np

reviews=pd.read_csv("restaurant_reviews.csv")

# 1-Preprocessing
import re
import nltk

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

stop=nltk.download("stopwords")
from nltk.corpus import stopwords
# corpus bir derlem bir çok kütüphane içeren

new_review=[]
for i in range(1000):
    # AlfaNumerik karakter filtreleme
    review=re.sub("[^a-zA-Z]"," ",reviews["Review"][i])
    # Büyük-küçük harf
    review=review.lower()
    # Kelimeleri python list çevirme
    review=review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review=" ".join(review)
    new_review.append(review)   

# 2-Feature Engineering
# Bag of Words(BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
X=cv.fit_transform(new_review).toarray()   #bağımsız değişken
#y ekseni 1000 x ekseni 2000 Her kelime için o yorumda var mı yokk mu taraması
# Tek kolon 2000 kolona dönüştü
reviews.fillna(0, inplace=True)
y=reviews.iloc[:,1].values #bağımlı değişken


# 3-Machine Learning


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm) # %72.5 accuracy 





