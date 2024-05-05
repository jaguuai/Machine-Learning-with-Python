import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("basket.csv",header=None)
t = []
for i in range(len(data.values)):
    t.append([str(data.values[i, j]) for j in range(len(data.values[i]))])
from apyori import apriori
rules=apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)
print(list(rules))