import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Ads_CTR_Optimisation.csv")

import random
N=10000
d=10
sumAd=0
choes=[]
for n in range(0,N):
    ad = random.randrange(d)
    choes.append(d)
    reward = data.values[n,ad]
    # verilerdeki n. satır 1 ise ödül 1 
    sumAd = sumAd + reward
plt.hist(choes)
plt.show()

