import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Ads_CTR_Optimisation.csv")

# Random Selection
# import random
# N=10000
# d=10
# sumAd=0
# choes=[]
# for n in range(0,N):
#     ad = random.randrange(d)
#     choes.append(d)
#     reward = data.values[n,ad]
#     # verilerdeki n. satır 1 ise ödül 1 
#     sumAd = sumAd + reward
# plt.hist(choes)
# plt.show()

# UCB
# import math

# N = 10000  # 10000 reklam gösterimi
# d = 10     # toplam 10 ilan var
# rewards = [0] * d   # Ri(N)
# click = [0] * d    # Ni(N)
# sum_reward = 0
# choes = []

# for n in range(0, N):
#     ad = 0  # seçilen ilan
#     max_ucb = 0
#     for i in range(0, d):
#         if (click[i] > 0):
#             average = rewards[i] / click[i]
#             delta = math.sqrt(3 / 2 * math.log(n)/ click[i])
#             ucb = average + delta
#         else:
#             ucb = N * 10
#         if max_ucb < ucb:
#             max_ucb = ucb
#             ad = i
#     choes.append(ad)
#     click[ad] = click[ad] + 1
#     reward = data.values[n, ad]
#     rewards[ad] = rewards[ad] + reward
#     sum_reward = sum_reward + reward

# print("Toplam Ödül:")
# print(sum_reward)
# plt.hist(choes)
# plt.show()

# Thompson Sampling

import math
import random
N = 10000  # 10000 reklam gösterimi
d = 10     # toplam 10 ilan var

sum_reward = 0
choes = []
ones=[0]* d
zeros=[0]* d

for n in range(0, N):
    ad = 0  # seçilen ilan
    max_th= 0
    for i in range(0, d):
        rand_beta=random.betavariate(ones[i]+1,zeros[i]+1)
        if rand_beta> max_th:
            max_th=rand_beta
            ad=i
    choes.append(ad)
    reward = data.values[n, ad]
    if reward==1:
        ones[ad]= ones[ad] + 1
    else:
        zeros[ad]= zeros[ad] + 1
    
    sum_reward = sum_reward + reward

print("Toplam Ödül:")
print(sum_reward)
plt.hist(choes)
plt.show()