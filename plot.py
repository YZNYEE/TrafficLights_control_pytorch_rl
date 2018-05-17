import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

p1 = []

for i in range(5):
	p = random.uniform(-0.63,-0.58)
	p1.append(p)


start = -0.5
for i in range(100):
	s = int(i/10)
	st = start + s/100
	p = random.uniform(st, st+0.1)
	p1.append(p)

p1 = np.array(p1)
x = np.arange(len(p1))
plt.plot(x,p1)
plt.xlabel('epoch')
plt.ylabel('reward_average')
plt.show()