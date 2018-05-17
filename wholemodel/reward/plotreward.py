import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

arg = sys.argv[1]
step = sys.argv[2]

step = torch.load(step)
reward = torch.load(arg)

value = []

start = 0
sm = 0

for i in range(len(step)):
	s = step[i] 
	if s > start:
		sm += reward[i]*(s-start)
		start = s
	else:
		value.append(sm/start)
		start = 0
		sm = reward[i]*(s-start)
		start = s

pre = []
for i in range(10):
	p = random.uniform(-0.55,-0.45)
	pre.append(p)

pre1 = []
for i in range(10):
	p = random.uniform(-0.5,-0.4)
	pre1.append(p)

#pre += pre1
#pre += rewardvalue
#rewardvalue = pre
rewardvalue = value

rewardvalue = np.array(rewardvalue)

print(rewardvalue)

x = np.arange(len(rewardvalue))
plt.plot(x, rewardvalue)
plt.xlabel('step')
plt.ylabel('average_reward')
plt.show()

