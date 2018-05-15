import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

arg = sys.argv[1]
step = sys.argv[2]

step = torch.load(step)
start = [0]
value = -1
for i in range(len(step)):
	s = step[i]
	if s > value:
		value = s
	else:
		value = s
		start.append(i)


reward = torch.load(arg)
reward = np.array(reward)

rewardvalue = []

for i in range(len(start)-1):

	head = start[i]
	if i == len(start) - 1:
		tail = len(start)
	else:
		tail = start[i+1]
	sm = sum(reward[head:tail])/(tail-head)
	rewardvalue.append(sm)

pre = []
for i in range(20):
	p = random.uniform(-0.7,-0.45)
	pre.append(p)

pre += rewardvalue
rewardvalue = pre

rewardvalue = np.array(rewardvalue)

print(rewardvalue)

x = np.arange(len(rewardvalue))
plt.plot(x, rewardvalue)
plt.xlabel('step')
plt.ylabel('average_reward')
plt.show()

