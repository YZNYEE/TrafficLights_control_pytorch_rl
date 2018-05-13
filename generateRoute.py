import numpy as np
import arrivalGen as ag
import scipy.stats as stats
import random

'''
effectiveLanesPairs = [['gneE5', 'gneE3'], ['gneE7', 'gneE1'], ['gneE0', 'gneE2'], ['gneE9', 'gneE6']]

x1 = np.arange(-3,3,0.15)
y1 = stats.norm.pdf(x1, 0, 2.5)

x2 = np.arange(-3,3,0.15)
y2 = stats.norm.pdf(x2, 0.2, 2.1)

x3 = np.arange(-3,3,0.15)
y3 = stats.norm.pdf(x3, 0, 2.7)

x4 = np.arange(-3,3,0.15)
y4 = stats.norm.pdf(x4, -0.2, 3)

FirstPro = y1[8:32]
SecondPro = y2[8:32]
ThirdPro = y3[8:32]
FourthPro = y4[8:32]

print(FirstPro)
print(SecondPro)
print(ThirdPro)
print(FourthPro)
'''

entrance = ('gneE5', 'gneE6', 'gneE1', 'gneE8')
destination = {'gneE5':{'l':'gneE7','s':'gneE0','r':'gneE2'}, 'gneE6':{'l':'gneE4','s':'gneE7','r':'gneE0'}, \
				'gneE1':{'l':'gneE2','s':'gneE4','r':'gneE7'}, 'gneE8':{'l':'gneE0','s':'gneE2','r':'gneE4'}}
flowspro = {'gneE5':[0.02,0.15,0.5,0.3,0.2,0.2,0.3,0.3,0.1,0.2,0.3,0.4], \
				'gneE6':[0.15,0.03,0.3,0.2,0.2,0.1,0.2,0.2,0.1,0.2,0.3,0.3], \
				'gneE1':[0.05,0.19,0.4,0.4,0.3,0.1,0.1,0.2,0.2,0.2,0.3,0.4], \
				'gneE8':[0.18,0.05,0.2,0.3,0.4,0.3,0.2,0.1,0.2,0.3,0.4,0.5]}
turnpro = {'gneE5':{'l':0.2,'s':0.7,'r':0.1},'gneE6':{'l':0.3,'s':0.55,'r':0.15},\
		'gneE1':{'l':0.15,'s':0.75,'r':0.1},'gneE8':{'l':0.1,'s':0.65,'r':0.25}}
direction = ('l', 's', 'r')


arrivalList = []

hour = 2

for i in range(2):

	delta = i*3600
	for j in range(3600):

		depart = delta+j
		for en in entrance:
			des = destination[en]
			flow = flowspro[en]
			turn = turnpro[en]

			pflow = random.uniform(0,1)
			if pflow < flow[i]:
				p = 0
				pt = random.uniform(0,1)
				for di in direction:
					p += turn[di]
					if p >= pt:
						arrivalList.append([depart, [en,des[di]]])
						break

ag.writeRouXml(arrivalList, 'my3direction.rou.xml')