import random
import torch
import net_information as ni
import sys
sys.path.append("..")
import torch
import arrivalGen as ag


other = ni.others

entrance = []

exits = []

unreachable = {}

for node in other:

	entrance.append(ni.linkEdges[node]['out'][0])
	exits.append(ni.linkEdges[node]['in'][0])
	unreachable[ni.linkEdges[node]['out'][0]] = ni.linkEdges[node]['in'][0]

flow = [0.06, 0.06, 0.06, 0.06 ,0.06]

tripsList = []

for i in range(7200):
	for j in range(len(entrance)):
		enEdge = entrance[j]
		f = flow[j]
		p = random.uniform(0,1)
		if p > f:
			continue
		p = int(random.uniform(0,len(exits)))
		des = exits[p]
		while des != unreachable[enEdge]:
			p = int(random.uniform(0,len(exits)))
			des = exits[p]
		tripsList.append([i, enEdge, des])

ag.writeTripsXml(tripsList, ni.tripsXml)