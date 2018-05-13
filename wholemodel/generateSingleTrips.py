import sys
sys.path.append("..")

import torch
import arrivalGen as ag
import numpy as np
import random

entranceEdge = ["gneE8", "gneE5", "gneE6", "gneE1"]

flowsOneHour = {"gneE8":0.2, "gneE5":0.4, "gneE6":0.1, "gneE1":0.5}

destinationP = {"gneE8":{"gneE2":0.7,"gneE0":0.1,"gneE4":0.2},\
				 "gneE5":{"gneE0":0.7,"gneE7":0.1,"gneE2":0.2},\
				  "gneE6":{"gneE7":0.7,"gneE4":0.1,"gneE0":0.2},\
				   "gneE1":{"gneE4":0.7,"gneE2":0.1,"gneE7":0.2}}

tripsList = []

for i in range(7200):
	for enEdge in entranceEdge:
		flow = flowsOneHour[enEdge]
		p = random.uniform(0,1)
		if p >= flow:
			continue
		desp = destinationP[enEdge]
		a = random.uniform(0,1)
		p = 0
		for k in desp:
			p += desp[k]
			if a < p:
				tripsList.append([i, enEdge, k])
				break

ag.writeTripsXml(tripsList, "1net/single.trip.xml")