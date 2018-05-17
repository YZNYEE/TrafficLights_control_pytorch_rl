import os
import sys

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.path.append("/usr/share/sumo/tools/")

import net_information as ni
import traci
import numpy as np
import random

global mapedge
mapedge = {}

cnt = 0
for edge in ni.edgelist:
	mapedge[edge] = cnt
	mapedge[cnt] = edge
	cnt += 1

def getDims():

	dim = 0
	for tl in ni.trafficLights:
		n = len(ni.linkEdges[tl]['in'])
		dim += n
		if ni.useHalting:
			dim += n
		if ni.useSpeed:
			dim += n
		if ni.usePhase:
			dim += 1
		if ni.useDuration:
			dim += 1
	if ni.useStep:
		dim += 1

	return dim

# self.record['trafficlight']['phase'],self.record['trafficlight']['duration'] = util.getTrafficLightRecord()
def getTrafficLightRecord():

	phase = {}
	duration = {}
	for tl in ni.trafficLights:
		phase[tl] = np.zeros(7300)
		duration[tl] = np.zeros(7300)
	return phase, duration

# self.record['edge']['speed'],self.record['edge']['number'] ,self.record['edge']['halting'] = util.getEdgeRecord()
def getEdgeRecord():

	speed = {}
	number = {}
	halting = {}
	for tl in ni.trafficLights:
		for edge in ni.linkEdges[tl]['in']:
			speed[edge] = np.zeros(7300)
			number[edge] = np.zeros(7300)
			halting[edge] = np.zeros(7300)
	return speed, number, halting

# self.record['vehicle']['edge'],self.record['vehcile']['waittime'] = util.getVehicleRecord()
def getVehicleRecord():

	edge = {}
	waittime = {}
	edge = np.zeros((3000, 7300)) - 1
	waittime = np.zeros((3000, 7300))
	return edge, waittime

def getStateRecord():

	state = {}
	for tl in ni.trafficLights:
		state[tl] = []
	return state

'''
util.addEdgeRecord(traci, self.record['edge'])
util.addTrafficLightRecord(traci, self.record['trafficlight'])
util.addVehicleRecord(traci, self.record['vehicle'])
'''
def addEdgeRecord(traci, record, step):

	for tl in ni.trafficLights:
		for edge in ni.linkEdges[tl]['in']:
			haltingNum = traci.edge.getLastStepHaltingNumber(edge)
			vehicleNum = traci.edge.getLastStepVehicleNumber(edge)
			speed = traci.edge.getLastStepMeanSpeed(edge)
			record['speed'][edge][step] = speed
			record['number'][edge][step] = vehicleNum
			record['halting'][edge][step] = haltingNum

def addTrafficLightRecord(traci, record, step):
	
	for tl in ni.trafficLights:
		phase = traci.trafficlights.getPhase(tl)
		record['phase'][tl][step] = int(phase)
		if step == 0:
			record['duration'][tl][step] = 0
		else:
			lastphase = record['phase'][tl][step - 1]
			if phase == lastphase:
				record['duration'][tl][step] = record['duration'][tl][step - 1] + 1
			else:
				record['duration'][tl][step] = 0

def addVehicleRecord(traci, record, step):

	if step > 0:
		record['waittime'][:, step] = record['waittime'][:, step - 1]
	for id in traci.vehicle.getIDList():
		wait = traci.vehicle.getAccumulatedWaitingTime(id)
		laneid = traci.vehicle.getLaneID(id)
		edgeid = traci.lane.getEdgeID(laneid)
		if edgeid in ni.edgelist:
			record['edge'][int(id), step] = mapedge[edgeid]
		record['waittime'][int(id), step] = max(record['waittime'][int(id), step], wait)

def adjustFlag(tl, record):

	step = record['step']
	phase = record['trafficlight']['phase'][tl][step]
	if not (phase == 0 or phase == 4):
		return False
	duration = record['trafficlight']['duration'][tl][step]
	if duration > 1 and (duration+1) % ni.actionDuration:
		return True
	return False

def getFeature(tl, record):
	feature = []
	step = record['step']
	for tl in ni.trafficLights:
		for edge in ni.linkEdges[tl]['in']:
			feature.append(record['edge']['number'][edge][step])
			if ni.useHalting:
				feature.append(record['edge']['halting'][edge][step])
			if ni.useSpeed:
				feature.append(record['edge']['speed'][edge][step])
		if ni.usePhase:
			feature.append(record['trafficlight']['phase'][tl][step])
		if ni.useDuration:
			feature.append(record['trafficlight']['duration'][tl][step])
	if ni.useStep:
		feature.append(step)
	return feature

def getReward(tl, record):
	num = len(record['state'][tl])
	if num == 0:
		laststep = 0
		step = record['step']
	else:
		laststep = record['state'][tl][-1][3]
		step = record['step']
	swaittime = sum(record['vehicle']['waittime'][:,step] - record['vehicle']['waittime'][:,laststep])
	return 0-0.02*swaittime/(step-laststep)

def getAction(tl, action, record, epsilon = 0.1):
	
	step = record['step']
	phase = record['trafficlight']['phase'][tl][step]
	duration = record['trafficlight']['duration'][tl][step]
	if duration < 20:
		if phase == 0:
			return 0, [0]
		if phase == 4:
			return 1, [1]
	if duration > 150:
		if phase == 0:
			return 1, [1]
		if phase == 4:
			return 0, [0]
	p = random.uniform(0,1)
	if p < epsilon:
		if action[0] > action[1]:
			return 1, [0,1]
		else:
			return 0, [0,1]
	else:
		if action[0] > action[1]:
			return 0, [0,1]
		else:
			return 1, [0,1]