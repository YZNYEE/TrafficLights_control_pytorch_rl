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
	for tl in ni.traffcLights:
		n = len(ni.traffcLights[tl]['in'])
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
	for tl in ni.traffcLights:
		phase[tl] = np.zeros(7300)
		duration[tl] = np.zeros(7300)
	return phase, duration

# self.record['edge']['speed'],self.record['edge']['number'] ,self.record['edge']['halting'] = util.getEdgeRecord()
def getEdgeRecord():

	speed = {}
	number = {}
	halting = {}
	for tl in ni.traffcLights:
		for edge in ni.linkEdges[tl]['in']:
			speed[edge] = np.zeros(7300)
			number[edge] = np.zeros(7300)
			halting[edge] = np.zeros(7300)

# self.record['vehicle']['edge'],self.record['vehcile']['waittime'] = util.getVehicleRecord()
def getVehicleRecord():

	edge = {}
	waittime = {}
	edge = np.zeros(10000, 7300) - 1
	waittime = np.zeros(10000, 7300)

def getStateRecord():

	state = {}
	for tl in ni.traffcLights:
		state[tl] = []
	return state

'''
util.addEdgeRecord(traci, self.record['edge'])
util.addTrafficLightRecord(traci, self.record['trafficlight'])
util.addVehicleRecord(traci, self.record['vehicle'])
'''
def addEdgeRecord(traci, record, step):

	for tl in ni.traffcLights:
		for edge in ni.linkEdges[tl]['in']:
			haltingNum = traci.edge.getLastStepHaltingNumber(edge)
			vehicleNum = traci.edge.getLastStepVehicleNumber(edge)
			speed = traci.edge.getLastStepMeanSpeed(edge)
			record['speed'][edge][step] = speed
			record['number'][edge][step] = vehicleNum
			record['halting'][edge][step] = haltingNum

def addTrafficLightRecord(traci, record, step):
	
	for tl in ni.traffcLights:
		phase = traci.trafficlights.getPhase(tl)
		record['phase'][step] = int(phase)
		if step = 0
			record['duration'][step] = 0
		else:
			lastphase = record['phase'][step - 1]
			if phase == lastphase:
				record['duration'][step] = record['duration'][step - 1] + 1
			else:
				record['duration'][step] = 0

def addVehicleRecord(traci, record, step):

	if step > 0:
		record['waittime'][:, step] = record['waittime'][:, step - 1]
	for id in traci.vehicle.getIDList():
		wait = traci.vehicle.getAccumulatedWaitingTime(id)
		laneid = traci.vehicle.getLaneID(id)
		edgeid = traci.lane.getEdgeID(laneid)
		record['edge'][int(id), step] = mapedge[edgeid]
		record['waittime'][int(id), step] = max(record['waittime'][int(id), step], wait)

def adjustFlag(tl, record):

	step = record['step']
	phase = record['trafficlight']['phase'][step]
	if not (phase == 0 or phase == 4):
		return False
	duration = record['trafficlight']['duration'][step]
	if duration > 1 and (duration+1) % ni.actionDuration:
		return True
	return False

def getFeature(tl, record):
	feature = []
	step = record['step']
	for tl in ni.traffcLights:
		for edge in ni.linkEdges[tl]['in']:
			feature.append(record['edge']['number'][step])
			if ni.useHalting:
				feature.append(record['edge']['halting'][step])
			if ni.useSpeed:
				feature.append(record['edge']['speed'][step])
		if ni.useSpeed
			feature.append(record['trafficlight']['phase'][step])
		if ni.useDuration:
			feature.append(record['trafficlight']['duration'][step])
	feature.append(step)

def getReward(tl, record):
	num = len(record['state'])
	if num == 0:
		laststep = 0
		step = record['step']
	else:
		laststep = record['state'][3]
		step = record['step']
	swaittime = sum(record['vehicle']['waittime'][:,laststep] - record['vehicle']['waittime'][:,step])
	return 0-0.02*swaittime/(step-laststep)

def getAction(tl, action, record, epsilon = 0.1):
	
	step = self.record['step']
	phase = self.record['trafficlight']['phase'][tl][step]
	duration = self.record['trafficlight']['duration'][tl][step]
	if duration < 20:
		if phase == 0:
			return 0
		if phase == 4:
			return 1
	if duration > 150:
		if phase == 0:
			return 1
		if phase == 4:
			return 0
	p = random
	if p < epsilon:
		if action[0] > action[1]:
			return 1
		else:
			return 0
	else:
		if action[0] > action[1]:
			return 0
		else:
			return 1