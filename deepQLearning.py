import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import reinforcement_utils.DeepQLearning as dq
from reinforcement_utils.DeepQLearning import averageMap

import os
import sys

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
#	sys.exit("please declare environment variable 'SUMO_HOME'")
	sys.path.append("/home/prik/sumo/sumo/tools/")

import traci
import xml
import time
import numpy as np
import random
import time
from pandas import Series, DataFrame
import pandas as pd

totalDays = 40
endStep = 7200
startStep = 4
minBatch = 32

gamma = 0

record = []
record_data = []

sumoBinary = "sumo"
sumoCmd = [sumoBinary, "-c", "single.sumo.cfg", "--summary-output","reinforcementResult/defaultOutput.xml", "--ignore-route-errors"]
edgesId = ['gneE0','gneE1','gneE2','gneE3','gneE5','gneE6','gneE7','gneE9']
lanesId = ['gneE0_0','gneE1_0','gneE2_0','gneE3_0','gneE5_0','gneE6_0','gneE7_0','gneE9_0']
trafficLightsId = 'gneJ1'

layerNum = 2
numLanes = 4
stateLanes = {'meanSpeed':0,'vehicleNum':0,'haltingNum':0,'waitingTime':0}
effectiveLanes = ['gneE7_0', 'gneE5_0', 'gneE0_0', 'gneE9_0']

saveModel = 'model/modelSampleGamma0.pt'

stateNum = numLanes * (len(stateLanes)-1) + 1
inputSize = stateNum

checkPoint = 1000

#action 0 means that keep this phase 5 sec. action 1 means that switch to next green phase
actionsNum = 2
vehicleAccumWaitTime = {}

trainingPro = 0.3

model = torch.load('model/modelSampleGamma0.pt')
#model = dq.simpleLinearMap(stateNum, 20, 2, actionsNum)

loss_function = nn.MSELoss()

#for p in model.parameters():
#	print([type(p.data), p.size()])

optimizer = optim.SGD(model.parameters(), lr=0.001)
sampleCount = []

bound = {'meanSpeed': {'max': 13.890000000000001, 'min': 0.0}, 'vehicle': {'max': 49.0, 'min': 0.0}, 'halting': {'max': 32.0, 'min': 0.0}, 'waitingTime': {'max': 699.0, 'min': 0.0}, 'occupy': {'max': 0.49469964664310956, 'min': 0.0}}

def initializeRouterFile(seed, fileTrip, fileRouter, endTime, period = 10):
	os.system('python3 generateRoute.py')

def getSumoCmd():
	return sumoCmd

def getStateVector(phase, bound = bound):

	vector = [phase%2]
	for lane in effectiveLanes:
		#meanSpeed = traci.lane.getLastStepMeanSpeed(lane)
		#meanSpeed = averageMapOther(bound['meanSpeed']['max'], bound['meanSpeed']['min'], meanSpeed)
		vehicle = traci.lane.getLastStepVehicleNumber(lane)
		vehicle = dq.averageMapOther(bound['vehicle']['max'], bound['vehicle']['min'], vehicle)
		halting = traci.lane.getLastStepHaltingNumber(lane)
		halting = dq.averageMapOther(bound['halting']['max'], bound['halting']['min'], halting)
		waitingTime = traci.lane.getWaitingTime(lane)
		waitingTime = dq.averageMapOther(bound['waitingTime']['max'], bound['waitingTime']['min'], waitingTime)
		vector += [vehicle, halting, waitingTime]
	#print(vector)
	return vector

	#occupy = traci.lane.getLastStepOccupancy(effectiveLanes[0]) 

def getAction(currStateVector, hour):

	actionVector = autograd.Variable(torch.FloatTensor(np.array(currStateVector + [hour])))
	QAction, ProAction = model(actionVector)
	a = random.uniform(0,1)
	p = 0
	for j in range(actionsNum):
		p += ProAction.data[j]
		#print(a,p)
		if p > a:
			return j,actionVector,QAction,ProAction

def getStepWaitingTime():
	twaitingTime = 0
	for lane in effectiveLanes:
		ls = traci.lane.getLastStepVehicleIDs(lane)
		for id in ls:
			wait = traci.vehicle.getAccumulatedWaitingTime(id)
			if vehicleAccumWaitTime.__contains__(id):
				twaitingTime += (wait - vehicleAccumWaitTime[id])
			else:
				twaitingTime += wait
			vehicleAccumWaitTime[id] = wait
	return twaitingTime

def getStepWaitingTimeMulti():
	twaitingTime = 0
	for lane in effectiveLanes:
		ls = traci.lane.getLastStepVehicleIDs(lane)
		for id in ls:
			wait = traci.vehicle.getAccumulatedWaitingTime(id)
			if vehicleAccumWaitTime.__contains__(id):
				twaitingTime += (wait * wait - vehicleAccumWaitTime[id] * vehicleAccumWaitTime[id]) 
			else:
				twaitingTime += wait * wait
			vehicleAccumWaitTime[id] = wait 
	return twaitingTime

def getStateVectorVariable(state, hour):
	actionVector = autograd.Variable(torch.FloatTensor(np.array(currStateVector + [hour])))
	return actionVector

def getNextStateStepWaitimgTime(timeDelta, waitingtime = []):

	accumWaitingTime = 0
	for i in range(timeDelta):
		traci.simulationStep()
		accumWaitingTime += getStepWaitingTime()
	meanAccumWaitingTime = accumWaitingTime/timeDelta
	return meanAccumWaitingTime/4, meanAccumWaitingTime

def getNextStateMeanSpeed(timeDelta, AmeanSpeed = []):

	meanSpeed = 0
	for i in range(timeDelta):
		traci.simulationStep()
		for lane in effectiveLanes:
			meanSpeed += traci.lane.getLastStepMeanSpeed(lane)
	meanSpeed = meanSpeed/(timeDelta*4)
	return meanSpeed/5,meanSpeed

def getNextState(timeDelta, lastWaitingTime):

	waitingTime = 0
	for i in range(timeDelta):
		traci.simulationStep()
		for lane in effectiveLanes:
			waitingTime += traci.lane.getWaitingTime(lane)
	waitingTime = waitingTime/(timeDelta*4)
	return 0-waitingTime/100,waitingTime

def sample(num):
	index = []

	if num > len(sampleCount):
		for i in range(num-len(sampleCount)):
			sampleCount.append(0)

	cnt = 0
	while cnt<minBatch:
		for i in range(num):
			e = 1/num
			p = random.uniform(0,1)
			if p < e:
				index.append(i)
				cnt += 1
			if cnt >= minBatch:
				break
	return index


losslist = []

for i in range(totalDays):

	step = 0

	#-------------------------- initialize router file -------------------------------
	seed = int(time.time())
	#initializeRouterFile(seed, 'single.trips.xml', 'single.rou.xml', 86400)

	#-------------------------- simulation -------------------------------------------
	sumoCmd = getSumoCmd()
	traci.start(sumoCmd)

	while step < startStep:
		traci.simulationStep()
		step += 1

	#-------------------------- collect data and train ------------------------------ 
	timeDelta = 1
	flag = False
	trainingData = []
	sumBatch = 0
	lastWaitingTime = 0
	action = 0
	cnt = 0

	totalLoss = []
	record = []

	while step < endStep:

		phase = traci.trafficlights.getPhase(trafficLightsId)
		#print(phase)

		hour = step/3600
		currStateVector = getStateVector(phase)

		#print(currStateVector)

		# only green phase take action
		if phase % 2 == 0 and step != startStep:
			
			#-------- get action ---------
			#print(currStateVector)
			
			action, actionVector, QAction, ProAction = getAction(currStateVector, hour)
			#print(action)

			if flag:
				# trainData [s, a, s', r , step]
				trainingData.append([[lastStateVector, lasthour], lastAction, [currStateVector, hour], reward, step])
				record.append([phase, reward, lastAction, lastWaitingTime, lastQ, QAction.data[0],QAction.data[1]])
				sumBatch += 1

			flag = True
			lastStateVector = currStateVector
			lasthour = hour
			lastAction = action
			lastQ = QAction.data[action]

			if action == 0:
				timeDelta = 4
				traci.trafficlights.setPhaseDuration(trafficLightsId, 4)
			elif action == 1:
				timeDelta = 8
				traci.trafficlights.setPhaseDuration(trafficLightsId, 0)
			'''
			elif action == 2:
				timeDelta = 24
				traci.trafficlights.setPhaseDuration(trafficLightsId, 0)
			elif action == 3:
				timeDelta = 34
				traci.trafficlights.setPhaseDuration(trafficLightsId, 0)
			elif action == 4:
				timeDelta = 44
				traci.trafficlights.setPhaseDuration(trafficLightsId, 0)
			'''

		#------------------- get next action ----------------------------------------
		step += timeDelta

		#waitingTime means meanSpeed
		#reward,waitingTime = getNextState(timeDelta, lastWaitingTime)
		reward, waitingtime = getNextStateStepWaitimgTime(timeDelta, lastWaitingTime)
		lastWaitingTime = waitingtime

		if step >= endStep:
			trainingData.append([[currStateVector, hour], 0, [], reward, step])

		#------------------------------update thelta ----------------------------
		if len(trainingData) >= minBatch:
			dataSetIndex = sample(len(trainingData))
			batchLoss = 0
			for index in dataSetIndex:
				data = trainingData[index]
				lastState = data[0]
				action = data[1]
				currState = data[2]
				reward = data[3]
				st = data[4]

				if st < endStep:
					ls = getStateVectorVariable(lastState[0], lastState[1])
					cs = getStateVectorVariable(currState[0], currState[1])

					ltarget, lpro = model(ls)
					ctarget, cpro = model(cs)

					label = reward + gamma*max(ctarget.data)
					labelTensor = []
					for j in range(actionsNum):
						if j == action:
							labelTensor.append(label)
						else:
							labelTensor.append(ltarget.data[j])
					labelTensor = autograd.Variable(torch.FloatTensor(labelTensor))
				else:
					ls = getStateVectorVariable(lastState[0], lastState[1])
					ltarget, lpro = model(ls)

					label = reward
					labelTensor = []
					for j in range(actionsNum):
						labelTensor.append(label)
					labelTensor = autograd.Variable(torch.FloatTensor(labelTensor))

				loss = loss_function(ltarget, labelTensor)
				loss.backward()
				batchLoss += loss.data[0]

			optimizer.step()
			optimizer.zero_grad()

			batchLoss /= minBatch
			totalLoss.append(batchLoss)

	record = DataFrame(record, columns = ['phase', 'reward','action', 'lastWaitingTime', 'lastQ', 'Q1', 'Q2'])
	record.to_csv('record/'+'accum'+str(i)+'.csv')

	bloss = np.array(totalLoss)
	print(bloss)
	np.save('loss/totalloss' + str(i) + '.npy', bloss)

	losslist.append(totalLoss)
	traci.close()
	torch.save(model, saveModel)

	vehicleAccumWaitTime = {}
	sampleCount = []
	#trainingData = []
	savetrain = np.array(trainingData)
	np.save('trainingData/trainData.npy', savetrain)

nploss = np.array(losslist)
np.save('loss/nploss.npy', nploss)
