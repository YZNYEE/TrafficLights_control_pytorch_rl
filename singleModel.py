import sys
sys.path.append("..")

import torch
import torch.autograd as autograd
import sae.sae as sae
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.path.append("/usr/share/sumo/tools/")

import traci
import xml
import time
import numpy as np
import random
import time
from pandas import Series, DataFrame
import pandas as pd
import sklearn.preprocessing as prep

from wholemodel.logger import MyLogger
from multiagent.memory import memory
from copy import deepcopy
from multiagent.gru.model import gru

import single_information as ni
from trainSae import sae

def getNewDemands():
	os.system('python3 generateRoute.py')

def soft_update(source, target, t):
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

def hard_update(source, target):
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(source_param.data)

class QValueMap(nn.Module):
	def __init__(self, inputSize, hiddenSize, actionNums, useRnn = False):
		super(QValueMap, self).__init__()

		self.inputSize = inputSize
		self.hiddenSize = hiddenSize
		self.map1 = nn.Linear(inputSize, hiddenSize)
		self.hidden = nn.Linear(hiddenSize, hiddenSize)
		self.map2 = nn.Linear(hiddenSize, actionNums)

	def forward(self, inputs):
		out = self.map1(inputs)
		out = F.relu(out)
		out = self.hidden(out)
		out = F.relu(out)
		out = self.map2(out)
		return out

class singleModel():

	def __init__(self, loggerfile, modelfile = False):
			
		self.logger = MyLogger(loggerfile)
		self.trafficLights = ni.trafficLights
		self.linkEdges = ni.linkEdges
		if modelfile == False:
			self.model = self.createModel()
		else:
			self.logger.debug(' load model : ' + modelfile)
			self.model = torch.load(modelfile)

	def createModel(self):
		
		tlNum = len(self.trafficLights)
		dim = tlNum * 4
		dims = 0
		dims += dim
		if ni.useHalting:
			dims += dim
		if ni.useSpeed:
			dims += dim
		if ni.usePhase:
			dims += tlNum
		if ni.useStep:
			dims += 1
		self.dims = dims
		self.actions = 2
		if ni.useSae:
			self.dims = int(dims/2)
			return {'sae':sae(dims,1), 'qvalue':QValueMap(int(dims/2), dims, 2)}
		if ni.useRnn:
			return {'qvalue':gru(dims, dims*2)}
		return {'qvalue':QValueMap(dims, dims*2, 2)}

	def save(self, file, onlymodel = False):
		if onlymodel:
			torch.save(self.model, file)
			return 
		rewardlist = self.memory.reward
		step = self.memory.step

		if ni.useSae:
			torch.save(step, 'saestep.pt')
			torch.save(rewardlist, 'saereward.pt')
			torch.save(self.loss, 'saeloss.pt')
			return
		if ni.useRnn:
			torch.save(step, 'rnnstep.pt')
			torch.save(rewardlist, 'rnnreward.pt')
			torch.save(self.loss, 'rnnloss.pt')
			return

		torch.save(step, 'step.pt')
		torch.save(rewardlist, 'reward.pt')
		torch.save(self.loss, 'loss.pt')

	def train(self, totaldays, modellast = '', loadSae = True):

		self.loadPreProcessScaler(ni.preProcessDataFile)

		if ni.useSae:
			if not loadSae:
				saeModel = self.model['sae']
				saeModel.getScaler(ni.preProcessDataFile + '.npy')
				self.logger.debug("train sae model")
				saeModel.train(self.logger)
				self.save('saemodel.pt', onlymodel = True)
			self.model = torch.load('saemodel.pt')
			#print(self.model)

		self.logger.debug(" ----------------------------- train -----------------------------")

		sumoBinary = "sumo"
		sumoCmd = [sumoBinary, "-c", ni.sumocfg, "--ignore-route-errors",  "--time-to-teleport", "300"]

		targetmodel = deepcopy(self.model['qvalue'])

		optimizer = optim.Adam(self.model['qvalue'].parameters(), lr=0.001)
		gamma = 0.95
		minibatch = 128
		exchange = 500
		modellast += str(gamma)
		tua = 0.01
		loss_function = nn.MSELoss(size_average = False)
		totalloss = 0
		evaluationDays = 5
		self.trainCnt = 0
		self.createTrainRecord()
		self.loss = []

		self.memory = memory(100000, 128)
		for i in range(totaldays):

			self.createRecord()
			getNewDemands()
			traci.start(sumoCmd)
			step = 0

			self.logger.debug("train days : "+str(i))
			while step < ni.step:
				flag = self.simulation(traci, step, record = True)
				step += 1
				loss = self.trainModel(targetmodel, optimizer, loss_function, gamma = gamma, minibatch = minibatch)
				if loss > 0:
					totalloss += loss

				if self.trainCnt>0 and self.trainCnt%exchange == 0 and totalloss > 0:
					self.logger.debug(" soft copy parameters ")
					soft_update(self.model['qvalue'], targetmodel, tua)
					self.logger.debug(" totalloss :" + str(totalloss/exchange))
					self.loss.append(totalloss/exchange)
					totalloss = 0

			traci.close()

			if i > 3 and (i+1)%evaluationDays == 0:
				self.logger.debug(" version: " + str(int((i+1)/evaluationDays)) + " days: " + str(i))
				self.hiddenRecord()
				self.evaluation()
				self.loadRecord()

				self.logger.debug(" save model ")
				modelfile = 'model/version_' + str(int((i+1)/evaluationDays)) + '_' + modellast + '.pt'
				self.save(modelfile)

			self.clearTrainRecord()
			#self.getCsvTrafficLight('csv/test.csv')

	def clearTrainRecord(self):
		self.trainRecord['state'] = []
		self.trainRecord['dataLen'] = 0

	def hiddenRecord(self):
		self.hidden = {}
		self.hidden['trainRecord'] = self.trainRecord
		self.hidden['record'] = self.record

	def loadRecord(self):
		self.record = self.hidden['record']
		self.trainRecord = self.hidden['trainRecord']

	def getMeanWaitTime(self):

		step = self.record['step']
		maxid = self.record['maxid'][step]
		return sum(self.record['vehiclewaittime'][:,step])/(maxid+1)

	def evaluation(self):

		self.loadPreProcessScaler(ni.preProcessDataFile)

		sumoBinary = "sumo"
		sumoCmd = [sumoBinary, "-c", ni.sumocfg, "--ignore-route-errors",  "--time-to-teleport", "2000"]

		getNewDemands()

		self.createRecord()
		self.createTrainRecord()
		step = 0
		traci.start(sumoCmd)
		while step < 7200:
			self.simulation(traci, step, record = False, epsilon = 0)
			step += 1
		traci.close()
		meanwaittime = self.getMeanWaitTime()
		self.getCsvTrafficLight('csv/model.csv')

		self.createRecord()
		step = 0
		traci.start(sumoCmd)
		while step < 7200:
			self.addRecord(traci, step)
			traci.simulationStep()
			step += 1
		traci.close()
		fixmeanwaittime = self.getMeanWaitTime()
		self.getCsvTrafficLight('csv/fix.csv', action = False)

		self.logger.debug(" fix : " + str(fixmeanwaittime) + " model: " + str(meanwaittime))

	def loadPreProcessScaler(self, file):
		file += '.npy'
		data = np.load(file)
		print(np.shape(data))
		self.logger.debug('loading pre process scaler : '+ file)
		self.scaler = prep.StandardScaler().fit(data)

	def getPreProcessData(self, totaldays, file):

		sumoBinary = "sumo"
		sumoCmd = [sumoBinary, "-c", ni.sumocfg, "--ignore-route-errors", "--time-to-teleport", "2000"]

		totalfeature = []
		for i in range(totaldays):

			getNewDemands()

			self.logger.debug('collecting days: '+str(i) + ' data')
			traci.start(sumoCmd)
			step = 0
			
			self.clearRecord()
			while step < 7200:
				self.addRecord(traci, step, getwaittime = False)
				adjustFlag = self.getAdjustFlag()
				if adjustFlag:
					feature = self.getFeatureList(-1)
					#print(len(feature))
					totalfeature.append(feature)
				traci.simulationStep()
				step += 1
			traci.close()
			
		tf = np.array(totalfeature)
		np.save(file, tf)

	def getCsvTrafficLight(self, file, action = True):
		lent = len(self.trafficLights)
		if action:
			array = np.zeros((7300,lent*2+1))
		else:
			array = np.zeros((7300,lent*2))
		cols = []
		for tl in self.trafficLights:
			cols.append(tl + 'p')
			cols.append(tl + 'd')
		if action:
			cols.append('action')
		for i in range(7200):
			for j in range(lent):
				tl = self.trafficLights[j]
				array[i,j*2+0] = self.record['phase'][tl][i]
				array[i,j*2+1] = self.record['duration'][tl][i]
			array[i,-1] = -1
		if action:
			for i in range(self.trainRecord['dataLen']):
				d = self.trainRecord['state'][i-1]
				a = d[1]
				step = d[2]
				array[step,lent*2] = a
		dataFrame = DataFrame(array, columns = cols)
		dataFrame.to_csv(file)

	def trainModel(self, targetmodel, optimizer, loss_function, gamma = 0.1, minibatch = 128):

		ln = self.memory.getMemoryLen()
		if ln < 5000:
			return -1

		self.trainCnt += 1

		indexlist = self.memory.sample()
							
		if ni.useRnn:
			input = torch.FloatTensor(len(indexlist), ni.actionDuration-1 ,self.dims)
			inputtarget = torch.FloatTensor(len(indexlist), ni.actionDuration-1 ,self.dims)
		else:
			input = torch.FloatTensor(len(indexlist), self.dims)
			inputtarget = torch.FloatTensor(len(indexlist), self.dims)

		target = torch.FloatTensor(len(indexlist), self.actions)
				
		for j in range(len(indexlist)):

			data = indexlist[j]
			s = data[0]
			a = data[1]
			sn = data[2]
			r = data[3]

			input[j,:] = torch.squeeze(s.data)
			inputtarget[j,:] = torch.squeeze(sn.data)

		input = autograd.Variable(input)
		inputtarget = autograd.Variable(inputtarget)

		if ni.useRnn:
			hidden = torch.zeros(1, len(indexlist),self.dims*2)
			hidden = autograd.Variable(hidden)
			qs = self.model['qvalue'](input, hidden)
			qsn = targetmodel(inputtarget, hidden)
		else:
			qs = self.model['qvalue'](input)
			qsn = targetmodel(inputtarget)

		qsdata = torch.squeeze(qs.data)
		qsndata = torch.squeeze(qsn.data)

		target = qsdata.clone()

		for j in range(len(indexlist)):
		
			data = indexlist[j]

			a = data[1]
			r = data[3]
			actionset = data[4]
			target[j,a] = r + gamma*max(torch.index_select(qsndata[j,:],0,torch.LongTensor(actionset)))
			#target[j,a] = r + gamma*max(qsndata[j,:])

		target = autograd.Variable(target)

		loss = loss_function(qs, target)
		loss.backward()

		for param in self.model['qvalue'].parameters():
			param.grad.data.clamp_(-1, 1)

		optimizer.step()
		optimizer.zero_grad()

		return loss.data[0]

	def simulation(self, traci, step, record = False, epsilon = 0.05):
		self.addRecord(traci, step)
		adjustFlag = self.getAdjustFlag()
		flag = False
		if adjustFlag:
			self.adjustTrafficLights(traci, record = record, epsilon = epsilon)
			flag = True
		traci.simulationStep()
		return flag

	def adjustTrafficLights(self, traci, record, epsilon = 0.05):
		feature = self.getFeature()
		if ni.useRnn:
			feature.data = torch.unsqueeze(feature.data, 0)
		if ni.useSae:
			feature = self.model['sae'].forward(feature)
		if ni.useRnn:
			hidden = autograd.Variable(torch.zeros(1,1,self.dims*2))
			qvalue = self.model['qvalue'](feature, hidden)
		else:
			qvalue = self.model['qvalue'](feature)
		if ni.useRnn:
			qvalue.data = torch.squeeze(qvalue.data)

		action,actionset = self.getAction(qvalue.data, epsilon)
		self.executeAction(traci,action)
		if record:
			self.trainRecord['dataLen'] += 1
			option = ni.option
			if self.trainRecord['dataLen'] == 1:
				laststep = 0
			else:
				laststep = self.trainRecord['state'][-1][2]
			step = self.record['step']
			reward = self.getReward(infor = [laststep, step, 0.02], options = option)
			#print(reward)
			self.trainRecord['state'].append([feature, action, step, actionset, reward])

			if self.trainRecord['dataLen'] > 1:
				lastdata = self.trainRecord['state'][-2]
				lastfeature = lastdata[0]
				lastaction = lastdata[1]
				laststep = lastdata[2]
				lastreward = lastdata[4]
				experience = [lastfeature, lastaction, feature, reward, actionset, laststep]
				self.memory.append(experience)

	def getReward(self, infor, options):
		if options == 1:
			laststep = infor[0]
			epsilon = infor[1]
			step = self.record['step']
			assert( step > laststep )
			waittime = epsilon * sum(self.record['vehiclewaittime'][:,step] - self.record['vehiclewaittime'][:,laststep])/(step-laststep)
			return 0-waittime
		if options == 2:
			llstep = infor[0]
			laststep = infor[1]
			step = self.record['step']
			#assert( step > laststep )
			lastmeanwait = sum(self.record['vehiclewaittime'][:,llstep])/(self.record['maxid'][llstep]+1)
			meanwait = sum(self.record['vehiclewaittime'][:,laststep])/(self.record['maxid'][laststep]+1)
			return lastmeanwait - meanwait
		if options == 3: 
			llstep = infor[0]
			laststep = infor[1]
			epsilon = infor[2]
			assert( laststep > llstep )
			waittime = epsilon * sum(self.record['vehiclewaittime'][:,laststep] - self.record['vehiclewaittime'][:,llstep])/(laststep-llstep)
			return 0-waittime

	def executeAction(self,traci,action):
		tl = self.trafficLights[0]
		phase = self.record['phase'][tl][-1]
		if phase == 0 and action == 0:
			traci.trafficlights.setPhaseDuration(tl, ni.actionDuration)
		if phase == 4 and action == 0:
			traci.trafficlights.setPhaseDuration(tl, 0)
		if phase == 0 and action == 1:
			traci.trafficlights.setPhaseDuration(tl, 0)
		if phase == 4 and action == 1:
			traci.trafficlights.setPhaseDuration(tl, ni.actionDuration)

	def getAction(self, qvalue, epsilon):
		for tl in self.trafficLights:
			phase = self.record['phase'][tl][-1]
			duration = self.record['duration'][tl][-1]
			if duration < 25:
				if phase == 0:
					return 0, [0]
				else:
					return 1, [1]
			elif duration > 160:
				if phase == 0:
					return 1, [1]
				else:
					return 0, [0]
		p = random.uniform(0,1)
		if p > epsilon:
			if qvalue[0] > qvalue[1]:
				return 0, [0,1]
			else:
				return 1, [0,1]
		else:
			if qvalue[0] > qvalue[1]:
				return 1, [0,1]
			else:
				return 0, [0,1]

	def getFeatureList(self, index):
		featureList = []
		for tl in self.trafficLights:
			for edge in self.linkEdges[tl]['in']:
				featureList.append(self.record['vehicleNum'][edge][index])
		if ni.useHalting:
			for tl in self.trafficLights:
				for edge in self.linkEdges[tl]['in']:
					featureList.append(self.record['haltingNum'][edge][index])
		if ni.useSpeed:
			for tl in self.trafficLights:
				for edge in self.linkEdges[tl]['in']:
					featureList.append(self.record['speed'][edge][index])
		if ni.usePhase:
			for tl in self.trafficLights:
				featureList.append(self.record['phase'][tl][index])
		if ni.useStep:
			featureList.append(self.record['step']+index+1)
		return featureList

	def getFeature(self):

		if ni.useRnn:
			feature = []
			for i in range(ni.actionDuration-1):
				featureList = self.getFeatureList(-1-i)
				feature.append(featureList)
			feature = self.scaler.transform(feature)
		else:
			featureList = self.getFeatureList(-1)
			feature = self.scaler.transform([featureList])
		#print(feature)
		array = torch.FloatTensor(feature)
		array = torch.squeeze(array)
		array = autograd.Variable(array)
		return array

	def getAdjustFlag(self):
		for tl in self.trafficLights:
			phase = self.record['phase'][tl][-1]
			if phase == 0 or phase == 4:
				duration = self.record['duration'][tl][-1]
				if duration > 1 and (duration+1)%self.record['maxduration'] == 0:
					return True
		return False

	def addRecord(self, traci, step, getwaittime = True):

		self.record['step'] = step

		'''
			get states of traffic lights and edge
		'''
		for tl in self.trafficLights:
			phase = traci.trafficlights.getPhase(tl)
			self.record['phase'][tl].append(phase)
			'''
				get duration
			'''
			if step == 0:
				self.record['duration'][tl].append(0)
			else:
				lastPhase = self.record['phase'][tl][-2]
				if phase == lastPhase:
					duration = self.record['duration'][tl][-1] + 1
					self.record['duration'][tl].append(duration)
				else:
					self.record['duration'][tl].append(0)

			for edge in self.linkEdges[tl]['in']:
				#print(edge)
				haltingNum = traci.edge.getLastStepHaltingNumber(edge)
				vehicleNum = traci.edge.getLastStepVehicleNumber(edge)
				speed = traci.edge.getLastStepMeanSpeed(edge)
				self.record['haltingNum'][edge].append(haltingNum)
				self.record['vehicleNum'][edge].append(vehicleNum)
				self.record['speed'][edge].append(speed)

		'''
			get vehicle wait time
		'''
		if getwaittime:
			if step > 0:
				self.record['vehiclewaittime'][:,step] = self.record['vehiclewaittime'][:,step-1]
			for id in traci.vehicle.getIDList():
				wait = traci.vehicle.getAccumulatedWaitingTime(id)
				intid = int(id)
				#self.record['vehiclewaittime'][intid, step] = wait
				self.record['vehiclewaittime'][intid, step] = max(self.record['vehiclewaittime'][intid, step], wait)
				self.record['maxid'][step] = max(self.record['maxid'][step], intid)

	def createTrainRecord(self):

		self.trainRecord = {}
		self.trainRecord['state'] = []
		self.trainRecord['data'] = []
		self.trainRecord['dataLen'] = 0
		self.trainRecord['maxLen'] = 100000

	def createRecord(self):

		self.record = {}
		self.record['vehicleNum'] = {}
		self.record['haltingNum'] = {}
		self.record['speed'] = {}
		
		self.record['step'] = 0
		self.record['maxduration'] = ni.actionDuration
		self.record['vehiclewaittime'] = np.zeros((7000, 7300))
		self.record['maxid'] = np.zeros(7300)
		self.record['phase'] = {}
		self.record['duration'] = {}

		for tl in self.trafficLights:
			self.record['phase'][tl] = []
			self.record['duration'][tl] = []
			for edge in self.linkEdges[tl]['in']:
				self.record['vehicleNum'][edge] = []
				self.record['haltingNum'][edge] = []
				self.record['speed'][edge] = []

	def clearRecord(self):
		self.createRecord()


model = singleModel('log/test_rnn.log')

#print(model.actionMap)
#model.evaluation()

#getNewDemands(ni.generateTrips, ni.netXml, ni.tripsXml, ni.rouXml)

#model.getPreProcessData(50, ni.preProcessDataFile)
model.train(100, modellast = '_op3_rnn')

#model.save('test.pt')