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
	print(' SUMO_HOME has been set')
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

from logger import MyLogger
from momery import memory
from copy import deepcopy

import net3_information as ni

def getNewDemands(generate, netfile, tripfile, roufile):
	oscmd = 'python3 ' + generate
	print(oscmd)
	os.system(oscmd)
	cmd = 'duarouter -n ' + netfile + ' -t ' + tripfile + ' -o ' + roufile + ' --ignore-errors'
	os.system(cmd)

def soft_update(source, target, t):
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

def hard_update(source, target):
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(source_param.data)

class QValueMap(nn.Module):
	def __init__(self, inputSize, hiddenSize, actionNums):
		super(QValueMap, self).__init__()
		self.inputSize = inputSize
		self.hiddenSize = hiddenSize
		#print(inputSize, hiddenSize)
		#self.bn = nn.BatchNorm1d(inputSize)

		self.map1 = nn.Linear(inputSize, hiddenSize)
		self.hidden = nn.Linear(hiddenSize, hiddenSize)
		self.map2 = nn.Linear(hiddenSize, actionNums)

	def forward(self, inputs):
		#inputs = self.bn(inputs)
		out = self.map1(inputs)
		out = F.relu(out)
		out = self.hidden(out)
		out = F.relu(out)
		out = self.map2(out)
		return out

'''
	every traffic light has three action:
	action 0: lighten N-S green light
	action 1: lighten E-W green light
	action 2: keep current phase
'''

class model():

	def __init__(self, loggerfile, modelfile = False):
			
		self.logger = MyLogger(loggerfile)
		self.trafficLights = ni.trafficLights
		self.linkEdges = ni.linkEdges
		if modelfile == False:
			self.model = self.createModel()
		else:
			self.logger.debug(' load model : ' + modelfile)
			self.model = torch.load(modelfile)
		self.actionMap = self.createActionMap()

	def createActionMap(self):

		map = {'inttostr':{},'strtoint':{}}
		tlLen = len(self.trafficLights)
		maxid = 3 ** tlLen
		for i in range(maxid):
			strid = ""
			mx = i
			for j in range(tlLen):
				st = int(mx%3)
				strid += str(st)
				mx = int(mx/3)
			map['inttostr'][i] = strid
			map['strtoint'][strid] = i
		return map

	def createModel(self):
		
		tlNum = len(self.trafficLights)
		dim = 0
		for tl in self.trafficLights:
			dim += len(self.linkEdges[tl]['in'])
		dims = 0
		dims += dim
		if ni.useHalting:
			dims += dim
		if ni.useSpeed:
			dims += dim
		if ni.usePhase:
			dims += tlNum
		self.dims = dims
		self.actions = 3 ** tlNum
		return QValueMap(dims, dims*2, 3 ** tlNum)

	def save(self, file):
		rewardlist = self.memory.reward
		torch.save(rewardlist, 'reward/' + file + '_reward.pt')
		torch.save(self.trainRecord['loss'], 'loss/' + file + '_loss.pt')
		torch.save(self.model, 'model/' + file + '.pt')

	def train(self, totaldays, modellast = ''):

		self.loadPreProcessScaler(ni.preProcessDataFile)

		self.logger.debug(" ----------------------------- train -----------------------------")

		sumoBinary = "sumo"
		sumoCmd = [sumoBinary, "-c", ni.sumocfg, "--ignore-route-errors"]

		targetmodel = deepcopy(self.model)

		optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		gamma = 0.95
		minibatch = 256

		exchange = 1000
		exchangeDays = 5

		modellast += str(gamma)

		tua = 0.01
		checkpoint = 1000

		loss_function = nn.MSELoss(size_average = False)

		totalloss = 0

		evaluationDays = 5

		self.trainCnt = 0
		self.createTrainRecord()

		self.memory = memory(100000, 128)

		for i in range(totaldays):

			self.createRecord()

			getNewDemands(ni.generateTrips, ni.netXml, ni.tripsXml, ni.rouXml)
			traci.start(sumoCmd)
			step = 0

			self.logger.debug("train days : "+str(i))

			while step < 7200:
				flag = self.simulation(traci, step, record = True)
				step += 1
				if flag:
					loss = self.trainModel(targetmodel, optimizer, loss_function, gamma = gamma, minibatch = minibatch)
					if loss > 0:
						totalloss += loss
						self.trainRecord['loss'].append(loss)

				if self.trainCnt>0 and self.trainCnt%exchange == 0 and totalloss > 0:
					self.logger.debug(" --------- soft copy parameters --------- ")
					soft_update(self.model, targetmodel, tua)

				if self.trainCnt>0 and self.trainCnt%checkpoint == 0 and totalloss > 0:
					self.logger.debug(" totalloss :" + str(totalloss/checkpoint))
					totalloss = 0

			traci.close()

			if i > 0 and (i+1)%evaluationDays == 0:
				self.logger.debug(" version: " + str(int((i+1)/evaluationDays)) + " days: " + str(i))
				self.hiddenRecord()
				self.evaluation()
				self.loadRecord()

				self.logger.debug(" --------- save model -----------")
				modelfile = 'version_' + str(int((i+1)/evaluationDays)) + '_' + modellast
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
		sumoCmd = [sumoBinary, "-c", ni.sumocfg, "--ignore-route-errors"]

		getNewDemands(ni.generateTrips, ni.netXml, ni.tripsXml, ni.rouXml)

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
		data = np.load(file)
		#print(np.shape(data))
		self.logger.debug('loading pre process scaler : '+ file)
		self.scaler = prep.StandardScaler().fit(data)

	def getPreProcessData(self, totaldays, file):

		sumoBinary = "sumo"
		sumoCmd = [sumoBinary, "-c", ni.sumocfg, "--ignore-route-errors"]

		totalfeature = []
		for i in range(totaldays):

			getNewDemands(ni.generateTrips, ni.netXml, ni.tripsXml, ni.rouXml)

			self.logger.debug('collecting days: '+str(i) + ' data')

			traci.start(sumoCmd)
			step = 0
			self.clearRecord()
			while step < 7200:
				self.addRecord(traci, step, getwaittime = False)
				adjustFlag = self.getAdjustFlag()
				if adjustFlag:
					feature = self.getFeatureList()
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
		if ln < 3000:
			return -1

		self.trainCnt += 1

		indexlist = self.memory.sample()
							
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

		qs = self.model(input)
		qsn = targetmodel(inputtarget)

		qsdata = torch.squeeze(qs.data)
		qsndata = torch.squeeze(qsn.data)

		target = qsdata.clone()

		for j in range(len(indexlist)):
		
			data = indexlist[j]

			a = data[1]
			r = data[3]
			actionset = data[4]
			#target[j,a] = r + gamma*max(torch.index_select(qsndata[j,:],0,torch.LongTensor(actionset)))
			target[j,a] = r + gamma*max(qsndata[j,:])

		target = autograd.Variable(target)

		loss = loss_function(qs, target)
		loss.backward()

		for param in self.model.parameters():
			param.grad.data.clamp_(-1, 1)

		optimizer.step()
		optimizer.zero_grad()

		return loss.data[0]

	def sample(self, batch):
		lt = []
		num = self.trainRecord['dataLen'] - 1
		for i in range(batch):
			p = random.uniform(0,num)
			lt.append(int(p))
		return lt

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
		#print(feature)
		qvalue = self.model(feature)
		action,actionset = self.getAction(qvalue.data, epsilon)
		self.executeAction(traci,action)
		if record:
			self.trainRecord['dataLen'] += 1
			self.trainRecord['state'].append([feature, action, self.record['step'], actionset])
			if self.trainRecord['dataLen'] > 1:
				lastdata = self.trainRecord['state'][-2]
				lastfeature = lastdata[0]
				lastaction = lastdata[1]
				laststep = lastdata[2]
				options = 2
				#print(' ')
				#print('reward: ' + str(reward) + ' laststep: '+str(laststep) + ' step: '+ str(self.record['step']))
				if options == 1:
					reward = self.getReward(infor = [laststep, 0.2], options = 1)
					experience = [lastfeature, lastaction, feature, reward, actionset]
					self.memory.append(experience)
				if options == 2 and self.trainRecord['dataLen'] > 2:
					llstep = self.trainRecord['state'][-3][2]
					reward = self.getReward(infor = [llstep, laststep, 0.2], options = 2)
					experience = [lastfeature, lastaction, feature, reward, actionset]
					self.memory.append(experience)
				if (options == 3 or options == 4 )and self.trainRecord['dataLen'] > 2:
					llstep = self.trainRecord['state'][-3][2]
					reward = self.getReward(infor = [llstep, laststep, 0.2], options = options)
					assert( reward <= 0)
					experience = [lastfeature, lastaction, feature, reward, actionset]
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
			assert( laststep > llstep )
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
		if options == 4: 
			llstep = infor[0]
			laststep = infor[1]
			epsilon = infor[2]
			assert( laststep > llstep )
			waittime = sum(self.record['vehiclewaittime'][:,laststep] - self.record['vehiclewaittime'][:,llstep])/(laststep-llstep)/(self.record['currentNum'][llstep]+1)
			return 0-waittime

	def executeAction(self,traci,action):
		straction = self.actionMap['inttostr'][action]
		for i in range(len(straction)):
			inti = int(straction[i])
			#print(i)
			tl = self.trafficLights[i]
			phase = self.record['phase'][tl][-1]
			if inti == 0:
				if phase == 0:
					traci.trafficlights.setPhaseDuration(tl, ni.actionDuration)
				if phase == 4:
					traci.trafficlights.setPhaseDuration(tl, 0)
			if inti == 1:
				if phase == 0:
					traci.trafficlights.setPhaseDuration(tl, 0)
				if phase == 4:
					traci.trafficlights.setPhaseDuration(tl, ni.actionDuration)
			if inti == 2:
				pass

	def getAction(self, qvalue, epsilon):
		nophase = []
		for tl in self.trafficLights:
			phase = self.record['phase'][tl][-1]
			if phase != 0 and phase != 4:
				nophase.append(2)
			else:
				duration = self.record['duration'][tl][-1]
				if duration < 20:
					if phase == 0:
						nophase.append(0)
					else:
						nophase.append(1)
				elif duration > 190:
					if phase == 0:
						nophase.append(1)
					else:
						nophase.append(0)
				else:
					nophase.append(-1)
		actionset = self.getActionSet(nophase)

		#self.logger.DEBUG("")

		action = actionset[0]
		maxvalue = qvalue[action]

		p = random.uniform(0,1)
		if p > epsilon:
			for a in actionset:
				value = qvalue[a]
				if value > maxvalue:
					action = a
					maxvalue = value
			return action,actionset
		else:
			pa = random.uniform(0,len(actionset))
			pa = int(pa)
			return actionset[pa],actionset

	def getActionSet(self, nophase):
		strlist = [""]
		for i in nophase:
			lt = []
			p = i
			if p == -1:
				for j in strlist:
					lt.append(j+"0")
					lt.append(j+"1")
			else:
				for j in strlist:
					lt.append(j+str(p))
			strlist = lt
		actionset = []
		for st in strlist:
			id = self.actionMap['strtoint'][st]
			actionset.append(id)
		return actionset

	def getFeatureList(self):
		featureList = []
		for tl in self.trafficLights:
			for edge in self.linkEdges[tl]['in']:
				featureList.append(self.record['vehicleNum'][edge][-1])
		if ni.useHalting:
			for tl in self.trafficLights:
				for edge in self.linkEdges[tl]['in']:
					featureList.append(self.record['haltingNum'][edge][-1])
		if ni.useSpeed:
			for tl in self.trafficLights:
				for edge in self.linkEdges[tl]['in']:
					featureList.append(self.record['speed'][edge][-1])
		if ni.usePhase:
			for tl in self.trafficLights:
				featureList.append(self.record['phase'][tl][-1])
		return featureList

	def getFeature(self):

		featureList = self.getFeatureList()
		feature = self.scaler.transform([featureList])
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
			idlist = traci.vehicle.getIDList()
			self.record['currentNum'][step] = len(idlist)
			for id in idlist:
				wait = traci.vehicle.getAccumulatedWaitingTime(id)
				intid = int(id)
				self.record['vehiclewaittime'][intid, step] = max(self.record['vehiclewaittime'][intid, step], wait)
				self.record['maxid'][step] = max(self.record['maxid'][step], intid)

	def createTrainRecord(self):

		self.trainRecord = {}
		self.trainRecord['state'] = []
		self.trainRecord['data'] = []
		self.trainRecord['dataLen'] = 0
		self.trainRecord['maxLen'] = 100000
		self.trainRecord['loss'] = []

	def createRecord(self):

		self.record = {}
		self.record['vehicleNum'] = {}
		self.record['haltingNum'] = {}
		self.record['speed'] = {}
		
		self.record['step'] = 0
		self.record['maxduration'] = ni.actionDuration
		self.record['vehiclewaittime'] = np.zeros((5000, 7300))
		self.record['maxid'] = np.zeros(7300)
		self.record['currentNum'] = np.zeros(7300)
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


model = model('log/option2_net3.log')

print(model.actionMap)
#model.evaluation()

#getNewDemands(ni.generateTrips, ni.netXml, ni.tripsXml, ni.rouXml)

#model.getPreProcessData(25, ni.preProcessDataFile)
model.train(200, modellast = '_op2_net3')

#model.save('test.pt')