import sys
import random
import net_information as ni
from multimodel import singleModel
import utils as util
import os
from logger import MyLogger
import torch
import gc
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import sklearn.preprocessing as prep
import torch.autograd as autograd

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
	sys.path.append("/usr/share/sumo/tools/")

import traci

def getNewDemand():
	os.system('python3 generateTrips.py')
	os.system('duarouter -n net4.net.xml -t net4.trip.xml -o net4.rou.xml')

class agent():

	def __init__(self, logfile):
		self.log = MyLogger(logfile)
		self.createModel()

	'''
		create model for every traffic light
	'''
	def createModel(self):
		self.model = {}
		for tl in ni.trafficLights:
			dims = util.getDims()	
			self.model[tl] = singleModel(dims)

	def clearRecord(self):
		del self.record
		gc.collect()

	def addRecord(self, step, addEdgeRecord = True, addTrafficLightRecord = True, addVehicleRecord = True):
		self.record['step'] = step
		if addEdgeRecord:
			util.addEdgeRecord(traci, self.record['edge'], step)
		if addTrafficLightRecord:
			util.addTrafficLightRecord(traci, self.record['trafficlight'], step)
		if addVehicleRecord:
			util.addVehicleRecord(traci, self.record['vehicle'], step)

	'''
		self.record record details for the whole network on every step
	'''
	def createRecord(self):
		self.record = {}
		self.record['edge'] = {}
		self.record['trafficlight'] = {}
		self.record['vehicle'] = {}
		# get edge record
		self.record['edge']['speed'],self.record['edge']['number'] ,self.record['edge']['halting'] = util.getEdgeRecord()
		# get tl record
		self.record['trafficlight']['phase'],self.record['trafficlight']['duration'] = util.getTrafficLightRecord()
		# get vehicle record
		self.record['vehicle']['edge'],self.record['vehicle']['waittime'] = util.getVehicleRecord()
		self.record['step'] = -1
		self.record['state'] = util.getStateRecord()

	def getPreProcessData(self, totaldays):

		preData = {}
		for i in ni.trafficLights:
			preData[i] = []
		sumoCmd = ["sumo", "-c", ni.sumocfg, "--ignore-route-errors", "--time-to-teleport", "600"]
		for i in range(totaldays):
			self.log.debug('collecting preprocess data: ' + str(i) + ' days')
			getNewDemand()
			traci.start(sumoCmd)
			self.createRecord()
			step = 0
			while step < 7200:
				self.addRecord(step, addVehicleRecord = False)
				step += 1
				for tl in ni.trafficLights:
					flag = util.adjustFlag(tl, self.record)
					if flag:
						feature = util.getFeature(tl, self.record)
						preData[tl].append(feature)
				traci.simulationStep()
			traci.close()
			#self.clearRecord()
			if (i+1)%10 == 0:
				for tl in ni.trafficLights:
					a = np.array(preData[tl])
					a = np.array(a)
					file = 'preprocess/' +tl 
					np.save(file, a)
				#torch.save(preData, 'preprocess.pt')

	def train(self, totaldays):

		self.loadScaler()
		
		savedays = 4
		starttraindays = 3 
		sumoCmd = ["sumo", "-c", ni.sumocfg, "--ignore-route-errors", "--time-to-teleport", "600"]
		for i in range(totaldays):
			
			getNewDemand()
			self.createRecord()
			self.log.debug('train ' + str(i) + ' epoch')

			traci.start(sumoCmd)
			step = 0
			while step < 7200:
				self.simulation(traci, step)
				step += 1
				if i >= starttraindays:
					for tl in ni.trafficLights:
						self.model[tl].train(tl ,self.log)
			traci.close()
			if (i+1)%savedays == 0:
				savefile = 'model/'+'version'+str(int((i+1)/savedays)) + '_'
				self.log.debug('save model, reward and step in ' + savefile)
				for tl in ni.trafficLights:
					file = savefile + tl
					self.model[tl].save(file)

	def clearRecord(self):
		self.createRecord()

	def simulation(self, traci, step):
		self.addRecord(step)
		for tl in ni.trafficLights:
			self.adjustTrafficLight(traci, tl)
		traci.simulationStep()

	def adjustTrafficLight(self, traci, tl, record = True):
		
		flag = util.adjustFlag(tl, self.record)
		if flag:
			step = self.record['step']
			feature = util.getFeature(tl, self.record)
			feature = self.transform(tl, feature)
			reward = util.getReward(tl, self.record)
			qvalue = self.model[tl].forward(feature)
			action, actionset = util.getAction(tl, qvalue.data, self.record, epsilon = 0.1)
			self.record['state'][tl].append([feature, action, reward, step, actionset])
			self.addexperience(tl)
			self.executeAction(tl, action, step)

	def transform(self, tl, feature):
		feature = self.scaler[tl].transform([feature])
		array = torch.FloatTensor(feature)
		array = torch.squeeze(array)
		array = autograd.Variable(array)
		return array

	def executeAction(self, tl, action, step):
		phase = self.record['trafficlight']['phase'][tl][step]
		if phase == 0 and action == 0:
			traci.trafficlights.setPhaseDuration(tl, ni.actionDuration)
		if phase == 4 and action == 0:
			traci.trafficlights.setPhaseDuration(tl, 0)
		if phase == 0 and action == 1:
			traci.trafficlights.setPhaseDuration(tl, 0)
		if phase == 4 and action == 1:
			traci.trafficlights.setPhaseDuration(tl, ni.actionDuration)

	def addexperience(self, tl):
		if len(self.record['state'][tl]) > 1:
			current = self.record['state'][tl][-1]
			last = self.record['state'][tl][-2]
			laststep = last[3]
			self.model[tl].append([last[0], last[1], current[0], last[2], current[4], laststep])

	def loadScaler(self):
		self.scaler = {}
		for tl in ni.trafficLights:
			file = 'preprocess/'+tl+'.npy'
			data = np.load(file)
			print(np.shape(data))
			self.log.debug('loading pre process scaler : '+ file)
			self.scaler[tl] = prep.StandardScaler().fit(data)

	def evaluation(self):
		pass

agent = agent('test.lg')
#agent.getPreProcessData(40)
agent.train(200)	