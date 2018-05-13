
import random
import net_information as ni
import traci
from multimodel import singleModel
import utils as util

class agent():

	def __init__(self):
		pass

	'''
		create model for every traffic light
	'''
	def createModel(self):
		self.model = {}
		for tl in ni.trafficLights:
			dims = util.getDims(tl)	
			self.model[tl] = singleModel(dims)

	def addRecord(self, step, addEdgeRecord = True, addTrafficLightRecord = True, addVehicleRecord = True):
		self.record['step'] = step
		if addEdgeRecord:
			util.addEdgeRecord(traci, self.record['edge'])
		if addTrafficLightRecord:
			util.addTrafficLightRecord(traci, self.record['trafficlight'])
		if addVehicleRecord:
			util.addVehicleRecord(traci, self.record['vehicle'])

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
		self.record['vehicle']['edge'],self.record['vehcile']['waittime'] = util.getVehicleRecord()
		self.record['step'] = -1
		self.record['state'] = util.getStateRecord()

	def train(self, totaldays):

		starttraindays = 3 
		for i in range(totaldays):
			sumoCmd = ["sumo", "-c", ni.sumocfg, "--ignore-route-errors", "--time-to-teleport", "2000"]
			traci.start(sumoCmd)
			step = 0
			while step < 7200:
				self.simulation(traci, step)
				step += 1
				if i >= starttraindays:
					for tl in ni.trafficLights:
						self.model[tl].train()
			traci.close()

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
			reward = util.getReward(tl, self.record)
			qvalue = self.model[tl].forward(feature)
			action = util.getAction(tl, qvalue, self.record, epsilon = 0.1)
			self.record['state'][tl].append(feature, action, reward, step)
			self.addexperience(tl)
			self.executeAction(tl, action)

	def executeAction(self, tl, action):
		phase = self.record['trafficlight'][tl]
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
			current = self.record['state'][-1]
			last = self.record['state'][-2]
			self.model[tl].append([last[0], last[1], current[0], last[2]])

	def evaluation(self):
		pass