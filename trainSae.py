import torch
import torch.autograd as autograd
from sae.sae import sparseAutoEncode
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys

if 'SUMO_HOME' in os.environ:
	tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
	sys.path.append(tools)
else:
#	sys.exit("please declare environment variable 'SUMO_HOME'")
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

trainDataFile = 'trainingData/saetrain.npy'
testDataFile = 'trainingData/saetest.npy'

class sae():

	def __init__(self, inputsize, layers):

		hiddensize = int(inputsize/2)
		self.saeModels = []
		for i in range(layers):
			self.saeModels.append(sparseAutoEncode(inputsize, hiddensize))
			inputsize = hiddensize
			hiddensize = int(hiddensize/2)


	def getScaler(self, trainDataFile):

		self.trainData = np.load(trainDataFile)
		self.scaler = prep.StandardScaler().fit(self.trainData)

		self.trainData = self.scaler.transform(self.trainData)

	def train(self, logger):

		optimizer = optim.Adam(self.saeModels[0].parameters(), lr=0.001)

		minbatch = 64
		maxepoch = 2000
		epoch = 0

		loss_function = nn.MSELoss()
		belta = 0.9

		checkpoint = 2000
		lasttestloss = 100000

		for epoch in range(maxepoch):

			dims = np.shape(self.trainData)
			head = 0
		
			totalloss = 0
			lastloss = 0
		
			while head < dims[0]:
				data = self.trainData[head:head+minbatch,:]
				input = autograd.Variable(torch.FloatTensor(data))

				hidden, out = self.saeModels[0](input)

				target = autograd.Variable(input.data)
				loss1 = loss_function(out, target)

				p = 0.05
				pmean = torch.sum(hidden, dim = 0)/hidden.size(0)
				onepmean = 1 - pmean
				item1 = p*torch.log(p/pmean)
				item2 = (1-p)*torch.log((1-p)/onepmean)
				item = item1 + item2
				loss2 = torch.sum(item)*belta

				sumloss = loss1 + loss2
				sumloss.backward()

				optimizer.step()
				optimizer.zero_grad()

				head += minbatch
				cnt += 1

				totalloss += sumloss.data[0]
		
				if cnt%checkpoint == 0 and cnt != 0:

					logger.debug('epoch: ' + str(epoch) + ' train totalloss: ' + str(totalloss/checkpoint))
					totalloss = 0

					'''
					#------------ test totalloss -----------------------
					dimstest = np.shape(self.testData)
					inputtest = autograd.Variable(torch.FloatTensor(testData))

					hiddentest, outtest = md(inputtest)

					target = autograd.Variable(inputtest.data)
					loss1 = loss_function(outtest, target)
					p = 0.05
					pmean = torch.sum(hiddentest, dim = 0)/hiddentest.size(0)
					onepmean = 1 - pmean
					item1 = p*torch.log(p/pmean)
					item2 = (1-p)*torch.log((1-p)/onepmean)
					item = item1 + item2
					loss2 = torch.sum(item)*belta
					sumloss = loss1 + loss2
			
					if sumloss.data[0] < lasttestloss:
						lasttestloss = sumloss.data[0]

					logger.debug('test totalloss: ' + str(sumloss.data[0]))
					'''



