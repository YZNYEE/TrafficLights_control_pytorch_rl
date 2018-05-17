import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from memory import memory

import utils as util
from copy import deepcopy

class QValueMap(nn.Module):
	def __init__(self, inputSize, hiddenSize, actionNums):
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

def soft_update(source, target, t):
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

def hard_update(source, target):
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(source_param.data)

class singleModel():

	def __init__(self, inputSize, actionNums = 2, memorySize = 20000, batch = 128):		

		self.model = QValueMap(inputSize, inputSize*2, actionNums)
		self.memory = memory(memorySize, batch)

		self.target = deepcopy(self.model)
		self.gamma = 0.95
		self.tua = 0.01

		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
		self.loss_function = nn.MSELoss(size_average = False)
		
		self.traincnt = 0
		self.totalloss = 0

		self.actions = actionNums
		self.dims = util.getDims()


	def load(self, file):

		self.model.load(file)

	def save(self, file):

		modelfile = file + '.pt'
		rewardfile = file + '.re'
		stepfile = file + '.st'
		rewardlist = self.memory.reward
		steplist = self.memory.step
		torch.save(self.model, file)
		torch.save(rewardlist, rewardfile)
		torch.save(steplist, stepfile)

	def append(self, experience):

		self.memory.append(experience)

	def sample(self):

		return self.memory.sample()

	def forward(self, input):

		return self.model(input)

	def train(self, tl, log):

		if self.memory.getMemoryLen() < 128:
			return

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
		qsn = self.target(inputtarget)

		qsdata = torch.squeeze(qs.data)
		qsndata = torch.squeeze(qsn.data)

		target = qsdata.clone()
		for j in range(len(indexlist)):
		
			data = indexlist[j]
			a = data[1]
			r = data[3]
			actionset = data[4]
			target[j,a] = r + self.gamma*max(torch.index_select(qsndata[j,:],0,torch.LongTensor(actionset)))
			#target[j,a] = r + gamma*max(qsndata[j,:])

		target = autograd.Variable(target)

		loss = self.loss_function(qs, target)
		loss.backward()

		self.totalloss += loss.data[0]

		for param in self.model.parameters():
			param.grad.data.clamp_(-1, 1)

		self.traincnt += 1

		self.optimizer.step()
		self.optimizer.zero_grad()

		if self.traincnt > 0 and self.traincnt % 500 == 0:
			soft_update(self.model, self.target, 0.01)
			log.debug('train ' + tl + 'totalloss : ' + str(self.totalloss/500))
			self.totalloss = 0
