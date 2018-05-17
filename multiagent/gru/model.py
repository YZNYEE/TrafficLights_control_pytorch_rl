import torch
import torch.nn as nn
import torch.nn.functional as F

class gru(nn.Module):
	def __init__(self, inputsize, hiddensize, numlayers = 1, batchfirst = True):
		super(gru, self).__init__()
		self.inputSize = inputsize
		#print(inputsize)
		self.gru = nn.GRU(inputsize, hiddensize, numlayers, batch_first = batchfirst, dropout = 0)
		self.mid = nn.Linear(hiddensize, hiddensize)
		self.map = nn.Linear(hiddensize, 2)

	def forward(self, inputs, hidden):
		output, hn = self.gru(inputs, hidden)
		#print(output)
		#print(hn)
		#out = F.relu(hn)
		out = self.mid(hn)
		out = F.relu(out)
		out = self.map(out)
		return out