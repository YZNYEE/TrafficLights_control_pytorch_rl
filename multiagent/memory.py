import math
import random

class memory:

	def __init__(self, maxsize, batch):

		self.maxsize = maxsize
		self.head = -1
		self.batch = batch
		self.memory = []
		self.reward = []

	def getMemoryLen(self):

		return len(self.memory)

	def append(self, experience):

		num = len(self.memory)
		reward = experience[3]

		self.reward.append(reward)

		if num < self.maxsize:

			self.head += 1
			if self.head == self.maxsize:
				self.head = 0
			self.memory.append(experience)

		else:

			self.head += 1
			if self.head == self.maxsize:
				self.head = 0
			self.memory[self.head] = experience

	def sample(self):

		num = len(self.memory)
		assert( num >= self.batch )

		experience = []
		for i in range(self.batch):
			index = random.uniform(0, num)
			index = int(index)
			experience.append(self.memory[index])

		return experience
