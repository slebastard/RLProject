# ToDo:
### Create tree structure for storing hierarchy of actions
### Create evaluate() method for finding the value of subtask m in state s

import numpy as np
import operator as op

from gridworld import *
from anytree import Node, PreOrderIter
from random import random

class option():
	def __init__(self, initSet, policy, quitMap):
		self.initSet = initSet
		self.policy = policy
		self.quitMap = quitMap
		self.log = 'active'

	def escape(self, state):
		r = random()
		if self.log == 'active' and r > self.quitMap[state]:
			self.log = 'quit'


class MAXQ():

	def __init__(self, V, C, GridWorld, alpha=0.25, actionsHierarchy=None):
		self.GridWorld = GridWorld
		self.actions = actionsHierarchy 	# a tree containing actions that are either primitives or options
		self.V = V 		# value function: a matrix mapping (primActionID, state) to expected value
		self.C = C 		# completion function: a matrix mapping (optionID, state, primActionID) to completion value
		self.alpha = alpha		# initial learning rate. Harmonic learning function is used here

	def evaluate(self, task, state):
		if task.type == 'primitive':
			return [self.V[task, state], task]
		else:
			value = {}
			for subtask in PreOrderIter(task):
				[value{subtask},_] = self.evaluate(subtask) + self.C[task, state, subtask]
			greedyAction = max(value.iteritems(), key=op.itemgetter(1))[0]
			return [self.V[greedyAction, state], greedyAction]

	def step(self, task, state, alpha):
		if task.type == 'primitive':
			[next_state, reward, absorb] = self.GridWorld.step(state, action)
			self.V[task, state] = (1-alpha)*self.V[task, state] + alpha*reward
			return 1
		else:
			count = 0
			while task.log == 'active':
				action = task.policy(state) 	# if policy is an array, replace () by []
				N = self.step(action, state)
				[next_state, reward, absorb] = self.GridWorld.step(state, action)
				[greedyValue, greedyAction] = self.evaluate(task, next_state)
				self.C[task, state, action] = (1-alpha)*self.C[task, state, action] + alpha*np.power(self.GridWorld.gamma, N)*greedyValue
				count = count + N
				state = next_state
			return count