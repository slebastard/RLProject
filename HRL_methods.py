# ToDo:
### Create tree structure for storing hierarchy of actions
### Create evaluate() method for finding the value of subtask m in state s

import numpy as np
import operator as op

from gridworld import *
from anytree import Node, PreOrderIter
from anytree.search import findall_by_attr
from random import random

class Option():
	def __init__(self, initSet, policy, quitMap, name='Unknown Option'):
		self.initSet = initSet
		self.policy = policy
		self.quitMap = quitMap
		self.log = 'active'
		self.name = name

	def escape(self, coords):
		r = random()
		if self.log == 'active' and r < self.quitMap[coords]:
			self.log = 'quit'


class MAXQ():
	def __init__(self, GridWorld, alpha0=0.25, expl0=0.9, n_iter=5000, actionsHierarchy=None, optionSet=None):
		self.GridWorld = GridWorld
		self.maxActionID = 0

		# A tree containing actions that are either primitives or options
		if actionsHierarchy is None:
			actionsHierarchy = Node('Root', type='option', option=self.rootOption())
			actionsHierarchy.n_prim = 0
			actionsHierarchy.n_opt = 0
			for action in self.GridWorld.action_names:
				Node(action, parent=actionsHierarchy, actionID = self.maxActionID, type='primitive')
				self.maxActionID += 1
				actionsHierarchy.n_prim += 1
		self.actions = actionsHierarchy		# check this is a shallow copy

		if optionSet is not None:
			for option in optionSet:
				self.addOption(option)

		self.V = np.random.rand(self.actions.n_prim + self.actions.n_opt,self.GridWorld.n_states) 		# value function: a matrix mapping (primActionID, state) to expected value
		self.C = np.random.rand(self.actions.n_opt + 1, self.GridWorld.n_states, self.actions.n_prim + self.actions.n_opt) 		# completion function: a matrix mapping (optionID, state, primActionID) to completion value
		self.alpha0 = alpha0		# initial learning rate. Harmonic learning function is used here
		self.expl0 = expl0			# initial exploration rate
		self.n_iter = n_iter
		self.unkOptCount = 0

		for iter in range(self.n_iter):
			initState = self.GridWorld.reset()
			self.time = 0
			self.run(self.actions, initState)
 
    
	def addOption(self, option):
		# NB: right now options can only be added at init. To be generalized
		optionName = option.name
		if option.name == 'Unknown Option':
			self.unkOptCount += 1
			optionName = 'Unknown Option ' + str(self.unkOptCount)

		Node(optionName, parent=self.actions, actionID = self.maxActionID, type='option', option=option)
		self.maxActionID += 1
		self.actions.n_opt += 1

	def evaluate(self, task, state):
		if task.type == 'primitive':
			return [self.V[task.actionID, state], task.actionID]
		else:
			value = {}
			for subtask in PreOrderIter(task):
				[value[subtask.actionID],_] = self.evaluate(subtask, state) + self.C[task.actionID, state, subtask.actionID
			greedyAction = max(value.iteritems(), key=op.itemgetter(1))[0]
			return [self.V[greedyAction, state], greedyAction]


	def learningRate(self, time):
		return alpha/float(time)


	def explorationRate(self):
		return expl0/float(self.time)


	def run(self, task, state):
		if task.type == 'primitive':
			[next_state, reward, absorb] = self.GridWorld.step(state, action)
			alpha = self.learningRate(self.time)
			self.V[task.actionID, state] = (1-alpha)*self.V[task.actionID, state] + alpha*reward
			self.time += 1
			return 1
		elif task.type == 'option':
			count = 0
			while task.option.log == 'active':
				action = task.option.policy(self.GridWorld.state2coord[state])
				N = self.run(action, state)
				[next_state, reward, absorb] = self.GridWorld.step(state, action)
				if absorb:
					task.option.log = 'quit'
				[greedyValue, greedyAction] = self.evaluate(task, next_state)
				alpha = self.rate(self.time)
				self.C[task.actionID, state, action] = (1-alpha)*self.C[task.actionID, state, action] + alpha*np.power(self.GridWorld.gamma, N)*greedyValue
				count = count + N
				state = next_state
				task.option.escape(self.GridWorld.state2coord[state])
			return count
		else:
			raise ValueError("Action type should be either 'primitive' or 'option'")


	def explorationPolicy(self, state):
		Q = self.V[:, state] + np.hstack(
			np.zeros((self.actions.n_prim,1))
			, self.C[0, state, :]
		)
		greedyAction = np.argmax(Q)
		failed = np.random.rand(1) < self.explorationRate()
		if failed or np.isnan(greedyAction):
			return np.random.choice(self.GridWorld.state_actions[state])
		else:
			return greedyAction


	def rootOption(self):
		initSet = np.ones((self.GridWorld.n_rows, self.GridWorld.n_cols))
		quitMap = np.zeros((self.GridWorld.n_rows, self.GridWorld.n_cols))
		return Option(initSet, self.explorationPolicy, quitMap, name='Root Option')
