# ToDo:
### Create tree structure for storing hierarchy of actions
### Create evaluate() method for finding the value of subtask m in state s

import numpy as np
import operator as op
import pdb

from gridworld import *
from anytree import Node, PreOrderIter
from anytree.search import findall_by_attr
from anytree.render import RenderTree
from random import random
from tqdm import tqdm

class Option():
	def __init__(self, initSet, policy, quitMap, name='Unknown Option'):
		self.initSet = initSet
		self.policy = policy
		self.quitMap = quitMap
		self.log = 'active'
		self.name = name

	def escape(self, coords, debug=False):
		r = random()
		if self.log == 'active' and r < self.quitMap[coords[0],coords[1]]:
			if debug:
				print('Escape at coords ({0},{1}) with {2} succeeds'.format(coords[0], coords[1], r))
			self.log = 'quit'
		elif debug:
			print('Escape at coords ({0},{1}) with {2} fails'.format(coords[0], coords[1], r))

class MAXQ():
	def __init__(self, GridWorld, alpha0=0.25, expl0=0.9, n_iter=5000, actionsHierarchy=None, optionSet=None, debug=False):
		self.GridWorld = GridWorld
		self.maxActionID = 0

		# A tree containing actions that are either primitives or options
		if actionsHierarchy is None:
			actionsHierarchy = Node('Root', actionID = -1, type='option', option=self.rootOption())
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
		self.C = np.random.rand(self.actions.n_prim + self.actions.n_opt + 1, self.GridWorld.n_states, self.actions.n_prim + self.actions.n_opt) 		# completion function: a matrix mapping (optionID, state, primActionID) to completion value
		self.alpha0 = alpha0		# initial learning rate. Harmonic learning function is used here
		self.expl0 = expl0			# initial exploration rate
		self.n_iter = n_iter
		self.unkOptCount = 0

		for pre, _, node in RenderTree(self.actions):
			print("%s%s" % (pre, node.name))

		for it in tqdm(range(self.n_iter), desc="Training MAXQ on {} runs".format(n_iter)):
			initState = self.GridWorld.reset()
			self.time = 1
			self.run(self.actions, initState, debug)
 

	def learningRate(self):
		return self.alpha0/float(self.time)


	def explorationRate(self):
		return self.expl0/float(self.time)   


	def addOption(self, option):
		# NB: right now options can only be added at init. To be generalized
		optionName = option.name
		if option.name == 'Unknown Option':
			self.unkOptCount += 1
			optionName = 'Unknown Option ' + str(self.unkOptCount)

		primActions = findall_by_attr(self.actions, value='primitive', name='type', maxlevel=2)

		option = Node(optionName, parent=self.actions, actionID = self.maxActionID, type='option', option=option)
		for action in primActions:
			Node(action.name, parent = option, actionID = action.actionID, type='primitive')
		self.maxActionID += 1
		self.actions.n_opt += 1


	def evaluate(self, task, state):
		# print("Evaluate from {}".format(task.name))
		if task.type == 'primitive':
			return [self.V[task.actionID, state], task.actionID]
		else:
			value = {}
			for subtask in PreOrderIter(task):
				if subtask == task:
					continue
				value[subtask.actionID] = self.evaluate(subtask, state)[0] + self.C[task.actionID + 1, state, subtask.actionID]
			greedyAction = max(zip(value.values(), value.keys()))[1]
			return [self.V[greedyAction, state], greedyAction]


	def run(self, task, state, debug=False):
		if debug:
			print("Run with {} at coords [{}, {}]".format(task.name, self.GridWorld.state2coord[state][0], self.GridWorld.state2coord[state][1]))
		if task.type == 'primitive':
			if debug:
				print('Primitive!')
			[next_state, reward, absorb] = self.GridWorld.step(state, task.actionID)
			alpha = self.learningRate()
			self.V[task.actionID, state] = (1-alpha)*self.V[task.actionID, state] + alpha*reward
			self.time += 1
			return [1, next_state, absorb]
		elif task.type == 'option':
			if debug:
				print('Option!')
			count = 0
			absorb = False
			while task.option.log == 'active':
				if debug:
					print("Time {}".format(self.time))
				subtaskID = task.option.policy(self.GridWorld.state2coord[state])
				if debug:
					print("Substask chosen {}".format(subtaskID))
				subtask = findall_by_attr(self.actions, value=subtaskID, name='actionID')[0]
				[N, next_state, absorb] = self.run(subtask, state, debug)
				if absorb:
					task.option.log = 'quit'
				[greedyValue, greedyAction] = self.evaluate(task, next_state)
				alpha = self.learningRate()
				self.C[task.actionID + 1, state, subtask.actionID] = (1-alpha)*self.C[task.actionID + 1, state, subtask.actionID] + alpha*np.power(self.GridWorld.gamma, N)*greedyValue
				count = count + N
				state = next_state
				task.option.escape(self.GridWorld.state2coord[state])
			return [count, state, absorb]
		else:
			raise ValueError("Action type should be either 'primitive' or 'option'")


	def explorationPolicy(self, coords):
		state = self.GridWorld.coord2state[coords[0],coords[1]]
		Q = self.V[:, state] + self.C[0, state, :].reshape((1,self.actions.n_prim + self.actions.n_opt))
		# Chech whether or not the options can be called from current state
		optList = findall_by_attr(self.actions, value='option', name='type')

		for opt in optList:
			if not opt.option.initSet[coords[0], coords[1]]:
				Q[0,opt.actionID] = 0

		adm_actionSet = np.hstack((np.zeros((1,self.actions.n_prim)),np.ones((1,self.actions.n_opt))))
		for actionID in self.GridWorld.state_actions[state]:
			adm_actionSet[0,actionID] = 1

		Q = Q*adm_actionSet
		greedyActionID = np.argmax(Q)
		actionID = greedyActionID
		failed = np.random.rand(1) < self.explorationRate()
		if failed or np.isnan(greedyActionID):
			actionID = np.random.choice(self.GridWorld.state_actions[state])

		return actionID


	def rootOption(self):
		initSet = np.ones((self.GridWorld.n_rows, self.GridWorld.n_cols))
		quitMap = np.zeros((self.GridWorld.n_rows, self.GridWorld.n_cols))
		return Option(initSet, self.explorationPolicy, quitMap, name='Root Option')


	def computeGreedyPolicy(self):
		Q = self.V + np.swapaxes(self.C[0, :, :].reshape((1,self.actions.n_prim + self.actions.n_opt)),0,1)
		self.policy = np.argmax(Q, axis=0)
