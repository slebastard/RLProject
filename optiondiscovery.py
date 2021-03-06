## Notes
# In this algorithm, each state can only become an option once. Therefore options will be indexed by their end state

import numpy as np
import operator as op
import gridrender as gui
import pdb

from gridworld import *
from RL_methods import *
from HRL_methods import *

from anytree import Node, PreOrderIter
from anytree.search import findall_by_attr
from anytree.render import RenderTree
from random import random
from tqdm import tqdm

class OptionDiscovery():
	def __init__(self, GridWorld, expl0=0.1, n_iter=5000):
		self.GridWorld = GridWorld
		self.MAXQ = MAXQ(GridWorld, expl0, n_iter, runOnCreate=False)
		self.subgoals = []	# will contain a list of options, with additional attribute rho_c
		self.importance = np.zeros(self.GridWorld.n_states)
		self.trajectories = [[]]
		self.options = []


	def run(self, debug=False):

		# Initialize enough trajectories to cross out erroneous concepts on the first runs
		n_init = 20
		for it in range(n_init):
			initState = self.GridWorld.reset()
			self.MAXQ.actions.option.log = 'active'
			self.MAXQ.it = it
			self.MAXQ.time = 1
			self.MAXQ.run(self.MAXQ.actions, initState, debug, history=True)
			self.trajectories.append([[s,x,y] for s,x,y in self.MAXQ.lastTraj if self.GridWorld.static_filter(s)])

		for it in tqdm(range(self.MAXQ.n_iter), desc="Discovering options on {} runs".format(self.MAXQ.n_iter)):
			initState = self.GridWorld.reset()
			self.MAXQ.actions.option.log = 'active'
			self.MAXQ.time = 1
			self.MAXQ.run(self.MAXQ.actions, initState, debug, history=True)
			self.trajectories.append([[s,x,y] for s,x,y in self.MAXQ.lastTraj if self.GridWorld.static_filter(s)])

			DD_map = self.get_DD_map()
			M_ddmap = np.max(DD_map)

			newConcepts = []
			seen = {}
			for s,_,_ in self.MAXQ.lastTraj:
				if s in seen:
					continue
				seen[s] = True

				i,j = self.GridWorld.state2coord[s]

				if DD_map[i,j] == M_ddmap:
					newConcepts.append(s)
			
			#We just add or update 1 concept:
			if len(newConcepts)>0:
				pdb.set_trace()
				self.makeOrUpdate(newConcepts[0])
			#self.makeOption(newConcept)
			#self.updateOption(option, option.conceptState)


	def get_DD_map(self):
    
	    DD_map = np.zeros((self.GridWorld.n_rows,self.GridWorld.n_cols))
	    count = {}
	    for s in range(self.GridWorld.n_states):
	        count[s] = 0
	    for traj in self.trajectories:
	        seen = {}
	        for x,_,_ in traj:

	            if not x in seen and self.GridWorld.static_filter(x):
	                i,j = self.GridWorld.state2coord[x]
	                DD_map[i,j] += 1
	                seen[x] = True
	    
	    return DD_map


	def makeOption(self, state):
		"""Computes a new option ending in state based on saved trajectories"""

		#1) Find all trajectories that crossed the concept state
		trajectories = get_trajectory_set(self.trajectories, state)

		#2) For each state on the union of trajectories, compute the value function as the mean of discount rate power time to goal
		value = np.zeros(self.GridWorld.n_states)
		n_visit = np.zeros(self.GridWorld.n_states)
		initSet = np.zeros((self.GridWorld.n_rows, self.GridWorld.n_cols))

		for traj in trajectories:
			time_to_goal = 0
			for step in reversed(traj):
				initSet[self.GridWorld.state2coord[step][0], self.GridWorld.state2coord[step][1]] = 1
				value[step] = float(np.power(self.GridWorld.gamma,time_to_goal))/(1+n_visit[step]) + float(value[step])/(1+n_visit[step])
				n_visit[step] += 1
				time_to_goal += 1

		#3) Compute greedy policy 
		policy = np.argmax(value)

		#4) Create action
		quitMap = np.zeros((self.GridWorld.n_rows, self.GridWorld.n_cols))
		option = Option(initSet, policy, quitMap, conceptState = state, name='Unknown Option')
		option.value = value
		option.n_visit = n_visit
		self.MAXQ.addOption(option)


	def updateOption(self, option, state):
		"""Updates option based on the last trajectory"""
		trajectory = get_trajectory_set(self.MAXQ.lastTraj, state)[0]
		time_to_goal = 0
		for step in reversed(trajectory):
			option.initSet[self.GridWorld.state2coord[step][0], self.GridWorld.state2coord[step][1]] = 1
			option.value[step] = float(np.power(self.GridWorld.gamma,time_to_goal))/(1+option.n_visit[step]) + float(option.value[step])/(1+option.n_visit[step])
			option.n_visit[step] += 1
			time_to_goal += 1


	def makeOrUpdate(self, conceptState):
		optionList = findall_by_attr(self.MAXQ.actions, value='option', name='type')
		if findall_by_attr(self.MAXQ.actions, value=conceptState, name='conceptState'):
			self.makeOption(conceptState)
			# ↑ This will compute I, beta and pi
			# create the option using option = Option() class constructor
			# then add the option to MAXQ instance by using self.MAXQ.addOption(option)

		else:
			option = findall_by_attr(self.MAXQ.actions, value=conceptState, name='conceptState')[0]
			self.updateOption(option, option.conceptState)
			# ↑ This will update I and pi using the new trajectory
			# Note that I may need to be broadened over time

def get_truncated_trajectories(trajectories, state):
	"""Finds all past trajectories containing state and truncates them until fist visit of state"""
	traj_toConcept = [[]]
	for traj in trajectories:
		if state in traj:
			traj_toConcept.append(traj[:traj.index(state)+1])