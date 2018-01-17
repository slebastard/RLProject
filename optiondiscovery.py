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
	def __init__(self, GridWorld, alpha0=0.25, expl0=0.9, n_iter=5000, theta=0.8, lbda=0.95, name='Unknown Option'):
		# Note: for the algorithm to pickup new options, we must have theta <= 1/(1-lbda)
		self.GridWorld = GridWorld
		self.MAXQ = MAXQ(GridWorld, alpha0, expl0, n_iter, runOnCreate=False)
		self.subgoals = []	# will contain a list of options, with additional attribute rho_c
		self.importance = np.zeros(self.GridWorld.n_states)
		self.trajectories = [[]]
		self.options = []
		self.theta = theta
		self.lbda = lbda


	def run(self, coords, debug=False):
		for it in tqdm(range(self.n_iter), desc="Discovering options on {} runs".format(n_iter)):
			initState = self.GridWorld.reset()
			self.MAXQ.actions.option.log = 'active'
			self.MAXQ.time = 1
			self.MAXQ.run()
			self.run(self.MAXQ.actions, initState, debug, history=True)
			self.trajectories.append(filter(self.MAXQ.lastTraj))

			DD_map = self.get_DD_map()
			newConcept = self.GridWorld.coord2state[np.argmax(DD_map)]

			conceptList = findall_by_attr(self.MAXQ.actions, value='option', name='type')
			if self.importance[newConcept] > self.theta and not findall_by_attr(self.MAXQ.actions, value=conceptList, name='conceptState'):
				self.makeOption(newConcept)
				# ↑ This will compute I, beta and pi
				# create the option using option = Option() class constructor
				# then add the option to MAXQ instance by using self.MAXQ.addOption(option)

			elif self.importance[newConcept] > self.theta:
				option = findall_by_attr(self.MAXQ.actions, value=conceptList, name='conceptState')[0]
				self.updateOption(option)
				# ↑ This will update I and pi using the new trajectory
				# Note that I may need to be broadened over time

			self.importance *= self.lbda


	def get_DD_map(self):
    
	    DD_map = np.zeros((self.GridWorld.n_rows,self.GridWorld.n_cols))
	    count = {}
	    for s in range(self.GridWorld.n_states):
	        count[s] = 0
	    for traj in self.trajectories
	        seen = {}
	        for x,_,_ in traj:

	            if not x in seen and self.GridWorld.static_filter(x):
	                i,j = twoRooms.state2coord[x]
	                DD_map[i,j] += 1
	                seen[x] = True
	    
	    return DD_map


	def makeOption(self, state):
		"""Computes a new option ending in state based on saved trajectories"""


	def updateOption(self, option):
		"""Updates option based on the last trajectory"""
