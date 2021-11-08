import gym
from gym import error, spaces, utils
from gym.utils import seeding
from fractions import gcd
from itertools import permutations
import numpy as np
from scipy.optimize import minimize
from sympy import Matrix as sp_matrix
import random
import cPickle as pickle
import uuid
import os
import datetime
from inspect import getsourcefile
import errno
from scipy import linalg
import random
from scipy import random as scipyrandom
import cPickle as pickle
import math
import numpy  as np
#import mpmath as mpm
#from mpmath import *

##set the numerical accuracy in mpm:
#mp.dps = 200

class Wishart(gym.Env):

	########
	# RL related methods
	
	def __init__(self):
		#self.nmod = 10
		#self.sigma = 1e-3
		#self.eps = 1e-3
		#self.barecc = mpf(-1)
		
		self.action_space = None
		self.observation_space = None
		#self.metric_index = None
		self.solved_factor=mpf(1e10)
		#self._outputFilePath = os.path.split(os.path.abspath(getsourcefile(lambda:0)))[0] + "/../output/Pickle_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "/outputOvercountsRun_"+ str(uuid.uuid4()) + ".pickle"
		self.max_var= 1e5
		self.min_var = 0
		self.max_mean = 10
		self.min_mean = -10
		#self.init_cc_printed = False
		self.global_t = 0
		self.trackscore = 0

	def second_init(self): # after nmod, sigma, eps are set, can run this
		###make sure eps is set at desired accuracy
		self.nmoments = len(self.moments)
		self.means = [[0 for i in range(self.nmod)] for j in range(self.nmod)]
		self.vars = [[1/self.nmod^(1/2) for i in range(self.nmod)] for j in range(self.nmod)]
		self.stepsize = (1e-2)*1/self.nmod^(1/2)
		self.sigma = mpf(self.sigma)
		self.state = self.vars[0]
		for i in range(1,self.nmod):
			self.state = self.state + self.vars[i]
		self.action_space = spaces.Discrete(2*self.nmod*self.nmod)
		self.observation_space = spaces.Box(low=int(self.min_var), high=int(self.max_var), shape=(self.nmod*self.nmod, 1))
		

	def step(self,action):
		done = False
		idx, sign = (action-action%2)/2, mpf((-1)**(action%2))
		self.state[idx] += sign*self.stepsize
		ii = floor(idx/(self.nmod*self.nmod))
		jj = idx %(self.nmod*self.nmod)
		self.vars[ii][jj] += sign*self.stepsize
		means, moment = self.sample()
		mean = np.mean(means)
		moment = np.mean(moments)

		if abs((mean - self.mean)/self.mean) < self.meantol and abs((moment - self.sigma)/self.sigma) < self.sigmatol:
			done = True
			#print 'huzzah!', cc, self.state
			my_reward = float(self.solved_factor)
		else:
			my_reward = self.reward(mean,moment)
			self.trackscore += my_reward

		return np.array([float(k) for k in self.state]), my_reward, done, {}
		
	def reset(self):
		self.state = self.vars[0]
		for i in range(1,self.nmod):
			self.state = self.state + self.vars[i]
		#self.ngvec = np.dot(np.array(self.origin),self.metric) # mpf due to self.origin is mpf above
		#self.cc = self.barecc + np.dot(np.dot(self.metric,self.origin),self.origin)
		#if self.init_cc_printed == True:
		#	print 'initial cc:', self.cc
		
		# create path to file if necessary (it shouldn't be, the path should've been created by the training program
		if not os.path.exists(os.path.dirname(self._outputFilePath)):
			try:
				os.makedirs(os.path.dirname(self._outputFilePath))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		   
		# update the file
		hnd = open(self._outputFilePath, 'a+')
		hnd.write("rws " + str((self.process_idx,self.global_t,self.trackscore,self.state))+"\n")
		hnd.close()
		self.trackscore = 0
		return np.array(self.state)
	
	def dist(self,mean,moment):
		return abs(mean-self.mean)*abs(moment-self.sigma)

	def reward(self,mean,moment):
		d = self.dist(mean,moment)
		#do = self.dist(self.occ)
		#print np.float(1/d), np.float(1/d**self.pow) 
		return np.float(1/d)
		#return 0
		

	def random_wishart(self): # pos def metric
		A = [[np.random.normal(loc = 0,scale = self.vars[pp][mm]) for mm in range(self.nmod)] for pp in range(self.nmod)]
		#A = [[np.random.normal(size=(self.nmod,self.nmod), scale = self.sigma)]]
		#A = np.array([[self.sigma*mpm.sqrt(mpf(2))*mpm.erfinv(mpf(2)*mpm.rand()-mpf(1)) for i in range(self.nmod)] for j in range(self.nmod)])
		#mpm addition
		#A = np.array([[mpm.npdf(0,self.sigma) for i in range(self.nmod)] for j in range(self.nmod)])
		#print "metric test\n", A
		return np.dot(A,A.transpose())

	def sample(self):
		means = []
		moments = []
		for num in range(self.numdraws):
			evals  = np.linalg.eig(random_wishart())[0]
			means.append(np.mean(evals))
			moments.append(np.std(evals))
		return means, moments


	def init_output(self):
		if not os.path.exists(os.path.dirname(self._outputFilePath)):
			try:
				os.makedirs(os.path.dirname(self._outputFilePath))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

		# write header data
		hnd = open(self._outputFilePath, 'a+')
		hnd.write("head: " + str((self.eps,self.nmod,self.sigma,self.metric_index))+"\n")
		hnd.close()

	# def output_min_pos_cc(self):
	# 	# create path to file if necessary (it shouldn't be, the path should've been created by the training program
	# 	if not os.path.exists(os.path.dirname(self._outputFilePath)):
	# 		try:
	# 			os.makedirs(os.path.dirname(self._outputFilePath))
	# 		except OSError as exc: # Guard against race condition
	# 			if exc.errno != errno.EEXIST:
	# 				raise
		   
	# 	# update the file
	# 	hnd = open(self._outputFilePath, 'a+')
	# 	hnd.write("p " + str((self.process_idx,self.global_t,self.min_pos_cc,self.state))+"\n")
	# 	hnd.close()

	# def output_max_neg_cc(self):
	# 	# create path to file if necessary (it shouldn't be, the path should've been created by the training program
	# 	if not os.path.exists(os.path.dirname(self._outputFilePath)):
	# 		try:
	# 			os.makedirs(os.path.dirname(self._outputFilePath))
	# 		except OSError as exc: # Guard against race condition
	# 			if exc.errno != errno.EEXIST:
	# 				raise
		   
	# 	# update the file
	# 	hnd = open(self._outputFilePath, 'a+')
	# 	hnd.write("n " + str((self.process_idx,self.global_t,self.max_neg_cc))+"\n")
	# 	hnd.close()

	# def output_solution(self,cc):
	# 	# create path to file if necessary (it shouldn't be, the path should've been created by the training program
	# 	if not os.path.exists(os.path.dirname(self._outputFilePath)):
	# 		try:
	# 			os.makedirs(os.path.dirname(self._outputFilePath))
	# 		except OSError as exc: # Guard against race condition
	# 			if exc.errno != errno.EEXIST:
	# 				raise
		   
	# 	# update the file
	# 	hnd = open(self._outputFilePath, 'a+')
	# 	hnd.write("s " + str((self.process_idx,self.global_t,cc,self.state))+"\n")
	# 	hnd.close()
	
	def setOutputFilePath(self,path):
		#self._outputFilePath = path + "/outputOvercountsRun_"+ str(uuid.uuid4()) + ".txt"
		self._outputFilePath = path + "/output.txt" 

	def setGlobal_t(self, global_t):
		self.global_t = global_t

	def setProcessIdx(self, idx):
		self.process_idx = idx
