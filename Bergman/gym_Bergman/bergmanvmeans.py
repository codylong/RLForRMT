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
import scipy
from scipy.stats import moment as mmt
#import mpmath as mpm
#from mpmath import *

##set the numerical accuracy in mpm:
#mp.dps = 200

#python train_a3c_gym.py 4 --steps 10000 --env wishart-v0 --outdir output/ --mean 0 --sigma 0.1 --nmod 2 --eval-interval 1000

class Bergman(gym.Env):

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
		self.solved_factor=1e10
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
		self.moments = eval(self.moments)
		self.nmoments = len(self.moments)
		self.weights = eval(self.weights)
		self.smallest_dist = 1e10
		if len(self.weights) != len(self.moments):
			self.weights = [1 for ii in range(len(self.moments))]
		#old code where we gave each entry its own distribution
		# self.means = [[0 for i in range(self.nmod)] for j in range(self.nmod)]
		# #self.vars = [[1/self.nmod**(1/2) for i in range(self.nmod)] for j in range(self.nmod)]
		# self.vars = [[0 for i in range(self.nmod)] for j in range(self.nmod)]
		self.means = [0 for i in range(self.nmod + 1)]
		self.initial_means = [0 for i in range(self.nmod + 1)]
		#self.vars = [[1/self.nmod**(1/2) for i in range(self.nmod)] for j in range(self.nmod)]
		self.vars = [1./np.float(self.nmod)**(1./2.) for i in range(self.nmod + 1)]
		self.initial_vars = [1./np.float(self.nmod)**(1./2.) for i in range(self.nmod + 1)]
		self.stepsize = np.float((self.stepsize))*1./np.float(self.nmod)**(1./2.)
		#self.sigma = self.sigma
		#self.state = self.vars[0]
		self.state = self.vars + self.means
		#for i in range(1,self.nmod):
		#	self.state = self.state + self.vars[i]
		# self.action_space = spaces.Discrete(2*self.nmod*self.nmod)
		# self.observation_space = spaces.Box(low=int(self.min_var), high=int(self.max_var), shape=(self.nmod*self.nmod, 1))
		self.action_space = spaces.Discrete(4*self.nmod + 4)
		self.observation_space = spaces.Box(low=int(self.min_var), high=int(self.max_var), shape=(2*self.nmod + 2, 1))
		

	def step(self,action):
		done = False
		idx, sign = (action-action%2)/2, (-1)**(action%2)
		self.state[idx] += np.float(sign)*self.stepsize
		if self.state[idx] <= 0:
				self.state[idx] += -2*sign*self.stepsize
		#ii = idx / self.nmod
		#jj = idx % self.nmod
		#print "HITHERE", ii, jj, len(self.vars), type(self.numdraws)
		#self.vars[ii][jj] += sign*self.stepsize
		self.vars = self.state[:self.nmod + 1]
		self.means = self.state[self.nmod+1:]
		mean, moments = self.sample()

		dist  = self.dist(moments,mean)


		if dist < self.tol:
			done = True
			#print 'huzzah!', cc, self.state
			my_reward = float(self.solved_factor)
		else:
			my_reward = self.reward(moments,mean)
			self.trackscore += my_reward

		if dist < self.smallest_dist:
			self.smallest_dist = dist
			print 'smalldist', self.process_idx, dist, self.state
			self.output_min_dist()

		return np.array([float(k) for k in self.state]), my_reward, done, {}

	def output_min_dist(self):
		# create path to file if necessary (it shouldn't be, the path should've been created by the training program
		if not os.path.exists(os.path.dirname(self._outputFilePath)):
			try:
				os.makedirs(os.path.dirname(self._outputFilePath))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise
		   
		# update the file
		hnd = open(self._outputFilePath, 'a+')
		hnd.write("smallestdist " + str((self.process_idx,self.global_t,self.smallest_dist,self.state))+"\n")
		hnd.close()
		
	def reset(self):
		# self.state = self.vars[0]
		# for i in range(1,self.nmod):
		# 	self.state = self.state + self.vars[i]
		self.state = self.initial_vars + self.initial_means
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
	
	def dist(self,moments, mean):
		thedist = sum([self.weights[ii]*(self.moments[ii] - moments[ii])**2 for ii in range(self.nmoments)]) + (self.mean - mean)**2
		return thedist


	def reward(self,moments, mean):
		d = self.dist(moments, mean)
		#do = self.dist(self.occ)
		#print np.float(1/d), np.float(1/d**self.pow) 
		return np.float(1/d)
		#return 0
		

	def random_wishart(self): # pos def metric
		#A = np.array([[np.random.normal(loc = 0,scale = np.exp(self.vars[pp][mm])) for mm in range(self.nmod)] for pp in range(self.nmod)])
		#A = np.array([np.random.normal(loc = 0,scale = np.exp(self.vars[pp]),size = self.nmod)for pp in range(self.nmod)])
		A = np.array([np.random.normal(loc = self.means[pp], scale = self.vars[pp],size = self.nmod + 1)for pp in range(self.nmod + 1)])
		#A = [[np.random.normal(size=(self.nmod,self.nmod), scale = self.sigma)]]
		#A = np.array([[self.sigma*mpm.sqrt(mpf(2))*mpm.erfinv(mpf(2)*mpm.rand()-mpf(1)) for i in range(self.nmod)] for j in range(self.nmod)])
		#mpm addition
		#A = np.array([[mpm.npdf(0,self.sigma) for i in range(self.nmod)] for j in range(self.nmod)])
		#print "metric test\n", A
		p = np.dot(A,A.transpose())
		p00 = p[self.nmod,self.nmod]
		p0a = [p[self.nmod][i] for i in range(self.nmod)]
		pab = [[p[j][i] for i in range(self.nmod)] for j in range(self.nmod)]
		m = pab/p00 - [[p0a[i]*p0a[j] for i in range(len(p0a))] for j in range(len(p0a))]/p00**2

		return m

	def sample(self):
		#samplemoments = [[] for kk in range(self.nmoments)] 
		evals = []
		for num in range(self.numdraws):
			evals  = evals + list(np.linalg.eig(self.random_wishart())[0])
		themean = scipy.mean(evals)
		samplemoments = [mmt(evals, moment = ii+2) for ii in range(self.nmoments)]
		#samplemoments[ii].append(mmt(evals, moment = ii+1))
		return themean, samplemoments


	def init_output(self):
		if not os.path.exists(os.path.dirname(self._outputFilePath)):
			try:
				os.makedirs(os.path.dirname(self._outputFilePath))
			except OSError as exc: # Guard against race condition
				if exc.errno != errno.EEXIST:
					raise

		# write header data
		hnd = open(self._outputFilePath, 'a+')
		hnd.write("head: " + str((self.nmod,self.numdraws,self.stepsize,self.mean))+"\n")
		hnd.close()


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
