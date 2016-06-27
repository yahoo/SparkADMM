# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

import sys,logging,pandas,datetime
import numpy as np
from ADMMDataFrames import SparseDataFrame,reconstructDataFrame,unravelDataFrame
from AbstractSolver import AbstractSolver

try: import simplejson as json
except ImportError: import json


def normalize_row(X):
	""" Normalizes the row of a matrix. 

	Used to generate synthetic data.
	"""
	X = np.array(X)
	n,d = X.shape
	row_norms = np.sqrt(np.sum(np.multiply(X,X),1))
	row_norms_rep =  np.asarray( [row_norms,]*d ).T
	return np.divide(X,row_norms_rep)

def generateData(n,d,m):
	"""
	Generates a synthetic logistic regression dataset.

	Used for testing purposes. 
	"""
	features = [ 'input_'+str(i) for i in range(d)]
	outputs =  [ 'output_'+str(i) for i in range(m)]
	X = np.random.normal( size= (n,d))
	beta = np.random.normal( size= (m,d))
	beta = normalize_row(beta) 
	Y = np.sign(2.0/ (1.0+np.exp(-np.dot(X,beta.T))) - 1.0)
  
	inputDF = SparseDataFrame(X,columns = features )
	outputDF = SparseDataFrame(Y, columns = outputs )
	true_beta = SparseDataFrame(beta,index= outputs, columns =features)
  
	return inputDF,outputDF,true_beta

def saveFileOutput(inputDF,outputDF, filename):
	"""
	Stores the inputs and outputs in two respective dataframes in a file.

	The resulting file contains three tab separated columns. The first is an incremental index, the second is a json object containing the inputs, and the third is a json object containing the outputs.
	"""
	n,d = inputDF.shape
	with open(filename,'w') as f:
		for i in range(n):
			f.write(str(i)+'\t'+json.dumps(inputDF.iloc[i,:].to_dict())+'\t'+json.dumps(outputDF.iloc[i,:].to_dict())+'\n')


class GradientDescent:
	"""
	A class implementing gradient descent.

	Can be used to solve an arbitrary unconstrained convex minimization problem.
	"""

	def __init__(self,alpha=0.3,beta=0.7):
		"""
		Constructor. Takes as input alpha and beta parameters used in linear backtracking.
		"""

		self.alpha = alpha
		self.beta = beta
   
	def linear_backtracking(self,x,f,grad,dx):
		"""
		Backtracking line search (see Boyd and Vanderberghe, Convex Optimization, pp. 464) 

		Takes as input
		-x: a vector 
		-f: a convex function
		-grad: the gradient of f at x (a vector)
		-dx: a direction (a vector)

		and returns a scalar t such that:

		f(x+ t * dx) <= f(x) + self.alpha * t * np.dot(grad,dx): 
		"""
		t = 1.0
	
		#eps = 0.0001
		#est_grad = [ (f(x+dd)-f(x))/eps for dd in [  np.array([ (i == j) * eps for i in range(len(x))]) for j in range(len(x))] ]
		#print est_grad,grad

		while f(x+ t * dx) > f(x) + self.alpha * t * np.dot(grad,dx):
			t = self.beta * t
		return t 

	def minimize(self,f,gradient,x0,eps=1.e-3,max_it = 1000):
		"""
		"""
		x = x0
		it = 0
		grad = gradient(x)

		while it < max_it and np.linalg.norm(grad)>eps:
			t = self.linear_backtracking(x,f,grad,-grad)
			x = x - t * grad
			grad = gradient(x) 
			it += 1		
			logging.debug(str(datetime.datetime.now())+'- Iteration:'+str(it)+ '\tGradient Norm:'+str(np.linalg.norm(grad))+'\t Step: '+str(t))
	
		return x, np.linalg.norm(grad), it < max_it, it

class LogisticRegressionSolver(AbstractSolver):
	def __init__(self,gamma=0.1,max_it=1000,eps=1.e-3,alpha = 0.3,beta =0.5):
		self.gamma = gamma
		self.max_it = max_it
		self.eps = eps
		self.alpha = alpha
		self.beta = beta 

	def readPointBatch(self,iterator):
		inputDicts = {}
		outputDicts = {}
	
		for line in iterator:
			try:
				key,inputString,outputString = line.strip('\r\t\n ').split('\t')
			except:
				logging.warning(str(datetime.datetime.now())+'- Skipping record, unable to read tab separated line: '+line)
				continue

			try:
				inputDict = json.loads(inputString)
			except:
				logging.warning(str(datetime.datetime.now())+'- Skipping record '+key+', unable to read input data json: '+inputString)
				continue

			try:
				outputDict = json.loads(outputString)
			except:
				logging.warning(str(datetime.datetime.now())+'- Skipping record '+key+', unable to read output json: '+outputString)
				continue
	
			inputDicts[key] = inputDict
			outputDicts[key] =  outputDict
  
		inputDF = pandas.DataFrame.from_dict(inputDicts, orient = 'index').fillna(0.0)
		outputDF = pandas.DataFrame.from_dict(outputDicts, orient = 'index').fillna(0.0)
		keys = [ (response,feature) for feature in list(inputDF.columns) for response in list(outputDF.columns)]

		stats = {}
		stats['Data Points'] = len(inputDF.index)
		stats['Features'] = len(keys)
		stats['Outputs'] = len(outputDF.columns)

		return (inputDF,outputDF),keys,stats

	def gradient(self,X,y,beta):
		n,d = X.shape
		scale_vec = np.divide(-y,1 + np.exp(np.multiply(y, np.dot(X,beta))))
		scale_rep = np.asarray( [scale_vec,]*d ).T
		X_scaled = np.multiply(X,scale_rep)
		return sum(X_scaled,0)
   
	def loss(self,X,y,beta):
		return np.sum(np.log(1.0+np.exp(np.multiply(-y, np.dot(X,beta)))))

	def proximal_loss(self,X,y,beta,rho,beta_target):
		return 	self.loss(X,y,beta) + rho * np.dot(beta-beta_target,beta-beta_target)
   
	def proximal_gradient(self,X,y,beta,rho,beta_target):
		return self.gradient(X,y,beta)+2*rho*(beta-beta_target)	



	def solveSingle(self,X,y,rho,beta_target):
		GD = GradientDescent(alpha = self.alpha,beta = self.beta)
	
		f = lambda beta : self.proximal_loss(X,y,beta,rho,beta_target) 
		df = lambda beta : self.proximal_gradient(X,y,beta,rho,beta_target) 
		return GD.minimize(f,df,beta_target,self.eps,self.max_it)

	def solveProximal(self,data,rho,master_Z):
		inputDF,outputDF = data

		n,d = inputDF.shape
		n,m = outputDF.shape

		features = inputDF.columns
		outputs = outputDF.columns
 
		#betas = SparseDataFrame(np.zeros(m,d), index = outputs, columns = features)

		#consolidate input data with master Z (features and/or outputs may be missing from either tables)
		#ghost_Z = SparseDataFrame(np.zeros((m,d)),index = outputs, columns = features)
		#target_Z = ghost_Z + master_Z
		#the columns (rows) of target_Z are a union of the columns (rows) of ghost_Z and master_Z; zeros are added as necessary

		target_Z = reconstructDataFrame(master_Z)

		betas = SparseDataFrame(target_Z.copy())
		
		grad_norm_sum = 0.0
		converged_sum = 0.0
		it_sum = 0.0

		count = 0.0
		for out in outputDF.columns: 
			beta_target = np.array(target_Z.loc[out,features])
			y = np.array(outputDF.loc[:,out])
			X = np.array(inputDF)
			beta, grad_norm, converged, it   = self.solveSingle(X,y,rho,beta_target)
			
			count += 1.0 
			grad_norm_sum +=grad_norm
			converged_sum += converged
			it_sum += it

			logging.info(str(datetime.datetime.now())+ '- Converged: ' + str(converged)+' Gradient norm: '+ str(grad_norm) )
			logging.info( str(datetime.datetime.now())+'- Beta learned: '+ str(beta))
			betas.loc[out,features] = beta
	
		stats = {}
		stats['Grad']=grad_norm_sum/(1.*count)
		stats['It']=it_sum/(1.*count)
		stats['Conv']=converged_sum/(1.*count)

		return unravelDataFrame(betas),stats
			
	def localObjective(self,data,Z):
		inputDF,outputDF = data

		n,d = inputDF.shape
		n,m = outputDF.shape

		features = inputDF.columns
		outputs = outputDF.columns
 
		#betas = SparseDataFrame(np.zeros(m,d), index = outputs, columns = features)

		#consolidate input data with  Z (features and/or outputs may be missing from either tables)
		#ghost_Z = SparseDataFrame(np.zeros((m,d)),index = outputs, columns = features)
		#target_Z = ghost_Z + Z
		#the columns (rows) of Z are a union of the columns (rows) of ghost_Z and master_Z; zeros are added as necessary

		target_Z = reconstructDataFrame(Z)

		result = 0.0
		for out in outputDF.columns: 
			beta = np.array(target_Z.loc[out,features])
			y = np.array(outputDF.loc[:,out])
			X = np.array(inputDF)
			result +=  self.loss(X,y,beta)	

		return result

	

if __name__=='__main__':
   logging.basicConfig(level=logging.DEBUG)
   #n = 10000
   #d = 4
   #m = 3
   rho = 0.1
   #inputDF,outputDF,true_beta = generateData(n,d,m)
   #saveFileOutput(inputDF,outputDF,'LR_example.txt')
	
   LR = LogisticRegressionSolver()
   (inputDF,outputDF), keys,stats = LR.readPointBatch(sys.stdin)

   #fudged_betas = true_beta +  pandas.DataFrame(0.1* np.random.random( (m,d)) ,index= true_beta.index, columns =true_beta.columns)
   #logging.info(str(datetime.datetime.now())+'True Beta \n'+str(true_beta)) 
   
   zeros = reconstructDataFrame(dict(zip(keys, [0.0]*len(keys))))
   
   logging.info(str(zeros))

   betas = reconstructDataFrame(LR.solveProximal( (inputDF,outputDF),rho, unravelDataFrame(zeros)))
   betas = pandas.DataFrame(normalize_row(betas) ,  index=betas.index,columns = betas.columns)


   logging.info(str(datetime.datetime.now())+'  Estimated Betas \n'+str(betas))
   
   #betas = LR.solveProximal( (inputDF,outputDF),rho,fudged_betas)
   #betas = SparseDataFrame(normalize_row(betas),index= true_beta.index, columns =true_beta.columns)

   #logging.info(str(true_beta)) 
   #logging.info(str(betas))

