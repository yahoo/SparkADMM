# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from abc import ABCMeta, abstractmethod

class AbstractSolver:
	"""
	An abstract class for solving separable problems with an l1 penalty through ADMM.
	"""

	__metaclass__ = ABCMeta

	@abstractmethod
	def readPointBatch(self,iterator):
		"""An abstract method receiving input data from a partition, and creating a single batch object. 
		
		The SparkADMM code will read data line by line from a file, partition it across machines, and create python objects storing 
		the data in each machine. An implementation of this abstract method specifies how this data should be implemented.

                This method should receive as input an 

	        iterator<string> 
 
                i.e., a list of strings; each string corresponds to a line in the file storing the data; each such line may 
		represent a different  datapoint d_i. It should then return a 3-tuple 
		off the following form: 
		
		(data,keys,stats)

		where: 
		
		* data is a collection (e.g., list, numpy array, pandas dataframe, etc.) representing the data d_i read. It is up to the 
		  implementation to determine how to best represent the data; whatever the representation is, this ought to be used by the
		  other two functions in the code.

		* keys is a list of keys (e.g., strings) representing which global variables are used by this partition.

		* stats is a (possibly empty) dictionary of key:numerical_value pairs, that contains some statistics about the data in this partition.
		  For example, the number of lines read, the number of global variables, the processing time, etc., 
                  can be reported in this dictionary. It is up to the developer to determine which statistics to report, if any. SparkADMM
		  will print the mean, max, and mean values of all these statistics across partitions.
		"""
		pass


	@abstractmethod
	def localObjective(self,data,Z):
		"""Abstract method receiving input data_i from partition i, and global variables, 
		and returns the local evaluation of the objective over the local data.

		The input of the function is
		* data: a data object (e.g., list, numpy array, pandas dataframe, etc.) representing the data in this partition. 
		  This is of the same form as the data returned by the readPointBatch function.
		* Z: a dictionary storing variables as key:numerical_value pairs. The dictionary will only contain values for keys in the 'keys' list 
		  returned by the readPointBatch function for this partition

		
		The output of the function is a scalar value, equal to 
		
		\sum_{d_i\in data} F(Z;d_i), 
		i.e., the value of the objective evaluated at this partition's data.  Recall that ADMM attempts to solve 
		
		\sum_{k=1}^K \sum_{i\in data_k} F(Z;d_i) + lam * || Z ||_1, so SparkADMM 
		uses this function to evaluate and keep track of the global objective.
		"""
		pass

	@abstractmethod
	def solveProximal(self,data,rho,master_Z):
		'''A local solver, including a proximal term. This implements the update of local variables in the ADMM algorithm.

		The input to the function is
		* data:  a data object (e.g., list, numpy array, pandas dataframe, etc.) representing the data in this partition. 
		  This is of the same form as the data returned by the readPointBatch function.

		* rho: a scalar value (e.g., 0.1)

		* master_Z: a dictionary storing variables as key:numerical_value pairs. The dictionary can be assumed to contain values 
	          only for keys in the 'keys' list returned by the readPointBatch function for this partition.


		The function should return a pair
		
		* (Z,stats)
		
		where Z is dictionary of the form { key:numerical_value,...}, obtained as the solution of the following
		minimization problem:

	   	Z = argmin_{Z} \sum_{d_i \in data} F(Z;d_i) + rho *\| Z - master_Z   \|_2^2 

		and stats is  a (possibly empty) dictonary of the form {statistic_label:numerical_value} containing statistics that the 
		programmer wishes to report (e.g., the time to completion, the number of iterations if the minimization was done by an iterative algorithm, whether the algorithm converged, etc.) It is up to the programmer to determine what statistics she wishes to report.

		'''
		pass	

