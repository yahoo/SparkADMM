# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

"""
Code supporting general consensus ADMM operations over Spark.
"""

import logging,datetime
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys

def ProximityOperator(kappa, a):
        """
        The soft thresholding operator, used to enforce L1 regularization. It produces the solution to: 
        \argmin_x 2 \kappa | x | + (x - a)^2 
        """
        if a > kappa: return a - kappa
        elif a < -kappa: return a + kappa
        else: return 0.

def simpleAddDictionaries(d1,d2):
        """
        Add the elements of two dictionaries. Assumes they have the same keys.
        """
        assert set(d1.keys()) == set(d2.keys())
        return  dict([ (key, d1[key]+d2[key]) for key in d1])

def simpleSubtractDictionaries(d1,d2):
        """
        Subtract the elements of the second dictionary from the ones in the first dictionary. Assumes both share the same keys.
        """
        assert set(d1.keys()) == set(d2.keys())
        return  dict([ (key, d1[key]-d2[key]) for key in d1])

def addDictionaries(d1,d2):
        """
        Add the elements of two dictionaries. Missing entries treated as zero.
        """
        lsum =  [( key, d1[key]+d2[key]) for key in d1 if key in d2]
        lsum +=  [( key, d1[key]) for key in d1 if key not in d2]
        lsum +=  [( key, d2[key]) for key in d2 if key not in d1]
        return dict(lsum)

def subtractDictionaries(d1,d2):
        """
        Subtract the elements of two dictionaries. Missing entries treated as zero.
        """
        d3 = dict([ (key,-d2[key]) for key in d2])
        return addDictionaries(d1,d3)

def tabStringDict(d):
        """
        Pretty print a dictionary into a string. 
        """
        return ' '.join([  ' '+str(key)+': '+str(d[key]) for key in d]) 

def maxDictionaries(d1,d2):
        """
        Return the maximum of the elements of two dictionaries. Assumes they share the same keys.
        """
        assert set(d1.keys()) == set(d2.keys())
        return dict( (key, max(d1[key],d2[key]))   for key in d1) 

def minDictionaries(d1,d2):
        """
        Return the minimum of two dictionaries. Assumes they share the same keys.
        """
        assert set(d1.keys()) == set(d2.keys())
        return dict( (key, min(d1[key],d2[key]))   for key in d1) 


class SparkADMM:
        """
        A class implementing Consensus ADMM over spark. 
        """

        def __init__(self,solver,rho = 0.1, lam = 0.1,N_out=50,N_parts=10,key_parts = None,eps=1.e-2, run_full=True):
                """
                Constructor. 

                -solver is an object from a class implementing the AbstractSolver class
                -rho is the scaling parameter of the quadratic/proximal term
                -lam is the regularization parameter of the l1 term
                -N_out is the number of outer iterations of ADMM to be executed, at most
                -N_parts is the number of partitions to be used in storing data and in ADMM operations.
                -key_parts is the number of partitions to be used in storing keys (i.e., variables), during the Z update stage. None sets key_parts=N_parts
                -eps sets the threshold on primal and dual residual norms to be used to determine the algorithm has converged
                -run_full: if this is true, eps will be ignored, and all N_out iterations will be executed
                """
                self.solver = solver
                self.N_out = N_out
                self.N_parts = N_parts
                if key_parts is None:
                        self.key_parts = N_parts
                else:
                        self.key_parts = key_parts 
                self.rho = rho
                self.lam = lam
                self.eps = eps
                self.run_full = run_full
                self.log = '' 
                self.logIt( str(datetime.datetime.now())+' ### Initializing SparkADMM object with  data partitions: %d,  key partitions: %d, iterations: %d,  rho: %f,  lambda: %f, eps: %f, run_full: %s' % (self.N_parts,self.key_parts,self.N_out,self.rho,self.lam,self.eps,str(self.run_full)) )

        def logIt(self,log_string):
                """
                Logging auxiliarry function.

                Logs log_string at INFO level, but also stores it in an internal string (self.log), for easy access at the conclusion of the execution.
                """

                logging.info(log_string)
                self.log += log_string+'\n'

        @staticmethod  
        def generateBatches(solver,key,iterator):
                """
                Prepare main rdd that stores data, Z_l, U_l variables, and some statistics. 

                Uses readPointBatch, from an abstract solver, that aggregates data read into a single variable (data).
                It is called by mapPartitionsWithIndex, so it receives as input:
                - the id/key of a partition
                - an iterator over the partition's data
                """

                data, keys, stats = solver.readPointBatch(iterator)
                Ul = dict(  zip (keys,[0.0]*len(keys))   )
                Zl = dict(  zip (keys,[0.0]*len(keys))   )
                return [ (key, (data,stats, Ul,Zl,0.0, float('Inf') )) ]

        @staticmethod 
        def localUpdate(solver,rho, Z,inputTuple):
                """
                Perform local ADMM updates.

                The function receives as input a parameter rho, the global variables Z (restricted to the keys relevant to the local optimization), 
                and an inputTuple of the form:
                data, stats, Ul, Zl, local_primal_residual,localObj

                The result is a new tuple, where
                data: remains the same
                stats: contains statistics about the update of Zl, as reported by the AbstractSolver
                Ul: is updated as Ul = Ul+Zl-Z
                Zl: is updates as Z_ell = argmin_{Z_ell} F(Z_ell) + \rho/2 *\| Z_ell - Z + U_ell  \|_2^2. This calls the solveProximal function provided by the internal solver.
                local_primal_residual : contains the local primal residual, i.e., ||Z_l-Z||_2^2
                localObj: contains the local objective value (without any regularization). This is evaluated on the input Z (so, technically, it is the objective at the previous step).
                """

                data, stats, Ul, Zl, local_primal_residual,localObj = inputTuple

                #evaluate local function
                localObj = solver.localObjective(data,Z)


                #Update Ul      
                #Ul = Ul+Zl-Z
                Ul = simpleSubtractDictionaries(simpleAddDictionaries(Ul,Zl),Z)

                #Update Zl
                #Z_ell = argmin_{Z_ell} F(Z_ell) + \rho/2 *\| Z_ell - Z + U_ell  \|_2^2
                ZmUl = simpleSubtractDictionaries(Z,Ul)


                Zl, stats = solver.solveProximal(data, rho=0.5 * rho, master_Z = ZmUl)



                #local primal residual  
                # ||Z_l-Z||_2^2
                ZlmZar = np.array(simpleSubtractDictionaries(Zl,Z).values()) 
                local_primal_residual = ZlmZar.dot(ZlmZar)

                return  (data, stats, Ul, Zl, local_primal_residual,localObj)
   
                
        def runADMM(self,input_filename,output_filename):
                """
                Main ADMM method.

                 Receives as input the name of the file storing the data, as well as the output file in which the learned Z variables will be stored when the algorithm terminates. 
                """

                def extractLocalKeys(inputTuple):
                        data, stats, Ul, Zl, local_primal_residual,localObj = inputTuple
                        return list(Ul.keys())  

                def unravelUlplZlAppendCounter(inputPair):
                        partitionKey, inputTuple = inputPair
                        data, stats, Ul, Zl, local_primal_residual,localObj = inputTuple
                        assert set(Zl.keys()) == set(Ul.keys())
                        return  [ (key,  (Zl[key]+Ul[key], 1)) for key in Ul] 


                sc = SparkContext(appName='PythonADMM')

                self.logIt(str(datetime.datetime.now())+ ' ### Reading data and initializing RDDs...')
                
                rdd = sc.textFile(input_filename,minPartitions=self.N_parts).mapPartitionsWithIndex(lambda key,iterator: SparkADMM.generateBatches(self.solver,key,iterator) ).cache()
                # (partition_id, (data, stats, Ul, Zl, local_primal_residual,localObj))
                
                keyPartitionIndex = rdd.mapValues(lambda (data, stats, Ul, Zl, local_primal_residual,localObj): Ul.keys()).flatMapValues(lambda x : list(x)).map(lambda x: (x[1],x[0])).partitionBy(self.key_parts).cache() # x[1] is a nested tuple, an element of Ul.keys(), i.e., feat ; x[0] = partition_id from sc.textFile()
                # output =  feat_partition (hidden) -> (feature, partition_id)

                master_Z = keyPartitionIndex.mapValues(lambda x: 0.0).cache()   # (feat_part, feature, feat_value)

                #strng = str(master_Z.collect())
                #logging.debug('### (key, value, partition_list): '+strng)

                actualNumParts = rdd.getNumPartitions() 
                actualKeyParts = master_Z.getNumPartitions()

                sumStats,maxStats,minStats = rdd.mapValues(lambda (data, stats, Ul, Zl, local_primal_residual,localObj): (stats,stats,stats)).values().reduce( lambda x,y: (simpleAddDictionaries(x[0],y[0]), maxDictionaries(x[1],y[1]),minDictionaries(x[2],y[2])) )
                reportStats =  dict([ (key, str(sumStats[key]/(1.0*actualNumParts))+','+str(minStats[key])+','+str(maxStats[key]))  for key in sumStats])

                self.logIt(str(datetime.datetime.now())+ ' ### ...Done reading data and initializing RDDs.')
                self.logIt(str(datetime.datetime.now())+ ' ### STATS\tData Partitions: %d\tKey Partitions: %d\t%s' % (actualNumParts,actualKeyParts,tabStringDict(reportStats)))
                
                self.logIt(str(datetime.datetime.now())+ ' ### Beginning iterations...' )

                iteration = 0
                primal_residual = float('Inf')
                dual_residual = float('Inf')
                while iteration <self.N_out and (self.run_full or dual_residual+primal_residual>self.eps):
                        iteration += 1 
                        collectedZs = master_Z.join(keyPartitionIndex,numPartitions = self.key_parts).map(lambda (key, (value,partition)) : (partition, (key,value))).groupByKey().mapValues(lambda x: dict(x)).partitionBy(self.N_parts, partitionFunc = lambda x: x)  # key = feature
                        # collectedZs gives (partition_id, {feat: value})

                        #print 'collectedZs printed!!!'
                        #print collectedZs.collect()

                        rdd = rdd.join(collectedZs,numPartitions=self.N_parts).mapValues(lambda (inputTuple,Z): SparkADMM.localUpdate(self.solver,self.rho, Z,inputTuple)).cache()
                        #print 'rdd printed!!!'
                        #print rdd.collect()
                        # unimportant
                        primal_sum, obj_sum, counter = rdd.map(lambda (partition_id, (data,stats,Ul,Zl,local_primal_residual,localObj)):
                                                                          (local_primal_residual,localObj,1) ).reduce(lambda x,y:(x[0]+y[0],x[1]+y[1],x[2]+y[2]) )
                        primal_residual = np.sqrt(primal_sum)
                        # end unimportant
                        old_Z = master_Z.cache()
                        #kappa = self.lam*1.0/(1.0* counter * self.rho) #  this is wrong!

                        # flatMap(unravelUlplZlAppendCounter):  each row is a data point in the input: (partitionKey, inputTuple) -> [ (key,  (Zl[key]+Ul[key], 1)) for key in Ul] ->flatten-> (key, (ZlplUl,1))

            #### verify that the reduceByKey is consistent with keyPartitionIndex in terms of partitioning of the features

                        #print 'rdd before soft thresholding!!!'
                        #print rdd.collect()

                        master_Z = rdd.flatMap(unravelUlplZlAppendCounter).reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1]), numPartitions = self.key_parts
                                                                                                        ).mapValues( lambda (summed, counter):
                                                                                                                        ProximityOperator(self.lam/(1.0 * counter * self.rho),summed/(1.0 * counter))).cache()
                        #print 'master_Z after soft thresholding!!!'
                        #print master_Z.collect()
                        dual_residual = self.rho * np.sqrt(master_Z.join(old_Z,numPartitions = self.key_parts).values().map(lambda x: (x[0]-x[1])*(x[0]-x[1])).sum())

                        lz = self.lam * old_Z.map(lambda x: np.absolute(x[1])).sum()
                        old_Z.unpersist()
                        global_obj = obj_sum + lz   # this is global obj from previous iteration!
                        #print 'obj_sum = ', obj_sum
                        #print 'lam * master_Z = ', lz

                        sumStats,maxStats,minStats = rdd.mapValues(lambda (data, stats, Ul, Zl, local_primal_residual,localObj): (stats,stats,stats)).values().reduce( lambda x,y: (simpleAddDictionaries(x[0],y[0]), maxDictionaries(x[1],y[1]),minDictionaries(x[2],y[2])) )
                        reportStats =  dict([ (key, str(sumStats[key]/(1.0*actualNumParts))+','+str(minStats[key])+','+str(maxStats[key]))  for key in sumStats])

                        self.logIt(str(datetime.datetime.now())+ ' ### It: '+str(iteration)+'\tPR: '+ str(primal_residual) +'\tDR: '+str(dual_residual)+'\tOBJ: '+str(global_obj) +'\t'+tabStringDict(reportStats))
                self.logIt(str(datetime.datetime.now())+ ' ### Saving Z to file %s...' % output_filename)
                master_Z.saveAsTextFile(output_filename)
                self.logIt(str(datetime.datetime.now())+ ' ### ... done saving to file %s.' %output_filename)
                sc.stop()
   
