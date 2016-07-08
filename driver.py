# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from LogisticRegressionSolver import LogisticRegressionSolver,normalize_row
from SparseLinearRegressionSolver import SparseLinearRegressionSolver
from SparkADMM import SparkADMM
import logging,datetime,argparse
import numpy as np
import sys
import pandas

if __name__=='__main__':
	logging.basicConfig(level = logging.INFO)
	

	parser = argparse.ArgumentParser(description = 'Train a logistic regression model through ADMM.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('inputfile',help = 'Training data. This should be a tab separated file of the form: index _tab_ features _tab_ output , where index is a number, features is a json string storing the features, and output is a json string storing output (binary) variables. See data/LR-example.txt for an example.')
	parser.add_argument('outputfile',help = 'Output file')
	parser.add_argument('--lam',default=1.0,type=float, help='Regularization factor')
	parser.add_argument('--rho',default=10.0,type=float, help='ADMM rho')
	parser.add_argument('--N_parts',default=10,type=int, help='Number of partitions. Remember to also correspondingly set the number of executors, through --num-executors in spark-submit')
	parser.add_argument('--N_in',default=100,type=int, help='Number of inner iterations')
	parser.add_argument('--eps_in',default = 1.e-3, type = float, help = 'Epsilon for inner loop convergence')
	parser.set_defaults(run_full_in=True)
	parser.add_argument('--N_out', default=20,type=int, help='Number of outer iterations')
	parser.add_argument('--eps_out',default = 1.e-2, type = float, help = 'Epsilon for outer loop convergence')
	parser.add_argument('--run_full_out', dest='run_full_out', action='store_true',help='Run all N_out outer iterations, even past converngece ')
	parser.add_argument('--run_conv_out', dest='run_full_out', action='store_false', help='Run outer iteratations at most until convergence')
	parser.set_defaults(run_full_out=True)




	args = parser.parse_args()


	#solver = SparseLinearRegressionSolver(max_it=args.N_in,eps=args.eps_in)
	solver = LogisticRegressionSolver(max_it=args.N_in,eps=args.eps_in)
	sadmm = SparkADMM(solver,rho=args.rho,lam = args.lam, N_out=args.N_out, N_parts =args.N_parts, eps = args.eps_out, run_full=args.run_full_out )
	
	sadmm.runADMM(args.inputfile,args.outputfile)
 
	logging.info(str(datetime.datetime.now())+'- Full log:\n'+sadmm.log)
	
	#Z = recontructDataFrame(dict(sadmm.Z.collect()))
	#logging.info(str(datetime.datetime.now())+'Final output:\n'+ str(pandas.DataFrame(normalize_row(Z), index = Z.index,columns = Z.columns))
