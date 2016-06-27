# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

#!/usr/bin/env python
import sys,re,math
try: import simplejson as json
except ImportError: import json
import numpy as np
from scipy import sparse
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge, Lasso
import argparse as ap
sys.path.append('../')
from LogisticRegressionSolver import GradientDescent, LogisticRegressionSolver
def upd(d,k):
	if not k in d:
		d[k] = len(d)

def objective(L,X,y,lamb):
	v = L.predict(X) - y
	return 1. / 1.0 * sum(t*t for t in v) + lamb * sum(abs(t) for t in L.coef_)

def objective_logistic(L,X,y,lamb):
	#print X.shape
	#print y.shape
	#print L.coef_.shape
	#print np.multiply(-y, np.dot(X,L.coef_.transpose()))
	ret = 1./1.0 * sum(np.log(1.0+np.exp(np.multiply(-y, np.dot(X.todense(),L.coef_.transpose())))).transpose().tolist()[0]) + lamb * sum(abs(t) for t in L.coef_.transpose().tolist()[0])
	#print ret
	return ret

if __name__ == '__main__':
	ps = ap.ArgumentParser()
	ps.add_argument('--datafile', dest='datafile',\
					default='../data/LR_example.txt')
	ps.add_argument('--outcol', dest='outcol',\
					default='output_1') # for Reuters.txt, use output0 etc
	ps.add_argument('--N_in',default=100,type=int, help='Number of inner iterations')
	ps.add_argument('--eps_in',default = 1.e-3, type = float, help = 'Epsilon for inner loop convergence')
	ps.add_argument('--lamb',default=10,type=float,help = 'l1 regularization factor' )
	args = ps.parse_args()

	lamb = args.lamb

	I,J,V,Y=[],[],[],[]
	fd = {}
	with open(args.datafile) as f:
		for i,line in enumerate(f.readlines()):
			id, x, l = line.strip().split('\t')
			for k,v in json.loads(x).items():
				I.append(i)
				J.append(k)
				V.append(v)
				upd(fd,k)
			Y.append(json.loads(l))
	n_data = i + 1.0
	Alpha = lamb / ( 2.0 * n_data )
	J = map(lambda k: fd[k], J)
	X = sparse.coo_matrix((V,(I,J)),shape=(I[-1]+1,len(fd)))
	#lr = LR(dual=False,C=lamb / 4.0)
	#lr = Ridge(alpha=0)
	lr = Lasso(alpha = Alpha ,fit_intercept=False,max_iter=10000,tol=0.001)

	target = [y[args.outcol] for y in Y]
	L = lr.fit(X,target) # ,omega=np.array([0.,-2000.,0.,0.,0.]))


	print objective(L,X,np.array(target),lamb)

	#print objective_logistic(L,X,np.array(target),lamb)

	#print L.coef_.tolist(), L.intercept_
	#lr2 = LogisticRegressionSolver(max_it=args.N_in,eps=args.eps_in)
	#Zl, stats = lr2.solveProximal(data, rho=0.5 * rho, master_Z = ZmUl)
