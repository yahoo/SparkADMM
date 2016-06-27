# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

import sys,re,math
try: import simplejson as json
except ImportError: import json
import numpy as np
from scipy import sparse
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge, Lasso
import argparse as ap
from numpy import linalg as LA
sys.path.append('../')

def pos(x):
        return np.maximum(x,0)

def SoftThreshold(kappa, a):
        return pos(a-kappa) - pos(kappa+a) + kappa + a

def upd(d,k):
        if not k in d:
                d[k] = len(d)

def objective(z,A,b,lam):
        v = b - A*z
        return LA.norm(v)*LA.norm(v) + lam*LA.norm(z,1)
        

def localupdate(b,A,z,u,rho,eps):
        ridge = Ridge(alpha=rho/2.0, fit_intercept=False, tol=eps)
        #print "b",b
        #print "z",z
        #print "u",u
        #print A * (z-u/rho)
        b_new = b - A * (z-u/rho)
        #print "bnew",b_new
        ret = ridge.fit(A,b_new)
        #print ret
        #print ret.coef_
        return (ret.coef_ + (z-u/rho))

if __name__ == '__main__':
        ps = ap.ArgumentParser()
        ps.add_argument('--datafile', dest='datafile',\
                                        default='../data/LR_example.txt')
        ps.add_argument('--outcol', dest='outcol',\
                                        default='output_1') # for Reuters.txt, use output0 etc
        ps.add_argument('--N_parts',default=10,type=int, help='Number of data partitions')
        ps.add_argument('--N_out',default=100,type=int, help='Number of outer iterations')
        ps.add_argument('--rho',default=10,type=float, help='RHO')
        ps.add_argument('--lam',default=10,type=float, help='LAMBDA')

        ps.add_argument('--eps_in',default = 1.e-3, type = float, help = 'Epsilon for inner loop convergence')
        args = ps.parse_args()
        
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
        print(n_data)
        J = map(lambda k: fd[k], J)
        A = sparse.csr_matrix((V,(I,J)),shape=(I[-1]+1,len(fd)))
        X = np.zeros((args.N_parts,A.shape[1])).astype(float)
        U = np.zeros((args.N_parts,A.shape[1])).astype(float)
        size_part = int(n_data/args.N_parts)
        for local_machine in range (0,args.N_parts):
                B = np.array(A.todense()[(local_machine*size_part):((local_machine+1)*size_part),:])
                #print np.where(~B.any(axis=0))[0]
                X[local_machine,np.where(~B.any(axis=0))] = float('NaN')        
                U[local_machine,np.where(~B.any(axis=0))] = float('NaN')
        print(X)
        print(U)
        counts = args.N_parts - (X != 0).sum(0)
        print(counts)
        z = np.zeros(A.shape[1]).astype(float)
        b = np.array([y[args.outcol] for y in Y])
        for iter in range(0,args.N_out):
                print "Iteration:",iter
                print "Objective:",objective(z,A,b,args.lam)
                for local_machine in range (0,args.N_parts):
                        
                        #print local_machine
                        #print(localupdate(b,A[(local_machine*size_part+1):(local_machine+1)*size_part,:],z,U[:,local_machine],args.rho,args.eps_in)             )
                        X[local_machine] = localupdate(b[(local_machine*size_part):((local_machine+1)*size_part)],A[(local_machine*size_part):((local_machine+1)*size_part),:],z,U[local_machine],args.rho,args.eps_in)
                z = SoftThreshold((args.lam/(counts*args.rho)),np.nanmean(X,axis=0) + np.nanmean(U,axis=0)/args.rho)
                #print "z",z
                #print "U",U
                #print "X",X
                #print np.subtract(X,z)
                U = U + args.rho*(np.subtract(X,z))
