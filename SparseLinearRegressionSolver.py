# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

import sys,logging,datetime,itertools,math
import numpy as np
from AbstractSolver import AbstractSolver
from scipy import sparse
from sklearn.linear_model import Ridge

try: import simplejson as json
except ImportError: import json
def upd(d,k):
    if not k in d:
        d[k] = len(d)

def SparseDotProduct(vec1,vec2):
    # v1,v2 are float valued dictionaries
    # pandas is too cumbersome, as one needs to first expand the vectors to match dimensions
    # alternatively, we can use numpy sparse matrix with one row
    if len(vec1) > len(vec2): vec1,vec2 = vec2,vec1
    return sum(v1 * vec2.get(k,0.0) * 1.0 for k,v1 in vec1.items() )


class SparseLinearRegressionSolver(AbstractSolver):
    def __init__(self,gamma=0.1,max_it=1000,eps=1.e-3,alpha = 0.3,beta =0.5):
        self.gamma = gamma
        self.max_it = max_it
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    def readPointBatch(self,iterator):
        inputDF = {}  # "0" -> {}
        outputDF = {} # output0 -> {"0":0.1, etc}
        keys = set()   # dict of sets

        for line in iterator:
            try:
                key,inputString,outputString = line.strip('\r\t\n ').split('\t')
            except:
                logging.warning(str(datetime.datetime.now())+'- Skipping record, unable to read tab separated line: '+line)
                continue

            try:
                inputDict = json.loads(inputString)
                inputDF[key] = inputDict
            except:
                logging.warning(str(datetime.datetime.now())+'- Skipping record '+key+', unable to read input data json: '+inputString)
                continue

            try:
                outputDict = json.loads(outputString)
                for k,v in outputDict.items():
                    outputDF.setdefault(k,{})[key] = v
            except:
                logging.warning(str(datetime.datetime.now())+'- Skipping record '+key+', unable to read output json: '+outputString)
                continue
            keys.update(itertools.product(outputDict.keys(),inputDict.keys()))


        stats = {}
        stats['Data Points'] = len(inputDF)
        stats['Features'] = len(keys)
        stats['Outputs'] = len(outputDF.keys())

        return (inputDF,outputDF),list(keys),stats

    def loss(self,X,y,beta):
        """
        Linear regression loss without regularization terms; 0.5 to be consistent with sklearn Lasso (the global problem we try to solve)
        :param X: data matrix for the local partition
        :param y: corresponding target vector for the local partition
        :param beta: feature weights for features used in X
        :return: \|X beta - y\|^2
        """
        ret = ''
        for k,v in X.items():
            ret += 'y = %s \n v = %s \n beta = %s \n SparseDotProduct(v,beta) = %s \n'%\
                    (str(y),str(v),str(beta),str(SparseDotProduct(v,beta)))
        ret += 'return = %s\n'%str(1.0 * (sum((math.pow((y[k] - SparseDotProduct(v, beta)),2)) for k,v in X.items())))
        #raise ValueError(ret)
        return 1.0 * (sum((math.pow((y[k] - SparseDotProduct(v, beta)),2)) for k,v in X.items()))


    def solveSingle(self,inputDF,outputDict,rho,beta_target):
        I,J,V,Y=[],[],[],[]
        fd = {} # mapping feature names to consecutive integers, starting with 0
        for i,(id, x) in enumerate(inputDF.items()):
            l = outputDict.get(id)
            for k,v in x.items():
                I.append(i)
                J.append(k)
                V.append(v)
                upd(fd,k)
            Y.append(l)
        J = map(lambda k: fd[k], J)
        X = sparse.coo_matrix((V,(I,J)),shape=(I[-1]+1,len(fd)))
        fd_reverse = [k for k,v in sorted(fd.items(), key = lambda t: t[1])]
        # y_new = y - X . beta_target
        # converting a proximal least square problem to a ridge regression
        ZmUl = np.array([beta_target.get(k,0) for k in fd_reverse])
        y_new = np.array(Y) - X * ZmUl
        ridge = Ridge(alpha =  rho , fit_intercept=False)
        ret = ridge.fit(X,y_new)
        #ret = self.lr.fit(X,y_new)
        # ordered list of feature names according to their integer ids in fd
        #raise ValueError('fd_reverse = %s \n X = %s \n J = %s \n I = %s \n V = %s \n Y = %s \n y_new = %s \n ret.coef_ = %s \n ZmUl = %s \n'\
        #            %(str(fd_reverse), str(X), str(J), str(I), str(V), str(Y), str(y_new), str(ret.coef_), str(ZmUl)))
        return dict(zip(fd_reverse, (ret.coef_ + ZmUl).tolist()))


    def solveProximal(self,data,rho,master_Z):
        inputDF, outputDF = data
        features_set = set(k for v in inputDF.values() for k in v.keys() )
        # master_Z is the same as z - u_\ell    dict of dict output_col -> {fn -> val}
        betas = {}
        for k,out in outputDF.items():
            beta_target = dict((a,b) for (c,a),b in master_Z.items() if a in features_set and c == k)
            betaDict  = self.solveSingle(inputDF,out,rho,beta_target)
            # betaDict is the solved Z_\ell; it doesn't contain U_\ell
            logging.info( str(datetime.datetime.now())+'- Beta learned: '+ str(betaDict))
            betas.update(dict(((k,f),v) for f,v in betaDict.items()))
        stats = {}
        return betas,stats

    def localObjective(self,data,Z):
        inputDF,outputDF = data # remember outputDF is transposed

        result = 0.0
        for k,target in outputDF.items():
            beta = dict((b,v) for (a,b),v in Z.items() if a == k)
            result +=  self.loss(inputDF,target,beta)
        #raise ValueError('loss!!! = %s\n'%str(result))
        return result



if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    #n = 10000
    #d = 4
    #m = 3
    rho = 0.1
    #inputDF,outputDF,true_beta = generateData(n,d,m)
    #saveFileOutput(inputDF,outputDF,'LR_example.txt')

    LR = SparseLinearRegressionSolver()
    (inputDF,outputDF), keys,stats = LR.readPointBatch(sys.stdin)

    #fudged_betas = true_beta +  pandas.DataFrame(0.1* np.random.random( (m,d)) ,index= true_beta.index, columns =true_beta.columns)
    #logging.info(str(datetime.datetime.now())+'True Beta \n'+str(true_beta))

    #zeros = reconstructDataFrame(dict(zip(keys, [0.0]*len(keys))))

    #logging.info(str(zeros))
    #print keys

    Z_init = dict(((k,v),0.0) for k,v in keys)

    betas,stats = LR.solveProximal( (inputDF,outputDF),rho, Z_init)

    logging.info('local object = ' + str(LR.localObjective((inputDF,outputDF), Z_init)) + '\n')
    #betas = pandas.DataFrame(normalize_row(betas) ,  index=betas.index,columns = betas.columns)


    logging.info(str(datetime.datetime.now())+'  Estimated Betas \n'+str(betas))

    #betas = LR.solveProximal( (inputDF,outputDF),rho,fudged_betas)
    #betas = SparseDataFrame(normalize_row(betas),index= true_beta.index, columns =true_beta.columns)

    #logging.info(str(true_beta))
    #logging.info(str(betas))

