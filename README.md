Spark ADMM
==========

The code in this repository provides a framework for solving arbitrary separable convex optimization problems with Alternating Direction Method of Multipliers (ADMM). In particular, the algorithm implemented is the generalized consensus algorithm described in the paper 

>Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. 
>S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein

which can be found [here](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html). The framework can be used to parallelize the solution of every problem of the form

  > min_{Z} \sum_{i=1}^N F(Z; d_i) + lam * ||Z||_1  

where `Z` in R^d, `F` is a convex function in `Z`, `d_i` are data points read from a file, `lam` is a regularization parameter, and `||.||_1` is the usual l1 norm.

The framework is built over Spark and is generic: to apply it to an arbitrary separable convex problem, a developer needs to implement only three functions (one that reads data from a file, one that evaluates the objective function, and one that solves a local optimization problem with an additional proximal penalty term). An example implementation of logistic regression is included in the code.

The code was developed by a team of Yahoo researchers including Stratis Ioannidis, Yunjiang Jiang, Nikolay Laptev, Hamid Javadi, and Saeed Amizadeh. It was used in the paper:

>Parallel News-Article Traffic Forecasting with ADMM. 
>S. Ioannidis, Y. Jiang, S. Amizadeh, and N. Laptev.
>International Workshop on Mining and Learning  from Time Series (MiLeTS), 2016

## Installation Instructions ##

Updated spark installation instructions can be found [here](http://spark.apache.org/docs/latest/). Make sure that your python distribution includes the following modules:
```
numpy
sklearn
pandas
logging
json
```



## Example Execution: Logistic Regression ##

To try out the code, upload the example data file to HDFS:

```
hadoop fs -put data/LR-example.txt
hadoop fs -chmod 755 LR-example.txt
```

and then run the following command:

```
spark-submit --master yarn --deploy-mode client --num-executors 20 --executor-memory 2g --driver-memory 2g --conf spark.driver.maxResultSize=0 --queue default  --py-files 'SparkADMM.py,LogisticRegressionSolver.py,ADMMDataFrames.py,AbstractSolver.py' driver.py LR_example.txt regression_output
```

Additional parameters can be passed to the driver. A help message is printed when calling

```
/homes/USERNAME/testroot/share/spark-*/bin/spark-submit driver.py -h
```

which returns this:

```
usage: driver.py [-h] [--lam LAM] [--rho RHO] [--N_parts N_PARTS] [--N_in N_IN] [--eps_in EPS_IN] [--N_out N_OUT] [--eps_out EPS_OUT] [--run_full_out]
                 [--run_conv_out]
                 inputfile outputfile

Train a logistic regression model through ADMM.

positional arguments:
  inputfile          Training data. This should be a tab separated file of the form: index _tab_ features _tab_ output , where index is a number, features is a
                     json string storing the features, and output is a json string storing output (binary) variables. See data/LR-example.txt for an example.
  outputfile         Output file

optional arguments:
  -h, --help         show this help message and exit
  --lam LAM          Regularization factor (default: 1.0)
  --rho RHO          ADMM rho (default: 10.0)
  --N_parts N_PARTS  Number of partitions. Remember to also correspondingly set the number of executors, through --num-executors in spark-submit (default: 10)
  --N_in N_IN        Number of inner iterations (default: 100)
  --eps_in EPS_IN    Epsilon for inner loop convergence (default: 0.001)
  --N_out N_OUT      Number of outer iterations (default: 20)
  --eps_out EPS_OUT  Epsilon for outer loop convergence (default: 0.01)
  --run_full_out     Run all N_out outer iterations, even past converngece (default: True)
  --run_conv_out     Run outer iteratations at most until convergence (default: True)
```

The resulting execution should print a final log that looks like this:

```
2015-05-28 22:09:36.473053 ### Initializing SparkADMM object with  data partitions: 10,  key partitions: 10, iterations: 20,  rho: 10.000000,  lambda: 1.000000, eps: 0.010000, run_full: True
2015-05-28 22:10:08.825243 ### Reading data and initializing RDDs...
2015-05-28 22:10:38.780563 ### ...Done reading data and initializing RDDs.
2015-05-28 22:10:38.780720 ### STATS    Data Partitions: 10 Key Partitions: 10   Outputs: 3.0,3,3  Features: 12.0,12,12  Data Points: 10000.0,9993,10054
2015-05-28 22:10:38.780822 ### Beginning iterations...
2015-05-28 22:11:02.650379 ### It: 1    PR: 46.8854597716   DR: 467.933781995   OBJ: 207986.725076   Grad: 0.000788458416586,0.000684263725427,0.000923344967733  It: 42.5,31.6666666667,52.0  Conv: 1.0,1.0,1.0
2015-05-28 22:11:23.226720 ### It: 2    PR: 22.1302771082   DR: 69.9754324662   OBJ: 22740.1688263   Grad: 0.000878238670161,0.000808567168721,0.000936343788982  It: 84.9,80.3333333333,88.3333333333  Conv: 1.0,1.0,1.0
2015-05-28 22:11:33.933045 ### It: 3    PR: 15.0054535762   DR: 47.442025163    OBJ: 15592.1458912   Grad: 0.000654701245552,0.000409245286274,0.000904717922178  It: 39.5,35.0,43.6666666667  Conv: 1.0,1.0,1.0
2015-05-28 22:11:44.543618 ### It: 4    PR: 11.603716828    DR: 36.6818811479   OBJ: 12854.6209477   Grad: 0.000819398193575,0.000588275613865,0.000950790164178  It: 41.5333333333,35.3333333333,48.0  Conv: 1.0,1.0,1.0
2015-05-28 22:11:59.232952 ### It: 5    PR: 9.58918604837   DR: 30.3092110008   OBJ: 11322.1845779   Grad: 0.000847139665319,0.000780326806134,0.000914748742523  It: 66.0666666667,61.6666666667,75.6666666667  Conv: 1.0,1.0,1.0
2015-05-28 22:12:16.671884 ### It: 6    PR: 8.24513295978   DR: 26.0573667465   OBJ: 10310.1176802   Grad: 0.000887377071897,0.000839421603416,0.000966897232257  It: 79.9666666667,74.3333333333,84.3333333333  Conv: 1.0,1.0,1.0
2015-05-28 22:12:32.161944 ### It: 7    PR: 7.27800303619   DR: 22.9979903758   OBJ: 9576.7800098    Grad: 0.000882209840757,0.000842467701669,0.000951129805512  It: 79.2666666667,73.3333333333,82.6666666667  Conv: 1.0,1.0,1.0
2015-05-28 22:12:46.877227 ### It: 8    PR: 6.54484165518   DR: 20.6789862671   OBJ: 9012.98631509   Grad: 0.000885117695984,0.000802837652469,0.000947572981701  It: 74.6,71.3333333333,77.6666666667  Conv: 1.0,1.0,1.0
2015-05-28 22:13:01.799061 ### It: 9    PR: 5.96746817939   DR: 18.8530662415   OBJ: 8561.38265583   Grad: 0.000842865646004,0.00076602879848,0.000898677929579  It: 67.7666666667,64.6666666667,72.0  Conv: 1.0,1.0,1.0
2015-05-28 22:13:16.539918 ### It: 10   PR: 5.49939059068   DR: 17.3731359204   OBJ: 8188.60147372   Grad: 0.000849797383444,0.000794274437477,0.000937924011613  It: 61.0,55.6666666667,65.0  Conv: 1.0,1.0,1.0
2015-05-28 22:13:33.865188 ### It: 11   PR: 5.11116090052   DR: 16.1460002895   OBJ: 7873.73837729   Grad: 0.000868477312735,0.000783938054373,0.000954891593157  It: 55.7333333333,52.6666666667,58.0  Conv: 1.0,1.0,1.0
2015-05-28 22:13:48.232758 ### It: 12   PR: 4.78321659468   DR: 15.1097259022   OBJ: 7602.93961338   Grad: 0.000838211239437,0.000711286540346,0.000910017961521  It: 53.1666666667,51.3333333333,57.0  Conv: 1.0,1.0,1.0
2015-05-28 22:14:00.694784 ### It: 13   PR: 4.50193892551   DR: 14.2211819555   OBJ: 7366.60771306   Grad: 0.00083252024587,0.000743679786528,0.000945067391604  It: 49.8666666667,46.6666666667,53.0  Conv: 1.0,1.0,1.0
2015-05-28 22:14:13.542489 ### It: 14   PR: 4.25765787488   DR: 13.4497347311   OBJ: 7157.85602129   Grad: 0.000786202839607,0.000662689932105,0.000874661867507  It: 44.1,40.0,48.3333333333  Conv: 1.0,1.0,1.0
2015-05-28 22:14:24.365103 ### It: 15   PR: 4.04317636768   DR: 12.7725775219   OBJ: 6971.59247144   Grad: 0.000780083313986,0.00066152037482,0.000968049596779  It: 40.5,37.3333333333,45.3333333333  Conv: 1.0,1.0,1.0
2015-05-28 22:14:35.572262 ### It: 16   PR: 3.85314614421   DR: 12.172759012    OBJ: 6803.96173598   Grad: 0.000770770527518,0.000694359302455,0.000934059205192  It: 36.4,30.3333333333,41.0  Conv: 1.0,1.0,1.0
2015-05-28 22:14:47.340098 ### It: 17   PR: 3.68342737503   DR: 11.6371571273   OBJ: 6651.98038654   Grad: 0.000724166742016,0.000560356749995,0.000931962911843  It: 31.8666666667,27.0,35.6666666667  Conv: 1.0,1.0,1.0
2015-05-28 22:14:56.210829 ### It: 18   PR: 3.53075584211   DR: 11.1554260699   OBJ: 6513.29894938   Grad: 0.000704864704774,0.000587964072036,0.000817911630703  It: 30.6333333333,25.6666666667,34.3333333333  Conv: 1.0,1.0,1.0
2015-05-28 22:15:04.743024 ### It: 19   PR: 3.39256281916   DR: 10.7194255276   OBJ: 6386.03902306   Grad: 0.000705672889658,0.000470960892426,0.000843435909192  It: 28.0666666667,25.0,30.0  Conv: 1.0,1.0,1.0
2015-05-28 22:15:11.839042 ### It: 20   PR: 3.26689199064   DR: 10.3229575443   OBJ: 6268.67737282   Grad: 0.000667401702067,0.000579652459082,0.000797838398045  It: 26.8,23.3333333333,33.3333333333  Conv: 1.0,1.0,1.0
2015-05-28 22:15:11.839222 ### Saving Z to file regression_output...
2015-05-28 22:15:14.659391 ### ... done saving to file regression_output.
```


## Description of Abstract Framework ##

The present implementation can be used to solve arbitrary problems of the following form:


  > min_{Z} \sum_{i=1}^N F(Z; d_i) + lam * ||Z||_1    

where `Z` in R^d, `F` is a convex function in `Z`, `d_i` are data points read from a file, `lam` is a regularization parameter, and `||.||_1` is the usual l1 norm. For example, in the case of logistic regression, d_i can be datapoints of the form `(x_i,y_i)`, where x_i in R^d and y_i in `{-1,+1}`, and F is the logistic loss `log(1+exp(-y_i* x_i^T * Z))`.

To use the present code to solve an arbitrary problem of the above form, a programmer needs to extend the abstract class `AbstractSolver`. In particular, a programmer needs to specify three functions:

##### Function `readPointBatch`: #####

This is  an abstract method receiving input data from a partition, and creating a single batch object. 
		
The SparkADMM code will read data line by line from a file, partition it across machines, and create python objects storing 
the data in each machine. An implementation of this abstract method specifies how this data should be implemented.
This method should receive as input an 

```iterator<string> ```
 
e.g., a list of strings. Each string corresponds to a line in the file storing the data; each such line may 
represent a different  datapoint d_i. It should then return a 3-tuple 
of the following form: 
		
```
(data,keys,stats) 
```

where: 
		
* `data` is a collection (e.g., list, numpy array, pandas dataframe, etc.) representing the data `d_i` read. It is up to the 
  implementation to determine how to best represent the data; whatever the representation is, this ought to be used by the
		  other two functions in the code.

* `keys` is a list of keys (e.g., strings) representing which global variables are used by this partition.

* `stats` is a (possibly empty) dictionary of `key:numerical_value` pairs, that contains some statistics about the data in this partition.
  For example, the number of lines read, the number of global variables, the processing time, etc., 
                  can be reported in this dictionary. It is up to the developer to determine which statistics to report, if any. SparkADMM
		  will print the mean, max, and mean values of all these statistics across partitions.

##### Function `localObjective`: #####

This is an abstract method receiving input data_i from partition i, and global variables, 
and returns the local evaluation of the objective over the local data.

The input of the function is
* `data`: a data object (e.g., list, numpy array, pandas dataframe, etc.) representing the data in this partition. 
  This is of the same form as the data returned by the readPointBatch function.
* `Z`: a dictionary storing variables as key:numerical_value pairs. The dictionary will only contain values for keys in the 'keys' list 
  returned by the readPointBatch function for this partition

		
The output of the function is a scalar value, equal to 
		
```\sum_{d_i\in data} F(Z;d_i)```

i.e., the value of the objective evaluated at this partition's data.  Recall that ADMM attempts to solve 
		
```\sum_{k=1}^K \sum_{i\in data_k} F(Z;d_i) + lam * || Z ||_1```

so SparkADMM uses this function to evaluate and keep track of the global objective.

##### Function `SolveProximal` #####

This local solver, that minimizes the objective locally, penalized by a proximal term. This implements the update of local variables in the ADMM algorithm.

The input to the function is
* `data`:  a data object (e.g., list, numpy array, pandas dataframe, etc.) representing the collection of data `d_i` in this partition. 
		  This is of the same form as the data returned by the readPointBatch function.

* `rho`: a scalar value (e.g., 0.1)

* `master_Z`: a dictionary storing variables as key:numerical_value pairs. The dictionary can be assumed to contain values 
  only for keys in the 'keys' list returned by the readPointBatch function for this partition.




The function should return a pair
		
```
(Z,stats)
```
		
where `Z` is dictionary of the form { key:numerical_value,...},  obtained as the solution of the following
minimization problem:

``` Z = argmin_{Z} \sum_{d_i \in data} F(Z;d_i) + rho *\| Z - master_Z   \|_2^2 ```

and `stats` is  a (possibly empty) dictionary of the form {statistic_label:numerical_value} containing statistics that the 
programmer wishes to report (e.g., the time to completion, the number of iterations if the minimization was done by an iterative algorithm, whether the algorithm converged, etc.) It is up to the programmer to determine what statistics she wishes to report.


## Useful Links ##

* [ADMM paper by Boyd et al.](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html) The code here implements the generalized consensus algorithm in Chapter 7 of this paper.
* [Spark documentation](https://spark.apache.org/docs/latest/)
* [Hadoop documentation](https://hadoop.apache.org/)
