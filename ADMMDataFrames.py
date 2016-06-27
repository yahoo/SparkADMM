# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

import numpy as np
import pandas

class SparseDataFrame(pandas.DataFrame):
	"""when A and B have different dimension, they can be added or multiplied, assuming 0 when missing"""
	def __add__(self,other):
		return SparseDataFrame(super(SparseDataFrame,self).add(other,fill_value=0.).fillna(0.))
	def __radd__(self,other):   
		"""reverse add, needed for sum()"""
		return self + other
	def __sub__(self,other):
		"""-other is the super version of unary -"""
		return self + (-other)  
	def add(self, other):
		return self + other
	def dot(self,other):
		return SparseDataFrame(super(SparseDataFrame,self).dot(other.loc[self.columns].fillna(0.)))
	def __mul__(self, other):
		return SparseDataFrame(super(SparseDataFrame,self).__mul__(other))
	def __rmul__(self, other):
		return self * other
	def concur(self,other):
		# expand self to accommodate additional index/columns of other, and fill the missing values with those of other
		tmp = other.copy() + self * 0.0
		tmp.loc[self.index,self.columns] = self * 0.0
		return self + tmp

def reconstructDataFrame(unravelled):
	d = {}
	for (row,column),value in unravelled.iteritems():
		if column in d:
			d[column][row] = value
		else:
			d[column] = { row:value }
	return pandas.DataFrame.from_dict(d)

def unravelDataFrame(df):
	return dict([  ((row,column), value2 )   for column,value1  in df.to_dict().iteritems() for row,value2 in value1.iteritems()])


