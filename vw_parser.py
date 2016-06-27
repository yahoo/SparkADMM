# Copyright 2016 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

import sys,re,math,logging
import json
allfeats = set()
for line in sys.stdin:
	try:
		tmp = line.strip('\r\t\n ').split('|')
		target = float(tmp[0].split(' ')[0])
		id = tmp[0].split(' ')[2]
		features = {}
		for t in tmp[1:]:
			tmp0 = t.split(' ')
			s = tmp0[0] 
			
			for u in tmp0[1:-1]:
				feat= '%s^%s'%(s,u.split(':')[0])
				allfeats.add(feat)
				features[feat] = float(u.split(':')[1]) 
		print '\t'.join(map(str, [id, json.dumps(features), json.dumps({'output':target})]))
		
	except:
		continue
logging.warning('Total number of features = %d'%len(allfeats))
