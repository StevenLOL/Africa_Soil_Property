#!/usr/bin/env python

# coding: utf-8
# @author: ngaht


import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import neighbors

def knn(k, dropCO2, output):
	train = pd.read_csv('../../data/training.csv')
	test = pd.read_csv('../../data/sorted_test.csv')
	labels = train[['Ca','P','pH','SOC','Sand']].values
	train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
	test.drop('PIDN', axis=1, inplace=True)
	if dropCO2 == 1:
		train.drop(['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76'], axis=1, inplace=True)
		test.drop(['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76'], axis=1, inplace=True)


	xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]

	knn = neighbors.KNeighborsRegressor(int(k), weights='distance')

	preds = np.zeros((xtest.shape[0], 5))
	for i in range(5):
	    knn.fit(xtrain, labels[:,i])
	    preds[:,i] =  knn.predict(xtest).astype(float)

	sample = pd.read_csv('../data/sample_submission.csv')
	sample['Ca'] = preds[:,0]
	sample['P'] = preds[:,1]
	sample['pH'] = preds[:,2]
	sample['SOC'] = preds[:,3]
	sample['Sand'] = preds[:,4]

	sample.to_csv("./exp/"+output, index = False)

if __name__=="__main__":
	import sys
	if len(sys.argv) < 4:
		print "\tSyxtax: python ./kNN.py <value-of-k> <drop-CO2> <output-file-name>\n\t\t drop-CO2 =1 or drop-CO2=0"
		exit()
	knn(sys.argv[1], sys.argv[2], sys.argv[3])

