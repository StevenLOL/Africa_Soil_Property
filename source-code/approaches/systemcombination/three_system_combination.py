
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.

import pandas as pd
import numpy as np
from sklearn import svm

result1 = pd.read_csv('./exp/15nn_withoutCO2.csv') #0.67
result2 = pd.read_csv('./exp/svr_withoutCO2.csv') #0.41
result3 = pd.read_csv('./exp/1stderivate.csv') #0.57

preds = np.zeros((result1.shape[0], 5))

sample = pd.read_csv('../data/sample_submission.csv')
sample['Ca'] = (result1['Ca']*0.33 + result2['Ca']*0.59 + result3['Ca']*0.43)/(0.33+0.59+0.43)
sample['P'] = (result1['P']*0.33 + result2['P']*0.59 + result3['P']*0.43)/(0.33+0.59+0.43)
sample['pH'] = (result1['pH']*0.33 + result2['pH']*0.59 + result3['pH']*0.43)/(0.33+0.59+0.43)
sample['SOC'] = (result1['SOC']*0.33 + result2['SOC']*0.59 + result3['SOC']*0.43)/(0.33+0.59+0.43)
sample['Sand'] = (result1['Sand']*0.33 + result2['Sand']*0.59 + result3['Sand']*0.43)/(0.33+0.59+0.43)

	sample.to_csv(output_file, index = False)

if __name__ == '__main__':
	import sys
	if len(sys.argv) < 6:
		print "Syntax: python ./two_system_combination.py <first-approach-result-file> <first-error-rate> <second-approach-result-file> <second-error-rate> <output-file>"		
		print "Example: python ./two_system_combination.py ./submission/15nn.csv 0.67 ./submission/svr.csv 0.40 ./submission/combination_result.csv"
		exit()
	two_combination(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], sys.argv[5])
	
