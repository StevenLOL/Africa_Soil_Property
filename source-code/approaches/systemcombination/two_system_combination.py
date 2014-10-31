#!/usr/bin/env python
# coding: utf-8
# @author: ngaht


import pandas as pd

def two_combination(first_app_file, first_err, second_app_file, second_err, output_file):
	result1 = pd.read_csv(first_app_file)
	result2 = pd.read_csv(second_app_file)
	rate1 = float(1 - float(first_err))
	rate2 = float(1 - float(second_err))


	sample = pd.read_csv('../../data/sample_submission.csv')

	sample['Ca'] = (result1['Ca']*rate1 + result2['Ca']*rate2)/(rate1+rate2)
	sample['P'] = (result1['P']*rate1 + result2['P']*rate2)/(rate1+rate2)
	sample['pH'] = (result1['pH']*rate1 + result2['pH']*rate2)/(rate1+rate2)
	sample['SOC'] = (result1['SOC']*rate1 + result2['SOC']*rate2)/(rate1+rate2)
	sample['Sand'] = (result1['Sand']*rate1 + result2['Sand']*rate2)/(rate1+rate2)

	sample.to_csv(output_file, index = False)

if __name__ == '__main__':
	import sys
	if len(sys.argv) < 6:
		print "Syntax: python ./two_system_combination.py <first-approach-result-file> <first-error-rate> <second-approach-result-file> <second-error-rate> <output-file>"		
		print "Example: python ./two_system_combination.py ./submission/15nn.csv 0.67 ./submission/svr.csv 0.40 ./submission/combination_result.csv"
		exit()
	two_combination(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], sys.argv[5])
	
