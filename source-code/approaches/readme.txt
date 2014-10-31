Note that the path to data may be need to change directly in the source code
1) to run knn
$ python ./kNN.py <value-of-k> <with-CO2> <output-file-name>

	where: with-CO2 = 1 mean analize using all the spectra features, otherwise, mean not using CO2 features
		out-put-file-name: example: 5nn.csv

2) to run svr
python ./svr.py
- The output will be in exp folder, named svr_withoutCO2.csv
3) system combination
- the Combination of 15nn and svr
$ python system_combination_15nn_svr.py 

- The combination of 15nn, svr and the first derivate (from Steven)
$ python system_combination_15nn_svr_1stderivate.py 

All the result will be output to exp folder
