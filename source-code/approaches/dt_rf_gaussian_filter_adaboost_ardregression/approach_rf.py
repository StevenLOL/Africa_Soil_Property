
'''
84	new	 Steven Du	
0.40861
6	 Tue, 30 Sep 2014 16:10:42
Your Best Entry
You improved on your best score by 0.02760. 
'''
import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestRegressor
rng = np.random.RandomState(1)
train = pd.read_csv('../../data/training.csv')
test = pd.read_csv('../../data/sorted_test.csv')
labels = train[['Ca','P','pH','SOC','Sand']].values
droplist=['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76']
train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)
train.drop(droplist,axis=1, inplace=True)
test.drop(droplist,axis=1, inplace=True)

xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]


#sup_vec = svm.SVR(C=10000.0, verbose = 2)

#rf=RandomForestRegressor(n_estimators=100,n_jobs=8,verbose=2, random_state=rng)
rf=RandomForestRegressor(n_estimators=100,n_jobs=8,verbose=2, random_state=rng)
preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    
    rf.fit(xtrain, labels[:,i])
    preds[:,i] = rf.predict(xtest).astype(float)

sample = pd.read_csv('../../data/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('beating_benchmark_sp_rf100_j8_outloop.csv', index = False)


