

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
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
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

def cvtestNone(xtrain,labels,tid):
    #test DecisionTreeRegressor with depth = None
    cvresult=np.zeros(7);
    for i in range(1):
        clf_1 = DecisionTreeRegressor(max_depth=None,random_state=rng,min_samples_leaf=2)
        cv = cross_validation.ShuffleSplit(xtrain.shape[0], n_iter=10,test_size=0.1, random_state=rng)
        scores = cross_validation.cross_val_score(clf_1, xtrain, labels[:,tid],cv=cv,scoring='mean_squared_error')
        cvresult[i-1]=np.sum(scores)/10
        print tid,'depth=None',np.sum(scores)/10,scores
        print clf_1.get_params()



def cvtest(xtrain,labels,tid):
    #find the best depth of DecisionTreeRegressor
    cvresult=np.zeros(7);
    for i in range(1,8):
        clf_1 = DecisionTreeRegressor(max_depth=i,random_state=rng)
        cv = cross_validation.ShuffleSplit(xtrain.shape[0], n_iter=10,test_size=0.1, random_state=rng)
        scores = cross_validation.cross_val_score(clf_1, xtrain, labels[:,tid],cv=cv,scoring='mean_squared_error',n_jobs=-1)
        cvresult[i-1]=np.sum(scores)/10
        print tid,'depth=,',i,np.sum(scores)/10,scores
    return np.argmin(cvresult)+1;




preds = np.zeros((xtest.shape[0], 5))
rmse=list();
#through cvtestNone and cvtest we found the best depth for each target. 
bestDepth=[6,2,3,5,6];
for i in range(5):
    #cvtestNone(xtrain,labels,i)
    #bestDepth=cvtest(xtrain,labels,i)    #given test given bestDepth=[6,2,3,5,6];
    print 'best=',bestDepth
    clf_1 = DecisionTreeRegressor(max_depth=bestDepth[i],random_state=rng)
    
    #cv = cross_validation.ShuffleSplit(xtrain.shape[0], n_iter=10,test_size=0.1, random_state=rng)
    #scores = cross_validation.cross_val_score(clf_1, xtrain, labels[:,i],cv=cv,scoring='mean_squared_error')
    #print i,'depth=4,',np.sum(scores)/10,scores
    #rmse.append(np.sum(scores)/10);

    clf_1.fit(xtrain, labels[:,i])
    preds[:,i] = clf_1.predict(xtest).astype(float)
#print rmse
sample = pd.read_csv('../../data/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('beating_benchmark_dt_none.csv', index = False)


