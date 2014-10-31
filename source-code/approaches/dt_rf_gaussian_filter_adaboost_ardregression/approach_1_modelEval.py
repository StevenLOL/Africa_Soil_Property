from __future__ import division
__author__ = 'steven'

'''
test with regressor with 10 fold cross validation and save predition result

First create a list or regressor, then eval with 10 fold 

Then predit result and save in the format of

regressor_name_cv_score.csv

Program entry point is @ line 140+  where if __name__ == '__main__':
'''

import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
import pandas as pd
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn import isotonic
from sklearn import neighbors
from sklearn import cross_decomposition
import math
import sys
import datetime
from sklearn import grid_search

def save_predictTion(iclf,train_x,train_target,test_x,saveTo):
    #traing model and save result
    print datetime.datetime.now(),'training on all data'
    #as we have five target
    n_task_d=5;
    preds = np.zeros((test_x.shape[0], n_task_d))
    
    #predict one by one 
    for i in range(n_task_d):
      iclf.fit(train_x,train_target[:,i]);
      preds[:,i]=iclf.predict(test_x).astype(float)
    #load submission template and fill in data then save
    sample = pd.read_csv('../../data/sample_submission.csv')
    sample['Ca'] = preds[:,0]
    sample['P'] = preds[:,1]
    sample['pH'] = preds[:,2]
    sample['SOC'] = preds[:,3]
    sample['Sand'] = preds[:,4]
    sample.to_csv(saveTo, index = False)
    print datetime.datetime.now(),'training on all data done'


def load():
    #loading data
    train = pd.read_csv('../../data/training.csv')
    test = pd.read_csv('../../data/sorted_test.csv')
    labels = train[['Ca','P','pH','SOC','Sand']].values
    co2list=['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76']
    landlist=['BSAN', 'BSAS', 'BSAV', 'CTI', 'ELEV', 'EVI', 'LSTD', 'LSTN', 'REF1', 'REF2', 'REF3', 'REF7', 'RELI', 'TMAP', 'TMFI']
    train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN','Depth'], axis=1, inplace=True)
    #drop some features
    #train.drop(co2list,axis=1, inplace=True)
    train.drop(landlist,axis=1, inplace=True)
    #train["Depth"] = train["Depth"].apply(lambda depth:0 if depth =="Subsoil" else 1)
    #train['Depth2']=abs(train['Depth']-1)
    test.drop(['PIDN','Depth'], axis=1, inplace=True)
    #test.drop(co2list,axis=1, inplace=True)
    test.drop(landlist,axis=1, inplace=True)
    #test["Depth"] = test["Depth"].apply(lambda depth:0 if depth =="Subsoil" else 1)
    #test['Depth2']=abs(test['Depth']-1)
    #print np.array(train)[:,3578]
    #xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]
    xtrain, xtest = np.array(train), np.array(test)
    print xtrain.shape,xtest.shape
    return xtrain,labels,xtest


def grid_search_svr(X,y,test):
    #search best parameter for SVR
    parameters = {'kernel':('rbf','poly','sigmoid'), 'C':[1, 10,100,1000,10000],'degree':[1,2,3,4],'epsilon':[0.1,0.01,0.001]}
    #parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    n_task_d=5
    for i in range(n_task_d):
      svr = svm.SVR(verbose=1)
      clf = grid_search.GridSearchCV(svr, parameters,n_jobs=-1)
      clf.fit(X,y[:,i])
      print clf
      print clf.best_estimator_
      print clf.best_score_
      print clf.best_params_
      break


#? GridSearch on Ridge?
def model_eval(X,y,test):
    n_folds = 10
    n_task_d=5;
    clfs = [#svm.SVR(C=10000.0), #best result
            ##linear_model.LogisticRegression(), #too slow
            #linear_model.LinearRegression(),  
            #linear_model.Ridge(alpha=.5),
            #linear_model.BayesianRidge(verbose=1),
            #linear_model.Lasso(alpha = 0.1),
            #linear_model.ElasticNet(),
            ##linear_model.SGDRegressor(alpha=100000,n_iter=100,learning_rate='optimal',verbose=1),
            #linear_model.Lars(verbose=1),
            #linear_model.LassoLars(alpha=.1,verbose=1),
            ##linear_model.ARDRegression(verbose=1),
            ##linear_model.RANSACRegressor(),   #http://scikit-learn.org/stable/modules/linear_model.html#robustness-to-outliers-ransac
            ##linear_model.RANSACRegressor(linear_model.LinearRegression()), #http://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#example-linear-model-plot-ransac-py
            #linear_model.PassiveAggressiveRegressor(verbose=1),
            ##isotonic.IsotonicRegression(),
            #neighbors.KNeighborsRegressor(),
            ##neighbors.RadiusNeighborsRegressor(),
            #cross_decomposition.PLSRegression(),
               #svr default 'rbf' kernel
            #svm.SVR(C=10000.0,kernel='poly',verbose=1),
            #svm.SVR(C=10000.0,kernel='linear',verbose=1),
            #svm.SVR(C=0.01,kernel='sigmoid',verbose=1)
            #DecisionTreeRegressor(),
            ExtraTreesRegressor(n_estimators=100),  #3000 give the best
            RandomForestRegressor(n_estimators=100),
            GradientBoostingRegressor(n_estimators=100),
            AdaBoostRegressor(n_estimators=100) #http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#example-ensemble-plot-adaboost-regression-py
            ]
    clfsMultiTask=[];

    returnResult=np.zeros((len(clfs),n_task_d));
    for j, clf in enumerate(clfs):
        try:
            print datetime.datetime.now(),j, clf
            np.random.seed(0) # seed to shuffle the train set
            for i in range(n_task_d):
              scores = cross_validation.cross_val_score(clf, X, y[:,i],n_jobs=-1,cv=n_folds,scoring='mean_squared_error')
              avgScore=math.sqrt(abs(np.sum(scores)/n_folds))
              print datetime.datetime.now(),'crv=', avgScore,scores
              returnResult[j,i]=avgScore;
            np.savetxt("modelEval_cv_"+clf.__class__.__name__+'%d.csv'%(X.shape[1]),np.hstack((returnResult[j,:],np.mean(returnResult[j,:]))), delimiter=",",fmt='%f')
            save_predictTion(clf,X,y,test,'modelEval_submit_'+clf.__class__.__name__+'%d.csv'%(X.shape[1]));
        except:
            print datetime.datetime.now() #,"Unexpected error:",sys.exec_info()[0]
            #otf=open('error.log','a');
            #otf.write("%d,%s\nUnexpected error: %s\n"%(j,clf.__class__.__name__,sys.exec_info()[0]))
            #otf.close();




    return returnResult;


if __name__ == '__main__':
    train_x, train_target, test_x = load()
    #find the best SVR parameters
    #grid_search_svr(train_x,train_target,test_x);
    #test list of regressors
    cv =model_eval(train_x,train_target,test_x);
    print cv;
    np.savetxt("modelEval_cv_%d.csv"%(train_x.shape[1]), cv, delimiter=",",fmt='%f')





