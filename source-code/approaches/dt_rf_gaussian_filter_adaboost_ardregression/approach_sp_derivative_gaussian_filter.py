
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
from scipy.ndimage import filters as flt
from sklearn import preprocessing
train = pd.read_csv('../../data/training.csv')
test = pd.read_csv('../../data/sorted_test.csv')
labels = train[['Ca','P','pH','SOC','Sand']].values
droplist=['m2379.76','m2377.83','m2375.9','m2373.97','m2372.04','m2370.11','m2368.18','m2366.26','m2364.33','m2362.4','m2360.47','m2358.54','m2356.61','m2354.68','m2352.76']
train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)
train.drop(droplist,axis=1, inplace=True)
test.drop(droplist,axis=1, inplace=True)


train_feature_list = list(train.columns)
spectra_features = train_feature_list
non_spectra_feats=['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth']

#can also drop all by drop a list object.
for feats in non_spectra_feats:
     spectra_features.remove(feats)
#train[spectra_features].to_csv('before_gf.csv', index = False)
sigma=20
order=2
#Refer maveric @ http://www.kaggle.com/c/afsis-soil-properties/forums/t/10184/first-derivative/53511
fltSpectra=flt.gaussian_filter1d(np.array(train[spectra_features]),sigma=sigma,order=order)
#xtrain["Depth"] = x_train["Depth"].apply(lambda depth:0 if depth =="Subsoil" else 1)
#scaler = preprocessing.StandardScaler().fit(fltSpectra)
scaler = preprocessing.MinMaxScaler().fit(fltSpectra)

train[spectra_features]=scaler.transform(fltSpectra)
#train[spectra_features].to_csv('after_gf.csv', index = False)
fltSpectra2=flt.gaussian_filter1d(np.array(test[spectra_features]),sigma=sigma,order=order)
test[spectra_features]=scaler.transform(fltSpectra2)


xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]


sup_vec = svm.SVR(C=10000.0, verbose = 2)

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    sup_vec.fit(xtrain, labels[:,i])
    preds[:,i] = sup_vec.predict(xtest).astype(float)

sample = pd.read_csv('../../data/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('beating_benchmark_sp_gaussian_filter_minmax_s%d_o%d.csv'%(sigma,order), index = False)


