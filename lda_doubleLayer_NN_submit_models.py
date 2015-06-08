import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, feature_extraction, preprocessing

import lda
import xgboost as xgb
import classifier as clf
import parameters as param


################################################################################

################################################################################

train = pd.read_csv('tfidf_lda_18features_train.csv')
test = pd.read_csv('tfidf_lda_18features_test.csv')
sample = pd.read_csv('sampleSubmission.csv')

train_y = train.values[:, -1].astype(np.int32)
train_X = train.values[:, :-1]

test_X = test.values
## convert NaN to zero
test_X = np.nan_to_num(test_X)

################################################################################



## list of prediction results from different classifiers 
#predictions = []


#############################
## Lasagne NN Classifier
#############################

# standardize the data for NN
unlabeled_X = np.concatenate((train_X, test_X), axis=0)
scaler = StandardScaler()
scaler.fit(unlabeled_X)
scaled_train_X = scaler.transform(train_X)
scaled_test_X = scaler.transform(test_X)


lasagne_twoLayer_param = param.lasagne_comp_doubleLayer_para()

lasagne_twoLayer_param['num_classes'] = 9
lasagne_twoLayer_param['num_features'] = train_X.shape[1]


lasagne_twoLayer_num = 15
#lasagne_twoLayer_num = 2
lasagne_twoLayer_result = []

#############################

for i in range(lasagne_twoLayer_num):
	twoLayer_lasagne_clf = clf.lasagne_twoLayer_classifier(lasagne_twoLayer_param, scaled_train_X, train_y)
	y_prob = twoLayer_lasagne_clf.predict_proba(scaled_test_X)
	if (i==0):
		lasagne_twoLayer_result = y_prob
	else:
		lasagne_twoLayer_result = np.add(lasagne_twoLayer_result, y_prob)	

lasagne_twoLayer_result = np.divide(lasagne_twoLayer_result, float(lasagne_twoLayer_num))


################################################################################


# create submission file
preds = pd.DataFrame(lasagne_twoLayer_result, columns=sample.columns[1:])
preds.index += 1 
preds.to_csv('final_doubleLayer_NN_results.csv', index_label='id')


