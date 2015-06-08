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
'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# drop ids and get labels
#labels = train.target.values # this gives a ndarray instead of DataFrame #
labels = train['target']
train = train.drop('id', axis=1)
X = train.drop('target', axis=1)
test = test.drop('id', axis=1)

unlabeled_X = np.concatenate((X, test), axis=0)

# encode the class labels 
encoder = LabelEncoder()
train_y = encoder.fit_transform(labels).astype(np.int32)

train_X = X.values
test_X = test.values


# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
tfidf = tfidf.fit(unlabeled_X)
tfidf_train_X = tfidf.transform(train_X).toarray()
tfidf_test_X = tfidf.transform(test_X).toarray()
'''

################################################################################

train = pd.read_csv('tfidf_lda_18features_train.csv')
test = pd.read_csv('tfidf_lda_18features_test.csv')
sample = pd.read_csv('sampleSubmission.csv')

train_y = train.values[:, -1].astype(np.int32)
train_X = train.values[:, :-1]

test_X = test.values

################################################################################



## list of prediction results from different classifiers 
predictions = []


#############################
##  XGBoost Classifier
#############################

test_X = np.nan_to_num(test_X)

xgb_param = param.xgboost_para()
xgboost_clf = clf.xgboost_classifer(xgb_param, train_X, train_y);

test_dummy_y = np.ndarray(shape=(test_X.shape[0],),dtype=float)
## xgboost input data requires specifc format
xg_test = xgb.DMatrix(test_X)
## prediction from xgboost
y_prob = xgboost_clf.predict(xg_test)

predictions.append(y_prob)


#############################
## Lasagne NN Classifier
#############################

# standardize the data for NN
unlabeled_X = np.concatenate((train_X, test_X), axis=0)
scaler = StandardScaler()
scaler.fit(unlabeled_X)
scaled_train_X = scaler.transform(train_X)
scaled_test_X = scaler.transform(test_X)


lasagne_oneLayer_param = param.lasagne_singleLayer_para()
lasagne_twoLayer_param = param.lasagne_doubleLayer_para()

lasagne_oneLayer_param['num_classes'] = lasagne_twoLayer_param['num_classes'] = 9
lasagne_oneLayer_param['num_features'] = lasagne_twoLayer_param['num_features'] = train_X.shape[1]

lasagne_oneLayer_num = 10
#lasagne_oneLayer_num = 2
lasagne_oneLayer_result = []

lasagne_twoLayer_num = 15
#lasagne_twoLayer_num = 2
lasagne_twoLayer_result = []

#############################

for i in range(lasagne_oneLayer_num):
	oneLayer_lasagne_clf = clf.lasagne_oneLayer_classifier(lasagne_oneLayer_param, scaled_train_X, train_y)
	y_prob = oneLayer_lasagne_clf.predict_proba(scaled_test_X)
	if (i==0):
		lasagne_oneLayer_result = y_prob
	else:
		lasagne_oneLayer_result = np.add(lasagne_oneLayer_result, y_prob)	

lasagne_oneLayer_result = np.divide(lasagne_oneLayer_result, float(lasagne_oneLayer_num))
predictions.append(lasagne_oneLayer_result)

#############################

for i in range(lasagne_twoLayer_num):
	twoLayer_lasagne_clf = clf.lasagne_twoLayer_classifier(lasagne_twoLayer_param, scaled_train_X, train_y)
	y_prob = twoLayer_lasagne_clf.predict_proba(scaled_test_X)
	if (i==0):
		lasagne_twoLayer_result = y_prob
	else:
		lasagne_twoLayer_result = np.add(lasagne_twoLayer_result, y_prob)	

lasagne_twoLayer_result = np.divide(lasagne_twoLayer_result, float(lasagne_twoLayer_num))
predictions.append(lasagne_twoLayer_result)



################################################################################


weights = [0.48, 0.047, 0.473]
weighted_predictions = [weight*prediction for weight, prediction in zip(weights,predictions)]
final_prediction = sum(weighted_predictions)

# create submission file
preds = pd.DataFrame(final_prediction, columns=sample.columns[1:])
preds.index += 1 
preds.to_csv('final_combined_results.csv', index_label='id')


