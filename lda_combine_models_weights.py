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

'''
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# drop ids and get labels
#labels = train.target.values # this gives a ndarray instead of DataFrame #
labels = train['target']
train = train.drop('id', axis=1)
X = train.drop('target', axis=1)
test = test.drop('id', axis=1)

unlabeled_X = np.concatenate((X, test), axis=0)


# encode the class labels 
encoder = LabelEncoder()
y = encoder.fit_transform(labels).astype(np.int32)

### we need a test set that we didn't train on to find the best weights for combining the classifiers
sss = StratifiedShuffleSplit(y, 1, test_size=0.1, random_state=1234)

for train_index, test_index in sss:
    print 'split the training data'
#print np.amax(train_index, axis=1)

train_X, train_y = X.values[train_index], y[train_index]
test_X, test_y = X.values[test_index], y[test_index]


# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
tfidf = tfidf.fit(unlabeled_X)
tfidf_train_X = tfidf.transform(train_X).toarray()
tfidf_test_X = tfidf.transform(test_X).toarray()
'''

################################################################################


train = pd.read_csv('tfidf_lda_18features_train.csv')

labels = train.values[:, -1].astype(np.int32)
train = train.values[:, :-1]


### we need a test set that we didn't train on to find the best weights for combining the classifiers
sss = StratifiedShuffleSplit(labels, 1, test_size=0.1, random_state=1234)

for train_index, test_index in sss:
    print 'split the training data'

train_X, train_y = train[train_index], labels[train_index]
test_X, test_y = train[test_index], labels[test_index]

################################################################################


## list of prediction results from different classifiers 
predictions = []


#############################
##  XGBoost Classifier
#############################

xgb_param = param.xgboost_para()

xgboost_clf = clf.xgboost_classifer(xgb_param, train_X, train_y);

## xgboost input data requires specifc format
xg_test = xgb.DMatrix(test_X, label=test_y)
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
lasagne_oneLayer_result = []

lasagne_twoLayer_num = 10
lasagne_twoLayer_result = []


for i in range(lasagne_oneLayer_num):
	oneLayer_lasagne_clf = clf.lasagne_oneLayer_classifier(lasagne_oneLayer_param, scaled_train_X, train_y)
	y_prob = oneLayer_lasagne_clf.predict_proba(scaled_test_X)
	if (i==0):
		lasagne_oneLayer_result = y_prob
	else:
		lasagne_oneLayer_result = np.add(lasagne_oneLayer_result, y_prob)	

lasagne_oneLayer_result = np.divide(lasagne_oneLayer_result, float(lasagne_oneLayer_num))
predictions.append(lasagne_oneLayer_result)


for i in range(lasagne_twoLayer_num):
	twoLayer_lasagne_clf = clf.lasagne_twoLayer_classifier(lasagne_twoLayer_param, scaled_train_X, train_y)
	y_prob = twoLayer_lasagne_clf.predict_proba(scaled_test_X)
	if (i==0):
		lasagne_twoLayer_result = y_prob
	else:
		lasagne_twoLayer_result = np.add(lasagne_twoLayer_result, y_prob)	

lasagne_twoLayer_result = np.divide(lasagne_twoLayer_result, float(lasagne_twoLayer_num))
predictions.append(lasagne_twoLayer_result)



'''
oneLayer_lasagne_clf = clf.lasagne_oneLayer_classifier(lasagne_oneLayer_param, scaled_train_X, train_y)
twoLayer_lasagne_clf = clf.lasagne_twoLayer_classifier(lasagne_twoLayer_param, scaled_train_X, train_y)

y_prob = oneLayer_lasagne_clf.predict_proba(scaled_test_X)
predictions.append(y_prob)

y_prob = twoLayer_lasagne_clf.predict_proba(scaled_test_X)
predictions.append(y_prob)
'''

#lasagne_result = clf.lasagne_classifier(lasagne_param, train_X, train_Y, test_X);

#predictions.append(lasagne_result.tolist())
#predictions.append(y_prob)



#############################

## function to restimate the weights for each classifier
def log_loss_func(weights):
    ### scipy minimize will pass the weights as a numpy array ###
	final_prediction = 0
	weighted_predictions = [weight*prediction for weight, prediction in zip(weights,predictions)]
	final_prediction = sum(weighted_predictions)
	return log_loss(test_y, final_prediction)

#the algorithms need a starting value, right not we chose 0.5 for all weights
#its better to choose many random starting points and run minimize a few times
starting_values = [0.5]*len(predictions)

#adding constraints and a different solver
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})

#our weights are bound between 0 and 1
bounds = [(0,1)]*len(predictions)

res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
print('Best Weights: {weights}'.format(weights=res['x']))

