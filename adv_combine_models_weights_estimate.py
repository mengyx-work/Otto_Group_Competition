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
## test the LDA model
lda_model = lda.LDA(n_topics=9, n_iter=100, random_state=1)
lda_model.fit(unlabeled_X)
lda_train_X = lda_model.transform(train_X)
lda_test_X = lda_model.transform(test_X)

tfidf_train_X = np.concatenate((tfidf_train_X, lda_train_X), axis=1)
tfidf_test_X = np.concatenate((tfidf_test_X, lda_test_X), axis=1)
'''


## list of prediction results from different classifiers 
predictions = []


#############################
##  XGBoost Classifier
#############################

# setup parameters for xgboost
xgb_param = {}
# use softmax multi-class classification
xgb_param['objective'] = 'multi:softprob'
xgb_param['silent'] = 1
xgb_param['eval_metric'] = 'mlogloss'
xgb_param['nthread'] = 3
xgb_param['num_class'] = 9


## parameters can be optimized ## 
xgb_param['eta'] = 0.25
xgb_param['max_depth'] = 12
xgb_param['min_child_weight'] = 1
xgb_param['min_child_width'] = 1  #default 1
xgb_param['num_round'] = 50

xgb_param['subsample'] = 0.8  #default 1.
xgb_param['colsample_bytree'] = 1.  #default 1.
xgb_param['gamma'] = 0  #default 0


xgboost_clf = clf.xgboost_classifer(xgb_param, tfidf_train_X, train_y);

## xgboost input data requires specifc format
xg_test = xgb.DMatrix(tfidf_test_X, label=test_y)
## prediction from xgboost
y_prob = xgboost_clf.predict(xg_test)

predictions.append(y_prob)


#############################
## Lasagne NN Classifier
#############################


# standardize the data for NN

unlabeled_tfidf_X = np.concatenate((tfidf_train_X, tfidf_test_X), axis=0)

scaler = StandardScaler()
scaler.fit(unlabeled_tfidf_X)
train_X = scaler.transform(tfidf_train_X)
test_X = scaler.transform(tfidf_test_X)

lasagne_oneLayer_param = {}
lasagne_twoLayer_param = {}


lasagne_oneLayer_param['num_classes'] = lasagne_twoLayer_param['num_classes'] = len(encoder.classes_)
lasagne_oneLayer_param['num_features'] = lasagne_twoLayer_param['num_features'] = train_X.shape[1]


lasagne_oneLayer_param['max_epochs'] = 40
lasagne_oneLayer_param['dense0_num_units'] = 900
lasagne_oneLayer_param['dropout_p'] = 0.1
lasagne_oneLayer_param['update_learning_rate'] = 0.01
lasagne_oneLayer_param['update_momentum'] = 0.9


lasagne_twoLayer_param['max_epochs'] = 40
lasagne_twoLayer_param['dense0_num_units'] = 300
lasagne_twoLayer_param['dense1_num_units'] = 300
lasagne_twoLayer_param['dropout_p'] = 0.5
lasagne_twoLayer_param['update_learning_rate'] = 0.02



oneLayer_lasagne_clf = clf.lasagne_oneLayer_classifier(lasagne_oneLayer_param, train_X, train_y);
twoLayer_lasagne_clf = clf.lasagne_twoLayer_classifier(lasagne_twoLayer_param, train_X, train_y);

y_prob = oneLayer_lasagne_clf.predict_proba(test_X)
predictions.append(y_prob)

y_prob = twoLayer_lasagne_clf.predict_proba(test_X)
predictions.append(y_prob)


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

