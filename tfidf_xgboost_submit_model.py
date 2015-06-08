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

raw_train_X = X.values
raw_test_X = test.values


# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
combined_X = np.concatenate((raw_train_X, raw_test_X), axis=0)
tfidf = tfidf.fit(combined_X)
train_X = tfidf.transform(raw_train_X).toarray()
test_X = tfidf.transform(raw_test_X).toarray()


################################################################################
'''
train = pd.read_csv('tfidf_lda_18features_train.csv')
test = pd.read_csv('tfidf_lda_18features_test.csv')
sample = pd.read_csv('sampleSubmission.csv')

train_y = train.values[:, -1].astype(np.int32)
train_X = train.values[:, :-1]

test_X = test.values
'''
################################################################################



## list of prediction results from different classifiers 
predictions = []


#############################
##  XGBoost Classifier
#############################

#test_X = np.nan_to_num(test_X)

xgb_param = param.xgboost_para()
xgb_param['num_round'] = 2

xgboost_clf = clf.xgboost_classifer(xgb_param, train_X, train_y);

test_dummy_y = np.ndarray(shape=(test_X.shape[0],),dtype=float)
## xgboost input data requires specifc format
xg_test = xgb.DMatrix(test_X)
## prediction from xgboost
y_prob = xgboost_clf.predict(xg_test)


################################################################################

# create submission file
preds = pd.DataFrame(y_prob, columns=sample.columns[1:])
preds.index += 1 
preds.to_csv('temp_final_tfidf_xgboost_results.csv', index_label='id')


