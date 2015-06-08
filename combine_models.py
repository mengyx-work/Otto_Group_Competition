import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import xgboost as xgb

import classifier as clf

#import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# alternative label encoding to give a matrix representation
#targets = pd.get_dummies(train.target.values)

X = train.values.copy()
unknown_X = test.values.copy()
#np.random.shuffle(X)
X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
unknown_X = unknown_X[:, 1:]

## pandas operation
#test = test.drop('id', axis=1)


# encode labels 
label_enc = LabelEncoder()
labels = label_enc.fit_transform(labels).astype(np.int32)
simple_labels = label_enc.fit_transform(labels)


### we need a test set that we didn't train on to find the best weights for combining the classifiers
sss = StratifiedShuffleSplit(labels, 1, test_size=0.1, random_state=1234)

for train_index, test_index in sss:
    print 'split the training data'


train_X, train_Y = X[train_index], labels[train_index]
test_X, test_Y = X[test_index], labels[test_index]


### building the classifiers
predictions = []

#############################

# setup parameters for xgboost
xgb_param = {}
# use softmax multi-class classification
xgb_param['objective'] = 'multi:softprob'
# scale weight of positive examples
xgb_param['eta'] = 0.1
xgb_param['max_depth'] = 15
xgb_param['silent'] = 1
xgb_param['eval_metric'] = 'mlogloss'
xgb_param['nthread'] = 4
xgb_param['min_child_weight'] = 3
xgb_param['num_class'] = 9
xgb_param['num_round'] = 40

xgboost_clf = clf.xgboost_classifer(xgb_param, train_X, train_Y);

## xgboost input data requires specifc format
xg_test = xgb.DMatrix(test_X, label=test_Y)
## prediction from xgboost
y_prob = xgboost_clf.predict(xg_test)

predictions.append(y_prob)

#############################

lasagne_param = {}

lasagne_param['max_epochs'] = 40
lasagne_param['dense0_num_units'] = 100
lasagne_param['dense1_num_units'] = 100
lasagne_param['dropout_p'] = 0.1
lasagne_param['num_classes'] = len(label_enc.classes_)
lasagne_param['num_features'] = train_X.shape[1]


scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)

lasagne_clf = clf.lasagne_classifier(lasagne_param, train_X, train_Y);

y_prob = lasagne_clf.predict_proba(test_X)
print 'reulst: ', len(y_prob)

#lasagne_result = clf.lasagne_classifier(lasagne_param, train_X, train_Y, test_X);

#predictions.append(lasagne_result.tolist())
predictions.append(y_prob)

#############################

#for element in predictions:
#	print 'the shape is: ', element.shape
  
#print 'label shape: ', labels.shape

 
def log_loss_func(weights):
    ### scipy minimize will pass the weights as a numpy array ###
	final_prediction = 0
	weighted_predictions = [weight*prediction for weight, prediction in zip(weights,predictions)]
	final_prediction = sum(weighted_predictions)
	return log_loss(test_Y, final_prediction)

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





'''
test_temp = np.ndarray(shape=(test_X.shape[0],),dtype=float)
xg_train = xgb.DMatrix(X, label=labels)
xg_test = xgb.DMatrix(test_X, label=test_temp)

trainList = [ (xg_train,'train'), ]
num_round = param['num_round']

bst = xgb.train(param, xg_train, num_round, trainList );
# get prediction
yprob = bst.predict( xg_test )
'''



'''

### building the classifiers
clfs = []

rfc = RandomForestClassifier(n_estimators=50, random_state=4141, n_jobs=-1)
rfc.fit(train_x, train_y)
print('RFC LogLoss {score}'.format(score=log_loss(test_y, rfc.predict_proba(test_x))))
clfs.append(rfc)

### usually you'd use xgboost and neural nets here

logreg = LogisticRegression()
logreg.fit(train_x, train_y)
print('LogisticRegression LogLoss {score}'.format(score=log_loss(test_y, logreg.predict_proba(test_x))))
clfs.append(logreg)

rfc2 = RandomForestClassifier(n_estimators=50, random_state=1337, n_jobs=-1)
rfc2.fit(train_x, train_y)
print('RFC2 LogLoss {score}'.format(score=log_loss(test_y, rfc2.predict_proba(test_x))))
clfs.append(rfc2)


### finding the optimum weights

predictions = []
for clf in clfs:
    predictions.append(clf.predict_proba(test_x))
    
def log_loss_func(weights):
    ### scipy minimize will pass the weights as a numpy array ###
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
        final_prediction += weight*prediction
    
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

'''
