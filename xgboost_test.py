import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing
import xgboost as xgb
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.decomposition.pca import PCA

import lda, sys

# import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sampleSubmission.csv')

# alternative label encoding to give a matrix representation
#targets = pd.get_dummies(train.target.values)

# drop ids and get labels
#labels = train.target.values # this gives a ndarray instead of DataFrame #
labels = train['target']
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
test = test.drop('id', axis=1)

# encode labels 
label_enc = preprocessing.LabelEncoder()
labels = label_enc.fit_transform(labels)

unlabeled_train = np.concatenate((train, test), axis=0)

# PCA training
#pca_LowDim = PCA(n_components=40)
#pca_LowDim.fit(unlabeled_train)


### we need a test set that we didn't train on to find the best weights for combining the classifiers
sss = StratifiedShuffleSplit(labels, 1, test_size=0.1, random_state=1234)

for train_index, test_index in sss:
    print 'split the training data'
    #print 'train data, count#', ',' , train_index[0:10]
    #print("TRAIN:", train_index, "TEST:", test_index)
    #break

train_X, train_Y = train.values[train_index], labels[train_index]
test_X, test_Y = train.values[test_index], labels[test_index]

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
tfidf = tfidf.fit(unlabeled_train)
tfidf_train_X = tfidf.transform(train_X).toarray()
tfidf_test_X = tfidf.transform(test_X).toarray()


#lda_model = lda.LDA(n_topics=18, n_iter=2, random_state=1)
#lda_model.fit(train_X)
## new features from LDA
#doc_topic = lda_model.doc_topic_
# contruct new feature matrix
#tfidf_train_X = np.concatenate((tfidf_train_X, doc_topic), axis=1)


## test the LDA model
lda_model = lda.LDA(n_topics=9, n_iter=100, random_state=1)
lda_model.fit(unlabeled_train)
lda_train_X = lda_model.transform(train_X)
lda_test_X = lda_model.transform(test_X)

tfidf_train_X = np.concatenate((tfidf_train_X, lda_train_X), axis=1)
tfidf_test_X = np.concatenate((tfidf_test_X, lda_test_X), axis=1)


#test_matrix = lda_model.fit_transform(train_X)
#transformed_matrix = lda_model.doc_topic_

#print 'total training data: ', unlabeled_train.shape
#print 'original train shape: ', train_X.shape, 'transformed matrix: ', transformed_matrix.shape
#print 'test matrix: ', test_matrix.shape
#sys.exit(1)



## PCA transform train/test to lower dimension
#train_X = pca_LowDim.transform(train_X)
#test_X = pca_LowDim.transform(test_X)

#test_X, test_Y = train.ix[test_index].values, labels.ix[test_index].values

#train_Y = labels
#train_Y = train_Y.astype(float)
test_dummy_label = np.ndarray(shape=(test.shape[0],),dtype=float)
xg_train = xgb.DMatrix(tfidf_train_X, label=train_Y)
xg_fulltrain = xgb.DMatrix(train, label=labels)
xg_test = xgb.DMatrix(tfidf_test_X, label=test_Y)
xg_result = xgb.DMatrix(test, label=test_dummy_label)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.02
param['max_depth'] = 15
param['silent'] = 1
param['eval_metric'] = 'mlogloss'
param['nthread'] = 4
param['min_child_weight'] = 2
param['num_class'] = 9

fulllist = [ (xg_train,'train'), (xg_test, 'test') ]
trainList = [ (xg_train,'train'), ]

num_round = 40

bst = xgb.train(param, xg_train, num_round, trainList );
# get prediction
yprob = bst.predict( xg_test )
print('LogLoss {score}'.format(score=log_loss(test_Y,yprob)))

## test the error rate 
#ylabel = np.argmax(yprob, axis=1)
#print ('predicting error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))

'''
bst = xgb.train(param, xg_fulltrain, num_round, trainList );
yprob = bst.predict( xg_result )

# create submission file
preds = pd.DataFrame(yprob, columns=sample.columns[1:])
preds.index += 1 
preds.to_csv('xgboost_benchmark.csv', index_label='id')
'''
