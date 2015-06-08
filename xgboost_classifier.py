import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing
import xgboost as xgb
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss

def xgboos_classifer(param):

	# import data
	train = pd.read_csv('train.csv')
	#test = pd.read_csv('test.csv')
	#sample = pd.read_csv('sampleSubmission.csv')

	# alternative label encoding to give a matrix representation
	#targets = pd.get_dummies(train.target.values)

	# drop ids and get labels
	#labels = train.target.values # this gives a ndarray instead of DataFrame #
	labels = train['target']
	train = train.drop('id', axis=1)
	train = train.drop('target', axis=1)
	#test = test.drop('id', axis=1)

	# encode labels 
	label_enc = preprocessing.LabelEncoder()
	labels = label_enc.fit_transform(labels)

	### we need a test set that we didn't train on to find the best weights for combining the classifiers
	sss = StratifiedShuffleSplit(labels, 1, test_size=0.1, random_state=1234)

	for train_index, test_index in sss:
		print 'split the training data'
    #print 'train data, count#', ',' , train_index[0:10]
    #print("TRAIN:", train_index, "TEST:", test_index)

	train_X, train_Y = train.values[train_index], labels[train_index]
	test_X, test_Y = train.values[test_index], labels[test_index]


	#test_temp = np.ndarray(shape=(test.shape[0],),dtype=float)
	xg_train = xgb.DMatrix(train_X, label=train_Y)
	xg_fulltrain = xgb.DMatrix(train, label=labels)
	xg_test = xgb.DMatrix(test_X, label=test_Y)
	#xg_result = xgb.DMatrix(test, label=test_temp)


	#fulllist = [ (xg_train,'train'), (xg_test, 'test') ]
	trainList = [ (xg_train,'train'), ]

	num_round = param['num_round']

	bst = xgb.train(param, xg_train, num_round, trainList );
	# get prediction
	yprob = bst.predict( xg_test )
	logloss_score=log_loss(test_Y,yprob)
	print('LogLoss {0}'.format(logloss_score))
	return logloss_score



############################

## test the error rate 
#ylabel = np.argmax(yprob, axis=1)
#print ('predicting error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))

## train a complete model with the full dataset
#bst = xgb.train(param, xg_fulltrain, num_round, trainList );
#yprob = bst.predict( xg_result )

# create submission file
#preds = pd.DataFrame(yprob, columns=sample.columns[1:])
#preds.index += 1 
#preds.to_csv('xgboost_benchmark.csv', index_label='id')

