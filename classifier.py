import pandas as pd
import numpy as np

import xgboost as xgb

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import StandardScaler

##########################################################################################

def xgboost_classifer(param, X, labels):

	xg_train = xgb.DMatrix(X, label=labels)
	
	#test_dummy_label = np.ndarray(shape=(test_X.shape[0],),dtype=float)
	#xg_test = xgb.DMatrix(test_X, label=test_dummy_label)

	trainList = [ (xg_train,'train'), ]
	num_round = param['num_round']

	bst = xgb.train(param, xg_train, num_round, trainList );
	return bst
	## get prediction
	#yprob = bst.predict( xg_test )
	#return yprob

##########################################################################################

def lasagne_twoLayer_classifier(lasagne_param, X, labels):

	## initialize the NN
	layers0 = [('input', InputLayer),
           	('dense0', DenseLayer),
           	('dropout', DropoutLayer),
           	('dense1', DenseLayer),
           	('output', DenseLayer)]


	net0 = NeuralNet(layers=layers0,

                 	input_shape=(None, lasagne_param['num_features']),
                 	dense0_num_units=lasagne_param['dense0_num_units'],
                 	dropout_p=lasagne_param['dropout_p'],
                 	dense1_num_units=lasagne_param['dense1_num_units'],
                 	output_num_units=lasagne_param['num_classes'],
                 	output_nonlinearity=softmax,
                 
                 	update=nesterov_momentum,
                 	update_learning_rate=lasagne_param['update_learning_rate'],
                 	update_momentum=lasagne_param['update_momentum'],
                 
                 	eval_size=0.02,
                 	verbose=1,
                 	max_epochs=lasagne_param['max_epochs'])

	## fit the results
	net0.fit(X, labels)
	
	return net0

	#y_prob = net0.predict_proba(test_X)
	#print 'reulst: ', len(y_prob)
	#return y_prob

##########################################################################################

def lasagne_oneLayer_classifier(param, X, labels):

	## initialize the NN
	layers0 = [('input', InputLayer),
           	('dense0', DenseLayer),
           	('dropout', DropoutLayer),
           	('output', DenseLayer)]


	net0 = NeuralNet(layers=layers0,

                 	input_shape=(None, param['num_features']),
                 	dense0_num_units=param['dense0_num_units'],
                 	dropout_p=param['dropout_p'],
                 	output_num_units=param['num_classes'],
                 	output_nonlinearity=softmax,
                 
                 	update=nesterov_momentum,
                 	update_learning_rate=param['update_learning_rate'],
                 	update_momentum=param['update_momentum'],
                 
                 	eval_size=0.02,
                 	verbose=1,
                 	max_epochs=param['max_epochs'])

	## fit the results
	net0.fit(X, labels)
	
	return net0



