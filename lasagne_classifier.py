import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedShuffleSplit

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


def lasagne_classifier(param):

	## load train/test data
	train_df = pd.read_csv('./train.csv')
	#test_df = pd.read_csv('./test.csv')

	X = train_df.values.copy()
	#test_X = test_df.values.copy()
	#np.random.shuffle(X)
	X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]

	## encode the labels for different classes
	encoder = LabelEncoder()
	y = encoder.fit_transform(labels).astype(np.int32)

	### we need a test set that we didn't train on to find the best weights for combining the classifiers
	sss = StratifiedShuffleSplit(y, 1, test_size=0.1, random_state=1234)

	for train_index, test_index in sss:
		print 'split the training data'

	## input for NN is within [0, 1], need to scale input
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	train_X, train_y = X[train_index], y[train_index]
	check_X, check_y = X[test_index], y[test_index]


	#test_X, test_ids = test_X[:, 1:].astype(np.float32), test_X[:, 0].astype(str)
	#test_X = scaler.transform(test_X)

	num_classes = len(encoder.classes_)
	num_features = train_X.shape[1]

	## initialize the NN
	layers0 = [('input', InputLayer),
           	('dense0', DenseLayer),
           	('dropout', DropoutLayer),
           	('dense1', DenseLayer),
           	('output', DenseLayer)]


	net0 = NeuralNet(layers=layers0,

                 	input_shape=(None, num_features),
                 	dense0_num_units=param['dense0_num_units'],
                 	dropout_p=param['dropout_p'],
                 	dense1_num_units=param['dense1_num_units'],
                 	output_num_units=num_classes,
                 	output_nonlinearity=softmax,
                 
                 	update=nesterov_momentum,
                 	update_learning_rate=param['update_learning_rate'],
                 	update_momentum=param['update_momentum'],
                 
                 	eval_size=0.2,
                 	verbose=1,
                 	max_epochs=param['max_epochs'])

	## fit the results
	net0.fit(train_X, train_y)


	y_prob = net0.predict_proba(check_X)
	logloss_result = log_loss(check_y, y_prob)
	print 'reulst: ', logloss_result
	#print('LogLoss {score}'.format(score=log_loss(check_y, y_prob)))
	return logloss_result 
