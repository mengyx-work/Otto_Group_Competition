import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import ensemble, feature_extraction, preprocessing

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet


'''
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

X = train_df.values.copy()
unkown_X = test_df.values.copy()
unkown_X = unkown_X[:, 1:].astype(np.float32)
#np.random.shuffle(X)
X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
unlabeled_X = np.append(X, unkown_X)
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

#'''
# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
tfidf = tfidf.fit(unlabeled_X)
tfidf_train_X = tfidf.transform(train_X).toarray()
tfidf_test_X = tfidf.transform(test_X).toarray()

lda_model = lda.LDA(n_topics=18, n_iter=2, random_state=1)
lda_model.fit(train_X)
#'''



# standardize the data for NN
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.fit_transform(test_X)

#test_X, test_ids = test_X[:, 1:].astype(np.float32), test_X[:, 0].astype(str)
#test_X = scaler.transform(test_X)


num_classes = len(encoder.classes_)
num_features = train_X.shape[1]


'''
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('output', DenseLayer)]


net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=300,
                 dropout_p=0.5,
                 dense1_num_units=300,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.036,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=30)
'''

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('output', DenseLayer)]


net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=800,
								 dense0_nonlinearity=rectify,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.036,
                 update_momentum=0.9,
                 
                 eval_size=0.2,
                 verbose=1,
                 max_epochs=30)



net0.fit(train_X, train_y)


y_prob = net0.predict_proba(test_X)
print('LogLoss {score}'.format(score=log_loss(test_y, y_prob)))
