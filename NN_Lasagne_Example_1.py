import pandas as pd
from sklearn.cross_validation import train_test_split 
from nolearn.lasagne import NeuralNet, BatchIterator
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import momentum, nesterov_momentum, sgd, rmsprop
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


def plot_loss(net):
    """
    Plot the training loss and validation loss versus epoch iterations with respect to 
    a trained neural network.
    """
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.plot(train_loss, linewidth = 3, label = "train")
    pyplot.plot(valid_loss, linewidth = 3, label = "valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    #pyplot.ylim(1e-3, 1e-2)
    pyplot.yscale("log")
    pyplot.show()


train_df = pd.read_csv('./data/train.csv') 
test_df = pd.read_csv('./data/test.csv') 

train_label = train_df.values[:, 0]
train_data = train_df.values[:, 1:]
print "train:", train_data.shape, train_label.shape

test_data = test_df.values
print "test:", test_data.shape

train_data = train_data.astype(np.float)
train_label = train_label.astype(np.int32)
train_data, train_label = shuffle(train_data, train_label, random_state = 21)


#train_data, valid_data, train_label, valid_label = train_test_split(train_data, train_label, test_size = 0.2, random_state = 21)

fc_1hidden = NeuralNet(
    layers = [  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('dropout', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape = (None, 784),  # 28x28 input pixels per batch
    hidden_num_units = 100,  # number of units in hidden layer
    dropout_p = 0.25, # dropout probability
    output_nonlinearity = softmax,  # output layer uses softmax function
    output_num_units = 10,  # 10 labels

    # optimization method:
    #update = nesterov_momentum,
    update = sgd,
    update_learning_rate = 0.001,
    #update_momentum = 0.9,

    eval_size = 0.1,

    # batch_iterator_train = BatchIterator(batch_size = 20),
    # batch_iterator_test = BatchIterator(batch_size = 20),

    max_epochs = 100,  # we want to train this many epochs
    verbose = 1,
    )

fc_1hidden.fit(train_data, train_label)
plot_loss(fc_1hidden)
