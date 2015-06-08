import pandas as pd
from lasagne import layers
from lasagne.nonlinearities import  softmax, rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np

def fit_model(train_x, y, test_x):
    """Feed forward neural network for kaggle digit recognizer competition.
    Intentionally limit network size and optimization time (by choosing max_epochs = 15) to meet runtime restrictions
    """
    print("\n\nRunning Convetional Net.  Optimization progress below\n\n")
    net1 = NeuralNet(
        layers=[  #list the layers here
            ('input', layers.InputLayer),
            ('hidden1', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],

        # layer parameters:
        input_shape=(None, train_x.shape[1]),
        hidden1_num_units=200, hidden1_nonlinearity=rectify,  #params of first layer
        output_nonlinearity=softmax,  # softmax for classification problems
        output_num_units=10,  # 10 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.05,
        update_momentum=0.7,

        regression=False,
        max_epochs=10,  # Intentionally limited for execution speed
        verbose=1,
        )

    net1.fit(train_x, y)
    predictions = net1.predict(test_x)
    return(predictions)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": range(1,len(preds)+1), "Label": preds}).to_csv(fname, index=False, header=True)

    
# Read data
train = pd.read_csv('../input/train.csv')
train_y = train.ix[:,0].values.astype('int32')

# Divide pixel brightnesses by max brightness so they are between 0 and 1
# This helps the network optimizer make changes on the right order of magnitude
pixel_brightness_scaling_factor = train.max().max()
train_x = (train.ix[:,1:].values/pixel_brightness_scaling_factor).astype('float32')
test_x = (pd.read_csv('../input/test.csv').values/pixel_brightness_scaling_factor).astype('float32')

# Fit a (non-convolutional) neural network
basic_nn_preds = fit_model(train_x, train_y, test_x)
write_preds(basic_nn_preds, "basic_nn.csv")

print("FOR the next step up, try a convolutional NN https://www.kaggle.com/users/9028/danb/digit-recognizer/convolutional-nn-in-python")

