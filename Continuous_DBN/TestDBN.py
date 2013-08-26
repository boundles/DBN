"""
"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from rbm import RBM
from my_load_data import my_load_data

def my_load_param(param_file):
    save_file = open(param_file, 'rb')
    W1 = theano.shared(value = (cPickle.load(save_file)), borrow = True)
    b1 = theano.shared(value = (cPickle.load(save_file)), borrow = True)
    W2 = theano.shared(value = (cPickle.load(save_file)), borrow = True)
    b2 = theano.shared(value = (cPickle.load(save_file)), borrow = True)
    W3 = theano.shared(value = (cPickle.load(save_file)), borrow = True)
    b3 = theano.shared(value = (cPickle.load(save_file)), borrow = True)
    W4 = theano.shared(value = (cPickle.load(save_file)), borrow = True)
    b4 = theano.shared(value = (cPickle.load(save_file)), borrow = True)
    save_file.close()
    return W1, b1, W2, b2, W3, b3, W4, b4

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W, b, activation = T.tanh):
        
        self.input = input
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W, b):
        self.W = W
        self.b = b
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(' y should have the same shape as self.y_pred',
                            ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class TestDBN(object):
    def __init__(self, rng, input, n_in, n_hidden1, n_hidden2, n_hidden3, n_out, W1, b1, W2, b2, W3, b3, W4, b4):
        self.hiddenLayer1 = HiddenLayer(rng = rng, input = input, n_in = n_in, n_out = n_hidden1, W = W1, b = b1, activation = T.nnet.sigmoid)

        self.hiddenLayer2 = HiddenLayer(rng = rng, input = self.hiddenLayer1.output, n_in = n_hidden1, n_out = n_hidden2, W = W2, b = b2, activation = T.nnet.sigmoid)

        self.hiddenLayer3 = HiddenLayer(rng = rng, input = self.hiddenLayer2.output, n_in = n_hidden2, n_out = n_hidden3, W = W3, b = b3, activation = T.nnet.sigmoid)

        self.logRegressionLayer = LogisticRegression(input = self.hiddenLayer3.output, n_in = n_hidden3, n_out = n_out, W = W4, b = b4)

        self.errors = self.logRegressionLayer.errors
def test(dataset = '/home/wangyang/DeepLearningTutorials/data/mnist.pkl.gz',
         param_file = '/home/wangyang/DeepLearningTutorials/data/DBN_Params',
         batch_size=10,
         n_hidden1 = 1000,
         n_hidden2 = 1000,
         n_hidden3 = 1000):
    
    datasets = my_load_data(dataset)
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for testing
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    W1, b1, W2, b2, W3, b3, W4, b4 = my_load_param(param_file)

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(123)

    test = TestDBN(rng = rng, input = x, n_in = 28*28, n_hidden1 = n_hidden1, n_hidden2 = n_hidden2, n_hidden3 = n_hidden3,
                   n_out = 10, W1 = W1, b1 = b1, W2 = W2, b2 = b2, W3 = W3, b3 = b3, W4 = W4, b4 = b4)

    test_model = theano.function(inputs = [index],
                                 outputs = test.errors(y),
                                 givens = {
                                     x: test_set_x[index * batch_size:(index+1) * batch_size],
                                     y: test_set_y[index * batch_size:(index+1) * batch_size]})
    test_losses = [test_model(i) for i
                   in xrange(n_test_batches)]
    test_score = numpy.mean(test_losses)

    print(('test error is : %f %%') % (test_score*100.))
        
if __name__ == '__main__':
    
    test()
