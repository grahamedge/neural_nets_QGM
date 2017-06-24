"""network3.py
~~~~~~~~~~~~~~

A Theano-based program by Michael Neilsen for training and running simple neural
networks.

Edited by Graham Edge to apply to learning the positions of atoms 
in a lattice fluoescence image

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.

"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import downsample

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh


#### Constants
GPU = False
verbose = False
if GPU:
    if verbose: print "Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False."
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
elif verbose:
    print "Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True."


#### Load the MNIST data
def load_data(filename = './Data/OneAtomAnywhere_100counts_13px.p'):
    #Load the datasets, keeping them as numpy arrays, for easy plotting and testing
    f = open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close

    return [training_data, validation_data, test_data]

def load_data_shared(filename = './Data/OneAtomAnywhere_100counts_13px.p'):
    #Load data (each of the datasets is a 2-tuple, with first element a list of images
    #   and the second element a list of classification vectors to label filled sites)
    training_data, validation_data, test_data = load_data(filename)

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, shared_y

    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers                    #layers input at creation, which are drawn from the classes below
        self.mini_batch_size = mini_batch_size  #number of data images to consider in each batch
        self.params = [param for layer in self.layers for param in layer.params] #weights (matrix) and biases (vectors) from all layers
        self.x = T.matrix("x")                  #tensor variable for the image input
                                                # batchsize x imagesize matrix
        self.y = T.matrix("y")                 #tensor variable for the classification
                                                # batchsize x latticesize matrix
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        rescale = 10
        training_x = rescale * training_x
        validation_x = rescale * validation_x
        test_x = rescale * test_x

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # Define the (regularized) cost function, symbolic gradients, and updates.
        #  running these lines doesn't actually compute anything yet, since they 
        #  call Theano shared variables such as layer.w, net.y, etc...
        #-----
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])    #sum of all squared weights
        cost = self.layers[-1].cost(self)+\
               0.5*lmbda*l2_norm_squared/num_training_batches   # calculate the regularized cost function from the final layer
        grads = T.grad(cost, self.params)       #all of the gradients of the cost function
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        # - accuracy calculations are based on single-integer identities of each image type
        i = T.lscalar() # mini-batch index
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        train_mb_atom_accuracy = theano.function(
            [i], self.layers[-1].atom_accuracy(self.y),
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        validate_mb_atom_accuracy = theano.function(
            [i], self.layers[-1].atom_accuracy(self.y),
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        test_mb_atom_accuracy = theano.function(
            [i], self.layers[-1].atom_accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
            #Build an atom accuracy test function which will return the accuracy for each image in the minibatch
        self.test_atom_accuracy = theano.function(
            [i], self.layers[-1].atom_accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })        
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_outputs = theano.function(
            [i], self.layers[-1].output,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_inputs = theano.function(
            [i], self.layers[-1].activation_input,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_atom_accuracy(j) for j in xrange(num_validation_batches)])
                    training_accuracy = np.mean(
                        [train_mb_atom_accuracy(j) for j in xrange(num_training_batches)])
                    print("Epoch {0}: training accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            perAtom_test_accuracy = np.mean(
                                 [test_mb_atom_accuracy(j) for j in xrange(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                perAtom_test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))
        self.finalAccuracy = test_accuracy
        self.perAtomAccuracy = perAtom_test_accuracy

        #Examine the final structure of the network
        weights = [layer.w.get_value() for layer in self.layers]
        biases = [layer.b.get_value() for layer in self.layers]

        return weights, biases

#### Define layer types

class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = downsample.max_pool_2d(
            input=conv_out, ds=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))      #batchsize x n_in
        self.output = self.activation_fn(                           #batchsize x n_out
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.activation_input = (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b
        self.y_out = T.round(self.output)                           #batchsize x n_out
        self.inpt_dropout = dropout_layer(                          #batchsize x n_in
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(                   #batchsize x n_out
            T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "For a fully connected layer with sigmoid activation, use cross-entropy cost."
        #Take the mean of cross-entropy costs for each output neuron of each batch image
        return -T.mean(net.y * T.log(self.output_dropout) + (1-net.y) * (T.log(1 - self.output_dropout))) 

    def accuracy(self, y):
        "Return the accuracy for a mini-batch."
        #input y is a matrix with the size (mini_batch_size)x(n_out)
        #self.y_out is a matrix of rounded network activations for each image in the mini_batch
        # accuracy is returned as the fraction of images in the mini_batch for which 
        # each of the n_out activations rounds to the correct lattice filling (1 or 0)

        #Keeping track of the math:
        # T.eq(y, self.y_out)                           has shape   batchsize x n_out
        # T.all(T.eq(y, self.y_out), axis=1)            has shape   batchsize x 1
        # T.mean(T.all(T.eq(y, self.y_out), axis=1))    has shape   1x1

        return T.mean(T.all(T.eq(y, self.y_out), axis=1)) 

    def atom_accuracy(self,y):
        "Return the accuracy per atom, averaged over a mini-batch"    
        #Main difference with 'self.accuracy' is that we don't
        #   use T.all to check for all sites being correct, and instead
        #   average over any site that might be correct or not
        

        return T.mean(T.eq(y,self.y_out))   

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in # 30
        self.n_out = n_out # 9
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.out_labels = np.asarray([2**n for n in np.arange(n_out)])   #binary labelling vector for output
        self.params = [self.w, self.b]
        self.printnum = 0

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.round(self.output)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        # output_dropout is batchsize * n_out
        # net.y is a matrix with the length of the minibatchsize x n_out
        # net.y.shape[0] is the length of the vector... the minibatchsize!
        # [T.arange(net.y.shape[0]), net.y] indexes the outputs (probabilities) of output_dropout
        #   that we would like to be equal to 1 (and so would result in cost = log(1)=0)
        #  !! this method of indexing only works if the desired outputs y can be used as indices
        #      to index the array - because in MNIST the images are actually labelled 0, 1, 2
        #      we can look for the 0, 1, or 2 element of the output vector and make correspondence
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        #it is not clear how best to do this in a softmax layer, when the case of lattice filling
        # requires that more than one lattice site is filled <-> more than one activatino is high

        #softmax will only work if we map each lattice config onto a different output, which
        #  seems like a wasteful way to do it
        return T.mean(T.eq(y, T.dot(self.y_out, self.out_labels)))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
