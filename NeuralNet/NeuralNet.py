#!/usr/bin/env python3

# NeuralNet
#
# Written and Maintained by Zahidul Islam (ZahidDev)

import numpy as np
from activations import Activator as av
from loss_validations import Validation
from optimizations import Optimizer


class NeuralNet:
    """Modular Neural Network framework.

    NeuralNet is a simple Deep Learning Neural Network that focuses on modularity.
    Uses numpy vectorization and math to optimize training time.

    Attributes:
        network (list): List of ndarrays containing the values for each node in each layer.
        layers_dim (list): List containing the # of neurons in each layer.
        hidden_count (int): Number of hidden layers in network.
        layer_activations (dict): Dict containing activation function for each layer.
        layer_signals (list): List of ndarrays containing the signals for each Layer.
        synapse_weights (list): List of ndarrays containing the weights for each layer.
        loss_validation (function): Loss Metric used for network 
            Default: Validation.binary_logistic_loss).
        optimizer (function): Optimization algorithm used for network 
            (Default: Optimizer.grad_descent).
        is_built (bool): Whether the network is built.
    """

    def __init__(self, inputs_dim):
        """Initialization of Neural Network.

        Args:
            inputs_dim (int): # of features/neurons in input.
        """
        self.network = [np.array([])]
        self.layers_dim = []
        self.hidden_count = 0
        self.layer_activations = {}
        self.layer_signals = []
        self.synapse_weights = []
        self.loss_validation = Validation.binary_logistic_loss
        self.optimizer = Optimizer.grad_descent
        self.is_built = False

        if inputs_dim <= 0:
            raise ValueError(
                "Must initialize Neural Network with at least 1 neuron for the Input layer")

        self.layers_dim.append(inputs_dim + 1)
        self.layer_activations['input'] = "INPUT"

    def add_layer(self, layer_type, layer_dim, layer_activation='elu'):
        """Adds a layer to Neural Network.

        Args:
            layer_type (str): Type of layer (Default: 'hidden')
            layer_dim (int): # of units/neurons in layer
            layer_activation (str): Activation function used on layer neurons 
                (Default: 'elu')

        Raises:
                OverflowError: Adding layer after adding a target layer
                ValueError: If given args aren't recognized
        """
        if 'target' in self.layer_activations:
            raise OverflowError(
                "Cannot add more layers to Neural Network. target layer already exists")

        if layer_type not in ['hidden', 'target']:
            raise ValueError(
                "Layer must be of either 'hidden' or 'target' type")

        if layer_activation not in ['elu', 'sigmoid', 'softmax', 'tanh']:
            raise ValueError(
                "Activation function: %s is not available. Please select either \
                ['sigmoid', 'tanh', 'elu', 'softmax']" % layer_activation)

        if layer_type == 'target' and layer_dim <= 0:
            raise ValueError("You must have at least 1 target node")

        if layer_type == 'hidden':
            if layer_activation == 'softmax':
                raise UserWarning(
                    "SOFTMAX activation shouldn't be used in any layer other than the target layer!")

            self.hidden_count += 1
            self.layer_activations[layer_type +
                                   str(self.hidden_count)] = layer_activation

            self.layers_dim.append(layer_dim + 1)
        else:
            self.layer_activations[layer_type] = layer_activation
            self.layers_dim.append(layer_dim)

    def build(self, loss_func, optimizer):
        """Compiles Neural Network.

        Sets loss and activation functions for network. Also randomizes weights based on added
        layers in the network.

        Args:
            loss_func (str): Type of loss validation function used
            optimizer (str): Gradient Optimization algorithm used

        Raises:
               RuntimeError: Building network without adding a target layer
        """
        if 'target' not in list(self.layer_activations.keys()):
            raise RuntimeError(
                "Network must contain at least one 'target' layer!")

        self.network *= len(self.layers_dim)
        network_dim = len(self.network)

        for i in range(network_dim - 1):
            self.synapse_weights.append(2 * np.random.random((self.layers_dim[i],
                                                              self.layers_dim[i + 1])) - 1)
        self.layer_signals = [np.array([])] * (network_dim - 1)
        loss_options = {
            'binary_log': Validation.binary_logistic_loss,
            'multi_log': Validation.multinomial_logistic_loss
        }

        optim_options = {
            'pure_GD': Optimizer.grad_descent,
            'SGD': Optimizer.stochastic_grad_descent
        }

        self.loss_validation = loss_options[loss_func]
        self.optimizer = optim_options[optimizer]
        self.is_built = True

    @staticmethod
    def activation(signal, activation_func, deriv=False):
        """Applies an activation function on given signals in a layer.

        Args:
            signal (ndarray): A layer of neuron values multiplied by their weights
            activation_func (str): Given activation function for the layer
            deriv (bool): Whether to use the derived activation function (Default: False)

        Returns:
            A numpy array of activations, which is the layer transformed by a activation function
        """
        options = {
            'sigmoid': av.sigmoid,
            'tanh': av.tanh,
            'elu': av.elu,
            'softmax': av.softmax
        }

        activation = options[activation_func](signal, deriv)

        return activation

    def forward_prop(self):
        """Forward propagates within the network, layer by layer.

        Performs forward propagation, which takes values from the input layer, through the
        hidden layer, to get a target layer. The target layer holds the predictions based on
        the network's model.
        """
        for i in range(1, len(self.network)):
            self.layer_signals[i - 1] = self.network[i -
                                                     1].dot(self.synapse_weights[i - 1])
            self.network[i] = self.activation(
                self.layer_signals[i - 1],
                list(self.layer_activations.values())[i])

            if i < len(self.network) - 1:
                self.network[i][-1] = 1

    def back_prop(self, labels):
        """Retrieves the gradient of the Neural Network.

        Performs Backpropagation to retrieve the gradient of the network which is required
        to optimize the network for better accuracy.

        Args:
            labels (ndarray): The actual labels of the training data

        Returns:
            A list of ndarrays which contains the gradient of error based on the current weights
            and error of predictions of the network for each layer.
        """
        layer_activations = list(self.layer_activations.values())
        output = self.network[-1]
        data_cnt = len(labels)

        loss_grad = self.loss_validation(
            output=output, labels=labels, deriv=True)

        # output_grad = self.activation(
        #     self.layer_signals[-1], layer_activations[-1], True)
        layer_errors = [loss_grad]
        layer_deltas = [self.network[-2].T.dot(layer_errors[0])]

        list(map(lambda w, l_e, z, f: layer_errors.append(l_e.dot(w.T) * self.activation(z, f, True)),
                 self.synapse_weights[::-
                 1], layer_errors, self.layer_signals[::-1][1:],
                 layer_activations[::-1][1:-1]))

        def get_grad(activation, l_e):
            gradient = activation.T.dot(l_e)
            layer_deltas.append(gradient)

        list(map(get_grad, self.network[::-1][1:], layer_errors[1:]))

        grad = list(map(lambda d: (1 / data_cnt) * d, layer_deltas))[::-1]

        return grad

    def optimize(self, labels, alpha, max_epoch, epsilon):
        """Optimizes network for accuracy.

        Updates the learning weights for each layer by a given Optimization algorithm.
        Performs this optimization until a desired accuracy or runtime limit.

        Args:
            labels (ndarray): The actual labels of the training data
            alpha (float): The learning rate; How big of a step the optimization takes.
            max_epoch (int): The maximum number of iterations the Optimization runs upon.
            epsilon (float): The required difference of error in each epoch.
        """
        epoch = 0
        loss = self.loss_validation(output=self.network[-1], labels=labels,
                                    deriv=False)
        delta_loss = 1
        while delta_loss > epsilon:
            if epoch >= max_epoch:
                break

            grad = self.back_prop(labels)
            self.synapse_weights = self.optimizer(
                weights=self.synapse_weights, grad=grad, alpha=alpha)

            self.forward_prop()
            delta_loss = loss - self.loss_validation(output=self.network[-1],
                                                     labels=labels, deriv=False)

            loss = self.loss_validation(output=self.network[-1],
                                        labels=labels, deriv=False)
            epoch += 1

        # GRAD DEBUG
        print(epoch, self.loss_validation(output=self.network[-1],
                                          labels=labels, deriv=False))

    def fit_transform(self, inputs, labels, alpha=0.001, max_epoch=6000, epsilon=1e-6):
        """Fits data into network and trains it.

         User Interface for accessing forward propagation and back propagation.

         Args:
             inputs (ndarray): Input Array with size (training size, # features)
             labels (ndarray): The actual labels of the training data.
             alpha (float): The learning rate; How big of a step the optimization takes.
             max_epoch (int): The maximum number of iterations the Optimization runs upon.
             epsilon (float): The required difference of error in each epoch.
        """
        if not self.is_built:
            raise RuntimeError(
                "Network hasn't been built yet. Please use the build() method on your network!")

        inputs = np.c_[inputs, np.ones((inputs.shape[0], 1))]
        labels = labels
        if not len(inputs) == len(labels):
            raise ValueError(
                "# of training features do not match # of training labels")

        if not inputs.shape[1] == self.layers_dim[0]:
            raise ValueError(
                "Shape of given training set is not (#train_sets, input_dim)")

        self.network[0] = inputs
        self.forward_prop()
        self.optimize(labels, alpha, max_epoch, epsilon)
