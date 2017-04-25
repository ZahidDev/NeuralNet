#!/usr/bin/env python3

import numpy as np
from activations import Activator as av
from loss_validations import Validation
from optimizations import Optimizer


class NeuralNet:
    def __init__(self, inputs_dim):
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
        options = {
            'sigmoid': av.sigmoid,
            'tanh': av.tanh,
            'elu': av.elu,
            'softmax': av.softmax
        }

        activation = options[activation_func](signal, deriv)

        return activation

    def forward_prop(self):
        for i in range(1, len(self.network)):
            self.layer_signals[i - 1] = self.network[i -
                                                     1].dot(self.synapse_weights[i - 1])
            self.network[i] = self.activation(
                self.layer_signals[i - 1],
                list(self.layer_activations.values())[i])

            if i < len(self.network) - 1:
                self.network[i][-1] = 1

    def back_prop(self, labels):
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
