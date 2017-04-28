#!/usr/bin/env python3

import numpy as np


class Activator:
    """Contains Activation functions for NeuralNet.

    Currently contains implementations of sigmoid, tanh, elu,
    and softmax functions
    """

    @staticmethod
    def sigmoid(signal, deriv=False):
        """Sigmoid Activation.

        Args:
             signal (ndarray): A layer of neuron values multiplied by their weights.
             deriv (bool): Whether to compute the derivative of the activation function.
        Returns:
            Takes signals from neurons in layer and applies a sigmoid transformation.
        """
        if deriv:
            return np.multiply(signal, 1 - signal)
        activation = 1 / (1 + np.exp(-signal))
        return activation

    @staticmethod
    def tanh(signal, deriv=False):
        """tanh Activation.

        Known as the Hyperbolic tanget, it's similar to sigmoid but it instead approximates
        the identity near the origin.

        --> 2 / (1 + np.exp(-2*signal)) - 1

        Args:
             signal (ndarray): A layer of neuron values multiplied by their weights.
             deriv (bool): Whether to compute the derivative of the activation function.
        Returns:
            Takes signals from neurons in layer and applies a tanh transformation.
        """
        if deriv:
            return 1 - np.power(np.tanh(signal), 2)
        activation = np.tanh(signal)
        return activation

    @staticmethod
    def elu(signal, deriv=False, alpha=1.0):
        """Exponential Learning Unit

        https://arxiv.org/abs/1511.07289

        Args:
            signal (ndarray): A layer of neuron values multiplied by their weights.
            deriv (bool): Whether to compute the derivative of the activation function.
        Returns:
            Takes signals from neurons in layer and applies a linear/exponential transformation.
        """
        activation = (signal >= 0).astype(int) * signal + \
                     (signal < 0).astype(int) * (alpha * (np.exp(signal) - 1))

        if deriv:
            derivation = (signal >= 0).astype(int) + \
                         (signal < 0) * (activation + alpha)
            return derivation

        return activation

    @staticmethod
    def softmax(signal, deriv=False):
        """Softmax Categorical Function

        Gives the probability of the output signal being in given classes. [0, 1]

        Args:
            signal (ndarray): A layer of neuron values multiplied by their weights.
            deriv (bool): Whether to compute the derivative of the activation function.
        Returns:
            Takes signals from neurons in layer and applies a SoftMax Regression.
        """
        signal = signal - np.max(signal)
        activation = np.exp(signal) / \
            np.array([np.sum(np.exp(signal), axis=1)]).T

        if deriv:  # Derivation looks fishy.
            jacobian = - activation[..., None] * activation[:, None, :]
            iy, ix = np.diag_indices_from(jacobian[0])
            jacobian[:, iy, ix] = activation * (1 - activation)

            return jacobian.sum(axis=1)

        return activation
