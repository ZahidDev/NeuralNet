#!/usr/bin/env python3

# Written and Maintained by Zahidul Islam (ZahidDev)

import numpy as np


class Validation:
    """Contains Loss metrics for NeuralNet.

    Currently contains log loss implementations of binary and multi-class Neural Networks.
    """

    @staticmethod
    def binary_logistic_loss(output, labels, deriv, epsilon=1e-11):
        """Computes the log loss on binary classifications.

        Args:
            output (ndarray): The predicted labels of the network.
            labels (ndarray): The actual labels of the training data.
            deriv (bool): Whether to use the derivation of the loss metric.
            epsilon (float): Prevent division of zero error
                (Default: 1e-11).

        Return:
            The mean log loss of the current Neural Net. [0, inf)
        """
        output = np.clip(output, epsilon, 1 - epsilon)

        if deriv:
            grad = ((output - labels) / np.maximum(output * (1 - output), epsilon))
            return grad

        loss = np.mean(-np.sum(labels * np.log(output) + (1 - labels) * np.log(1 - output), axis=1))
        return loss

    @staticmethod
    def multinomial_logistic_loss(output, labels, deriv, epsilon=1e-11):
        """Computes the log loss on multi-classifications.

        Args:
            output (ndarray): The predicted labels of the network.
            labels (ndarray): The actual labels of the training data.
            deriv (bool): Whether to use the derivation of the loss metric.
            epsilon (float): Prevent division of zero error
                (Default: 1e-11).

        Return:
            The mean log loss of the current Neural Net. [0, inf)
        """
        output = np.clip(output, epsilon, 1 - epsilon)

        if deriv:
            grad = output - labels

            return grad

        loss = np.mean(-np.sum(labels * np.log(output), axis=1))

        return loss
