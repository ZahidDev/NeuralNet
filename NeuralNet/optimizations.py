#!/usr/bin/env python3


class Optimizer:
    """Contains Optimization algorithms for NeuralNet.

    Currently contains working implementation of Classic Gradient Descent.
    """

    @staticmethod
    def grad_descent(weights, grad, alpha):
        """Updates weights with gradient of errors.

        new_weights = old_weights - learning_rate * gradient

        Args:
            weights (list): List of ndarrays containing the weights for each layer.
            grad (list): List of ndarrays containing the gradient of error for each layer.
            alpha (float): The learning rate; How big of a step the optimization takes.

        Returns:
            New list of weights
        """
        desc_weights = list(map(lambda l_w, l_g: l_w - alpha * l_g, weights, grad))
        return desc_weights

    @staticmethod
    def stochastic_grad_descent(weights, batch_size, alpha, grad):
        """Updates weights with gradient of errors stochastically.

        Args:
            weights (list): List of ndarrays containing the weights for each layer.
            batch_size (int): # of training example in each batch.
            grad (list): List of ndarrays containing the gradient of error for each layer.
            alpha (float): The learning rate; How big of a step the optimization takes.

        Returns:
            New list of weights
        """
        pass  # NotImplementedYet
