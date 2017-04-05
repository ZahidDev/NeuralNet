import numpy as np


class Activator:
    @staticmethod
    def sigmoid(signal, deriv=False):
        if deriv:
            return np.multiply(signal, 1 - signal)
        activation = 1 / (1 + np.exp(-signal))
        return activation

    @staticmethod
    def tanh(signal, deriv=False):
        if deriv:
            return 1 - np.power(np.tanh(signal), 2)
        activation = np.tanh(signal)
        return activation

    @staticmethod
    def elu(signal, deriv=False, alpha=1.0):

        activation = (signal >= 0).astype(int) * signal + \
                     (signal < 0).astype(int) * (alpha * (np.exp(signal) - 1))

        if deriv:
            activation = (signal >= 0).astype(int) + \
                         (signal < 0) * (Activator.elu(signal) + alpha)

        return activation

    @staticmethod
    def softmax(signal, deriv=False):
        signal = signal - np.max(signal)
        # Implement correct derivation for the softmax normalization
        if deriv:
            return np.exp(signal) * (1 - np.exp(signal))

        activation = np.exp(signal) / np.array([np.sum(np.exp(signal), axis=1)]).T
        return activation
