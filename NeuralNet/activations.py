import numpy as np


def sigmoid(signal, deriv=False):
    if deriv:
        return np.multiply(signal, 1 - signal)
    activation = 1 / (1 + np.exp(-signal))
    return activation


def tanh(signal, deriv=False):
    if deriv:
        return 1 - np.power(np.tanh(signal), 2)
    activation = np.tanh(signal)
    return activation


def elu(signal, deriv=False, alpha=1.0):
    activation = np.piecewise(signal, [signal < 0, signal >= 0], [
        lambda z: alpha * (np.exp(z) - 1), lambda z: z])

    if deriv:
        activation[activation >= 0] = 1

        x = activation[activation < 0]
        derivative = np.piecewise(x, [x < 0, x >= 0], [lambda z: alpha * (np.exp(z) - 1), lambda z: z]) + alpha

        activation[activation < 0] = derivative

    return activation


def softmax(signal, deriv=False):
    signal = signal - np.max(signal)
    if deriv:
        return np.exp(signal) * (1 - np.exp(signal))

    activation = np.exp(signal) / np.array([np.sum(np.exp(signal), axis=1)]).T
    return activation
