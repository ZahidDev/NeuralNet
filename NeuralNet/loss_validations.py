#!/usr/bin/env python3
import numpy as np


class Validation:
    @staticmethod
    def binary_logistic_loss(output, labels, deriv, epsilon=1e-11):
        output = np.clip(output, epsilon, 1 - epsilon)

        if deriv:
            grad = ((output - labels) / np.maximum(output * (1 - output), epsilon))
            return grad

        loss = np.mean(-np.sum(labels * np.log(output) + (1 - labels) * np.log(1 - output), axis=1))
        return loss

    @staticmethod
    def multinomial_logistic_loss(output, labels, deriv, epsilon=1e-11):
        output = np.clip(output, epsilon, 1 - epsilon)

        if deriv:
            grad = output - labels

            return grad

        loss = np.mean(-np.sum(labels * np.log(output), axis=1))

        return loss
