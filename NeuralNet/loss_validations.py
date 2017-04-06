import numpy as np


class Validation:

    @staticmethod
    def binary_cross_entropy(output, labels, epsilon=1e-11):
        output = np.clip(output, epsilon, 1 - epsilon)
        t_len = len(output)

        loss = (-1 / t_len) * np.sum(labels * np.log(output) + (1 - labels) * np.log(1 - output))
        return loss

