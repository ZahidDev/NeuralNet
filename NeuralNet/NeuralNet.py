import numpy as np
from activations import Activator as av


class NeuralNet:
    def __init__(self, inputs_dim):

        self.network = np.array([])
        self.layers_dim = []
        self.hidden_count = 0
        self.layer_activations = {}
        self.layer_signals = []
        self.synapse_weights = []
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
                "Activation function: %s is not available. Please select either ['sigmoid', 'tanh', 'elu', 'softmax']"
                % layer_activation)

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

    def build(self):
        if 'target' not in list(self.layer_activations.keys()):
            raise RuntimeError(
                "Network must contain at least one 'target' layer!")

        network_dim = len(self.layers_dim)
        self.network = np.empty(network_dim, dtype=object)

        for i in range(network_dim - 1):
            self.synapse_weights.append(2 * np.random.random((self.layers_dim[i],
                                                              self.layers_dim[i + 1])) - 1)

        self.is_built = True

    @staticmethod
    def activation(signal, activation_func, deriv=False):

        options = {
            'sigmoid': av.sigmoid,
            'tanh': av.tanh,
            'elu': av.elu,
            'softmax': av.softmax
        }

        a = options[activation_func](signal, deriv)

        return a

    def forward_prop(self):
        for i in range(1, len(self.network)):
            self.layer_signals.append(self.network[i - 1].dot(self.synapse_weights[i - 1]))
            self.network[i] = self.activation(
                self.layer_signals[i - 1],
                list(self.layer_activations.values())[i])

            if i < len(self.network) - 1:
                self.network[i][-1] = 1

        self.network = self.network.reshape((len(self.network), 1))

    def fit_transform(self, inputs, labels):
        if not self.is_built:
            raise RuntimeError("Network hasn't been built yet. Please use the build() method on your network!")

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
