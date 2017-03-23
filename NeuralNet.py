import numpy as np
import activations as av


# noinspection PyTypeChecker
class NeuralNet:
    def __init__(self, inputs_dim):

        self.network = np.array([])
        self.layers_dim = []
        self.hidden_count = 0
        self.layer_activations = {}
        self.layer_signals = np.array([])
        self.synapse_weights = np.array([])
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
        self.synapse_weights = np.empty(network_dim - 1, dtype=object)
        self.layer_signals = np.empty(network_dim - 1, dtype=object)

        for i in range(len(self.synapse_weights)):
            self.synapse_weights[i] = 2 * np.random.random((self.layers_dim[i], self.layers_dim[i + 1])) - 1

        self.synapse_weights = self.synapse_weights.reshape((len(self.synapse_weights), 1))

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
            self.layer_signals[i - 1] = np.dot(self.network[i - 1], self.synapse_weights[i - 1][0])
            self.network[i] = self.activation(
                self.layer_signals[i - 1],
                list(self.layer_activations.values())[i])

            if i < len(self.network) - 1:
                self.network[i][-1] = 1

        self.network = self.network.reshape((len(self.network), 1))
        self.layer_signals = self.layer_signals.reshape((len(self.layer_signals), 1))

    def cross_entrophy_error(self, labels, deriv=False, lmbda=0.01, epsilon=1e-11):
        output = np.clip(self.network[-1][0], epsilon, 1 - epsilon)
        num_inputs = len(self.network[0][0])

        if deriv:
            gradient =

        regularize = (lmbda / (2 * num_inputs)) * np.power(np.sum(
            np.sum(self.synapse_weights[w][0]) for w in range(len(self.synapse_weights))), 2)

        loss = (-1 / num_inputs) * np.sum(labels * np.log(outputs) + (1 - labels) * np.log(1 - outputs)) + regularize

        return loss

    # def back_prop(self, labels):
    #     layer_activations = list(self.layer_activations.values())
    #     output = self.network[-1][0]
    #
    #     layer_errors = np.empty(len(self.network) - 1, dtype=object)
    #     layer_errors[-1] = (output - labels) * self.activation(self.layer_signals[-1][0], layer_activations[-1], True)
    #
    #     layer_deltas = np.empty(len(layer_errors), dtype=object)
    #     layer_deltas[-1] = np.dot(self.network[-2][0].T, layer_errors[-1])
    #
    #     for i in range(-2, -len(self.network), -1):
    #         hidden_error = (np.dot(layer_errors[i + 1], self.synapse_weights[i + 1][0].T)) * \
    #                        self.activation(self.layer_signals[i][0], layer_activations[i], True)
    #
    #         hidden_error[:, -1] = 1
    #         layer_errors[i] = hidden_error
    #
    #         hidden_delta = np.dot(self.network[i - 1][0].T, layer_errors[i])
    #         hidden_delta[:, -1] = 1
    #         layer_deltas[i] = hidden_delta
    #
    #     layer_deltas = layer_deltas.reshape((len(layer_deltas), 1))
    #     cost_deriv = (1 / len(self.network[0][0])) * (layer_deltas + (0.01 * self.synapse_weights))
    #
    #     return cost_deriv
    #
    # def gradient(self, labels, alpha, error_thresh):
    #     loss = self.cross_entrophy_error(labels)
    #     epoch = 0
    #     while loss > error_thresh:
    #         self.network = self.network.reshape((len(self.network)))
    #         self.layer_signals = self.layer_signals.reshape((len(self.layer_signals)))
    #
    #         self.forward_prop()
    #         cost_deriv = self.back_prop(labels)
    #         self.synapse_weights += alpha * cost_deriv
    #
    #         loss = self.cross_entrophy_error(labels)
    #         epoch += 1

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
        self.gradient(labels, 0.001, 0.05)


train_sets = 4
input_dim = 2
output_dim = 1
hidden_dim = int(np.ceil(np.mean([input_dim, output_dim])))

neural_net = NeuralNet(input_dim)
neural_net.add_layer(layer_type='hidden', layer_dim=hidden_dim)
neural_net.add_layer(layer_type='hidden', layer_dim=hidden_dim)
neural_net.add_layer(layer_type='target', layer_dim=output_dim,
                     layer_activation='softmax')
neural_net.build()

X = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
y = np.array([[0], [1], [0], [1]])
neural_net.fit_transform(X, y)
