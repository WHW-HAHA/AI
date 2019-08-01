import numpy as np

class NeuralNetwork():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_to_hidden = np.random.normal(0, self.input_nodes**-0.5,
                                                        (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0, self.hidden_nodes**-0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        self.activation_function = lambda x: 1/(1+np.exp(-x))

    def train(self, features, targets):
        features = np.array(features, ndmin= 2)
        targets = np.array(targets, ndmin= 2)

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            X = np.array(X, ndmin = 2)
            y = np.array(y, ndmin = 2)

            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = final_inputs

            self.error = y - final_outputs
            error_term = self.error

            hidden_error = np.dot(self.weights_hidden_to_output, error_term) # shape should be hidden_nodes x 1 --> 16 x 1 or 2 x 1
            hidden_error_term = hidden_error * hidden_outputs.T * (1-hidden_outputs).T # 2 x 1

            delta_weights_h_o += error_term * hidden_outputs.T  # shape should like self.w_h_o --> 1 x 2 or 1 x 16
            # delta_weights_i_h += hidden_error_term * X        # shape should like self.w_i_h --> 2 x 3 or 16 x 56
            delta_weights_i_h += X.T * hidden_error_term.T
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        features = np.array(features, ndmin=2)
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs  = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs