import numpy as np
from scipy.special import expit


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, learning_rate: float):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = float(learning_rate)

        self.wih = np.random.normal(
            loc=0.0,
            scale=self.inodes ** -0.5,
            size=(self.hnodes, self.inodes),
        )
        self.who = np.random.normal(
            loc=0.0,
            scale=self.hnodes ** -0.5,
            size=(self.onodes, self.hnodes),
        )

        self.activation = expit

    def query(self, inputs):
        inputs = np.asarray(inputs, dtype=np.float32).reshape(-1, 1)

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        return final_outputs

    def train(self, inputs, targets):
        inputs = np.asarray(inputs, dtype=np.float32).reshape(-1, 1)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 1)

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            hidden_outputs.T,
        )

        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            inputs.T,
        )