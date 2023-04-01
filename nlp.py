import numpy as np

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator so that the results are reproducible
        np.random.seed(1)

        # Create a 3x1 matrix of random weights between -1 and 1
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, inputs):
        # Multiply the inputs by the weights
        z = np.dot(inputs, self.synaptic_weights)

        # Apply the sigmoid function
        output = self.sigmoid(z)

        return output

    def back_propagation(self, inputs, targets):
        # Calculate the error
        error = targets - output

        # Calculate the derivative of the sigmoid function
        derivative_of_sigmoid = self.sigmoid(z) * (1 - self.sigmoid(z))

        # Calculate the gradient of the weights
        gradients = np.dot(error, inputs.T) * derivative_of_sigmoid

        return gradients

    def train(self, inputs, targets):
        # Initialize the learning rate to a small value
        learning_rate = 0.01

        # Iterate over the training data
        for i in range(len(inputs)):
            # Calculate the gradients
            gradients = self.back_propagation(inputs[i], targets[i])

            # Update the weights
            self.synaptic_weights -= learning_rate * gradients

    def generate_text(self, inputs):
        # Initialize the output to 0
        output = 0

        # Iterate over the inputs
        for i in range(len(inputs)):
            # Multiply the output by the weights and add the bias
            z = np.dot(inputs[i], self.synaptic_weights) + self.bias

            # Apply the sigmoid function
            output = self.sigmoid(z)

            # Return the output
            return output
