# Create a neural network
neural_network = NeuralNetwork()

# Train the neural network on the training data
neural_network.train(inputs, targets)

# Generate some text
output = neural_network.generate_text([0, 0, 1])

# Print the text
print(output)
