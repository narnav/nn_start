"""
    This application tech a neural network XOR
    Train the neural network using backpropagation and gradient descent.

    The training loop runs for 10,000 epochs. In each epoch:
    - Perform feedforward propagation to compute the activations of each layer.
    - Calculate the error and perform backpropagation to compute the gradients.
    - Update the weights using the gradients and the learning rate.

    Parameters:
    None

    Returns:
    None
    """

import numpy as np
from icecream import ic
# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0],[1],[2],[3],[4],[5]])

# Output dataset
y = np.array([[1],[0],[1],[0],[1],[0]])

# Seed for reproducibility
np.random.seed(1)

# Initialize weights randomly with mean 0
input_layer_neurons = X.shape[1]  # Number of features in input
hidden_layer_neurons = 4  # Number of neurons in hidden layer
output_neurons = 1  # Number of neurons in output layer

# Weight initialization
# print(hidden_layer_neurons)
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# Learning rate
lr = 0.1

# Training the neural network
for epoch in range(1000000):
    # Feedforward
    hidden_layer_activation = np.dot(X, weights_input_hidden)
    # print(weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    # print(X)
    
    # print(hidden_layer_output)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_activation)
    # print(predicted_output[0])
    # Backpropagation

    # print(y)
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * lr
    weights_input_hidden += X.T.dot(d_hidden_layer) * lr



# Testing the neural network
test_input = np.array([[4]])


# Define a function to make predictions
def predict(X):
    
    hidden_layer_activation = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_activation)
    ic(output_layer_activation)
    return predicted_output

print("Predictions for test input: ")
print(predict(test_input))
