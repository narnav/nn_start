import numpy as np
import json
from icecream import ic

LEARN =False

def sigmoid(x):
    """
    Compute the sigmoid activation function.
    
    Parameters:
    x (numpy.ndarray): The input array for which to compute the sigmoid function.
    
    Returns:
    numpy.ndarray: The output array with the sigmoid of each element of x.
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Compute the derivative of the sigmoid function.
    
    Parameters:
    x (numpy.ndarray): The input array for which to compute the derivative of the sigmoid function.
    
    Returns:
    numpy.ndarray: The output array with the derivative of the sigmoid function applied to each element of x.
    """
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output dataset
y = np.array([[0], [1], [1], [0]])

# Seed for reproducibility
np.random.seed(1)

# Initialize weights randomly with mean 0
input_layer_neurons = X.shape[1]  # Number of features in input
hidden_layer_neurons = 4  # Number of neurons in hidden layer
output_neurons = 1  # Number of neurons in output layer

# Weight initialization
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))

# Learning rate
lr = 0.1
if LEARN:
    # Training the neural network
    for epoch in range(100000):
        # Feedforward
        hidden_layer_activation = np.dot(X, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_activation)

        output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_activation)

        # Backpropagation
        error = y - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

        # Updating Weights
        weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * lr
        weights_input_hidden += X.T.dot(d_hidden_layer) * lr

    # Save the weights to a JSON file
def save_weights(weights_input_hidden, weights_hidden_output, filename='model_weights.json'):
    """
    Save the neural network weights to a JSON file.

    Parameters:
    weights_input_hidden (numpy.ndarray): Weights between input and hidden layer.
    weights_hidden_output (numpy.ndarray): Weights between hidden and output layer.
    filename (str): Name of the file to save the weights.

    Returns:
    None
    """
    if LEARN:
        weights = {
            'weights_input_hidden': weights_input_hidden.tolist(),
            'weights_hidden_output': weights_hidden_output.tolist()
        }
        with open(filename, 'w') as json_file:
            json.dump(weights, json_file)

# Load the weights from a JSON file
def load_weights(filename='model_weights.json'):
    """
    Load the neural network weights from a JSON file.

    Parameters:
    filename (str): Name of the file to load the weights from.

    Returns:
    tuple: Tuple containing weights between input and hidden layer, and weights between hidden and output layer.
    """
    with open(filename, 'r') as json_file:
        weights = json.load(json_file)
    weights_input_hidden = np.array(weights['weights_input_hidden'])
    weights_hidden_output = np.array(weights['weights_hidden_output'])
    return weights_input_hidden, weights_hidden_output

# Save the trained weights
save_weights(weights_input_hidden, weights_hidden_output)

# Load the trained weights
weights_input_hidden, weights_hidden_output = load_weights()

# Define a function to make predictions
def predict(X):
    """
    Predict the output for a given input using the trained neural network.

    Parameters:
    X (numpy.ndarray): The input array for which to make the prediction.

    Returns:
    numpy.ndarray: The predicted output.
    """
    hidden_layer_activation = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_activation)
    ic(output_layer_activation)
    return predicted_output

# Testing the neural network with test input
test_input = np.array([[1, 1]])
print("Predictions for test input: ")
print(f'Prediction for {test_input},{predict(test_input)}')
