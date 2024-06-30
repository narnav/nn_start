import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Load the weights
weights_input_hidden = np.load('weights_input_hidden.npy')
weights_hidden_output = np.load('weights_hidden_output.npy')

# Predict function
def predict(number):
    normalized_number = np.array([[number / 1000.0]])
    hidden_layer_input = np.dot(normalized_number, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)
    return 'Even' if predicted_output < 0.5 else 'Odd'

# Test the prediction function
test_number = 15
print(f"{test_number} is {predict(test_number)}")
