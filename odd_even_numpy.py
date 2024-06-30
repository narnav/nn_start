import numpy as np
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Generate dataset
X = np.arange(1000).reshape(-1, 1)
y = (X % 2).reshape(-1, 1)

# Normalize the input data
X = X / 1000.0

# Split the dataset into training and test sets
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Neural network parameters
input_neurons = 1
hidden_neurons = 10
output_neurons = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights randomly with mean 0
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Training the neural network
for epoch in range(epochs):
    # Feedforward
    hidden_layer_input = np.dot(X_train, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    error = y_train - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X_train.T.dot(d_hidden_layer) * learning_rate

# Convert weights to lists for JSON serialization
weights_input_hidden_list = weights_input_hidden.tolist()
weights_hidden_output_list = weights_hidden_output.tolist()

# Save the weights to a JSON file
model_data = {
    'weights_input_hidden': weights_input_hidden_list,
    'weights_hidden_output': weights_hidden_output_list
}

with open('model_weights.json', 'w') as json_file:
    json.dump(model_data, json_file)

# Loading the weights from JSON
with open('model_weights.json', 'r') as json_file:
    model_data = json.load(json_file)

# Convert lists back to NumPy arrays
weights_input_hidden = np.array(model_data['weights_input_hidden'])
weights_hidden_output = np.array(model_data['weights_hidden_output'])

# Testing the neural network
hidden_layer_input = np.dot(X_test, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
predicted_output = sigmoid(output_layer_input)

# Convert predictions to binary outcomes (0 or 1)
predicted_output_binary = (predicted_output > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predicted_output_binary == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

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
