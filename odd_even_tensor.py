import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generate dataset
X = np.arange(100)  # Numbers from 0 to 999
y = X % 2  # 0 for even, 1 for odd

# Normalize the input data
X = X / 100.0

# Split the dataset into training and test sets
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Build the neural network
model = Sequential([
    Dense(16, input_dim=1, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict on new data
def is_even_or_odd(number):
    normalized_number = number / 100.0
    prediction = model.predict(np.array([normalized_number]))
    return 'Even' if prediction < 0.5 else 'Odd'

# Test the prediction function
test_number = 12
print(f"{test_number} is {is_even_or_odd(test_number)}")
print(f"{11} is {is_even_or_odd(11)}")
print(f"{14} is {is_even_or_odd(14)}")
print(f"{44} is {is_even_or_odd(44)}")
