import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Load dataset from CSV file
# Replace 'dataset.csv' with the actual path to your CSV file
df = pd.read_csv('dataset.csv')

# Assuming your CSV has columns: 'u_k_x', 'u_k_y', 'u_k_theta', 'X', 'Y', 'Theta', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta'
X_train = df[['u_k_x', 'u_k_y', 'u_k_theta']].values
y_train = df[['X', 'Y', 'Theta', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta']].values

# Define the neural network model
model = tf.keras.Sequential([
    layers.Input(shape=(3,)),  # Assuming input is ['u_k_x', 'u_k_y', 'u_k_theta']
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(6)  # Output has 6 dimensions: ['X', 'Y', 'Theta', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta']
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=1)

# Example input data for prediction
input_data = df[['u_k_x', 'u_k_y', 'u_k_theta']].iloc[0].values.reshape(1, -1)  # Use the first row as an example
# Predict next position and covariance
predictions = model.predict(input_data)

# Print the predictions
print("Predicted next position and covariance:")
print(predictions)
model.save('trained_model.keras')