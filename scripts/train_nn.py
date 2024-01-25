import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset from CSV file
# Replace 'dataset.csv' with the actual path to your CSV file
df = pd.read_csv('dataset.csv')

# CSV has columns: 'Step', 'u_k_x', 'u_k_y', 'u_k_theta', 'relative_X', 'relative_Y', 'relative_Theta', 'Obs_dist', 'Obs_tetha', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle'
# Assuming your CSV has columns: 'Step', 'u_k_x', 'u_k_y', 'u_k_theta', 'relative_X', 'relative_Y', 'relative_Theta', 'Obs_dist', 'Obs_tetha', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle'
X = df[['u_k_x', 'u_k_y', 'u_k_theta', 'Obs_dist', 'Obs_tetha']].values
y = df[['relative_X', 'relative_Y', 'relative_Theta', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle']].values

# Split into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    layers.Input(shape=(5,)),  # Input is ['u_k_x', 'u_k_y', 'u_k_theta', 'Obs_dist', 'Obs_tetha']
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(8)  # Output has 8 dimensions: ['relative_X', 'relative_Y', 'relative_Theta', 'Obs_dist', 'Obs_tetha', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle']
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Create a callback to capture training metrics and plot graphs
class TrainingMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
        for metric_name, value in logs.items():
            print(f"{metric_name}: {value:.4f}")

# Train the model with the callback
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=1, callbacks=[TrainingMetricsCallback()])

model.save('trained_model.keras')