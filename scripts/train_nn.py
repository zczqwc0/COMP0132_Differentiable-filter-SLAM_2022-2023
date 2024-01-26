import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset from CSV file
# Replace 'dataset.csv' with the actual path to your CSV file
df = pd.read_csv('dataset.csv')

# CSV has columns: 'Step', 'pose_x', 'pose_y', 'pose_theta', 'u_k_x', 'u_k_y', 'u_k_theta', 'relative_X', 'relative_Y', 'relative_Theta', 'Obs_dist', 'Obs_tetha', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle'
X = df[['pose_x', 'pose_y', 'pose_theta', 'u_k_x', 'u_k_y', 'u_k_theta']].values
y = df[['relative_X', 'relative_Y', 'relative_Theta', 'Obs_dist', 'Obs_tetha', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle']].values

# Normalize the data
scaler_X = StandardScaler().fit(X)
X_scaled = scaler_X.transform(X)

# Split into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    layers.Input(shape=(6,)),  # Input is ['pose_x', 'pose_y', 'pose_theta', 'u_k_x', 'u_k_y', 'u_k_theta']
    layers.Dense(64, activation='relu'), # Consider regularization techniques: kernel_regularizer=regularizers.l2(0.01)
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)  # Output has 10 dimensions: ['relative_X', 'relative_Y', 'relative_Theta', 'Obs_dist', 'Obs_tetha', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle']
])

# Compile the model
learning_rate = 0.01 # Experiment with different values (e.g. 0.01, 0.001, 0.0005)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer='adam', loss='mean_squared_error')

class TrainingMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        # Store the loss values
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        # Print the metrics
        print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
        for metric_name, value in logs.items():
            print(f"{metric_name}: {value:.4f}")

    def on_train_end(self, logs=None):
        # Plot loss graph
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='Training Loss')
        plt.plot(range(1, len(self.val_loss) + 1), self.val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

# Create an instance of the callback
metrics_callback = TrainingMetricsCallback()

# Train the model with the callback
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=1, callbacks=[TrainingMetricsCallback()])

model.save('trained_model.keras')