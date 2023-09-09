import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

class NoiseAutoencoder:
    def __init__(self):
        self.encoder = self.create_encoder()
        self.autoencoder = self.create_model()

    def compute_covariance(self, noise):
        noise_mean = tf.reduce_mean(noise, axis=0, keepdims=True)
        noise_centered = noise - noise_mean
        covariance_matrix = tf.matmul(noise_centered, noise_centered, transpose_a=True) / tf.cast(tf.shape(noise)[0], tf.float32)
        return covariance_matrix

    def custom_loss(self, y_true, y_pred):
    # Covariance matrix of the predicted values (noise)
        noise = y_pred
        covariance_matrix = self.compute_covariance(noise)
        
        # Special diagonal matrix
        average_diagonal_value = tf.reduce_mean(tf.linalg.diag_part(covariance_matrix))
        special_diagonal = tf.eye(noise.shape[-1]) * average_diagonal_value
        
        # Loss based on the difference between the covariance matrix and the special diagonal matrix
        matrix_diff = covariance_matrix - special_diagonal
        covariance_loss = tf.norm(matrix_diff, ord='fro', axis=[-1, -2])
        
        return covariance_loss

    
        # Encoder with dense layers (Fully connected)
    def create_encoder(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(9,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(3)
        ])
        return model
    
        # Encoder with 1D convolution
    # def create_encoder(self):
    #     model = tf.keras.Sequential([
    #         tf.keras.layers.Reshape((9, 1), input_shape=(9,)),  # Reshape for 1D convolution
    #         tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
    #         tf.keras.layers.BatchNormalization(),
    #         tf.keras.layers.Conv1D(16, 3, activation='relu', padding='same'),
    #         tf.keras.layers.BatchNormalization(),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(3)
    #     ])
    #     return model  

    def create_model(self):
        encoder = self.create_encoder()
        model = tf.keras.Sequential([encoder])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss=self.custom_loss)
        return model


    def train(self, X, epochs=50, test_size=0.2):
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
        self.autoencoder.fit(X_train, X_train, epochs=epochs, validation_data=(X_test, X_test))

    def predict_noise(self, data):
        data = np.array(data)
        return self.autoencoder.predict(data)


    def save(self, path):
        self.autoencoder.save(path)
        print('Model saved.')

    def load(self, path):
        self.autoencoder = tf.keras.models.load_model(path, custom_objects={'custom_loss': self.custom_loss})