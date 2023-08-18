import jaxlie
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

class LearnedProcessNoise:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(3,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def train(self, X, y, epochs=50, test_size=0.2):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
        test_loss = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {test_loss}')


    def predict_noise(self, displacement):
        displacement = np.array(displacement) 
        return self.model.predict(displacement.reshape(1, -1))

    def save(self, path):
        self.model.save(path)
        print('Model saved.')

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
