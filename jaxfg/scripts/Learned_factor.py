from typing import Tuple
import jaxlie
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


def simulate_step(current_state: jaxlie.SE2, control: jaxlie.SE2) -> Tuple[jaxlie.SE2, jnp.array]:
    """
    Simulates a single step of the robot's motion.

    Args:
        current_state: Current state of the robot as a jaxlie.SE2 object.
        control: Control input as a jaxlie.SE2 object (without noise).
        
    Returns:
        new_state: New state of the robot after applying the control input and noise.
        noise: Noise added to the control input as a jnp.array.
    """

    # Define noise standard deviation
    noise_std = jnp.array([0.03, 0.03, 0.01])

    # Generate noise from a normal distribution
    noise = jnp.array([np.random.normal(0, std) for std in noise_std])

    # Predicted new state without noise
    predicted_new_state = current_state.multiply(control)

    # Apply noise to the control input
    T_noise = jaxlie.SE2.exp(noise)
    new_state = predicted_new_state.multiply(T_noise)

    # Compute the difference between the predicted and actual new state to get the noise
    noise_vector = (predicted_new_state.inverse().multiply(new_state)).log()

    return new_state, noise_vector

def generate_data(num_steps: int, step_size: float) -> Tuple[np.ndarray, np.ndarray]:
    initial_states, final_states, noises = [], [], []
    current_state = jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)
    control = jaxlie.SE2.from_xy_theta(step_size, 0.0, 0.0)
    for _ in range(num_steps):
        new_state, noise_vector = simulate_step(current_state, control)
        initial_states.append(current_state.parameters())
        final_states.append(new_state.parameters())
        noises.append(noise_vector)
        current_state = new_state
    return (
        np.hstack((initial_states, final_states)),
        np.array(noises)
    )

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(8,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def plot_noises(actual_noises, predicted_noises):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    titles = ['X Noise', 'Y Noise', 'Theta Noise']
    for i in range(3):
        axs[i].plot(actual_noises[:, i], label='Actual Noise')
        axs[i].plot(predicted_noises[:, i], label='Predicted Noise', linestyle='dashed')
        axs[i].set_title(titles[i])
        axs[i].legend()
    plt.tight_layout()
    plt.show()

def main():
    # Generate Data
    X, y = generate_data(num_steps=10000, step_size=1)

    # # Using a sequential split
    # train_ratio = 0.8
    # val_ratio = 0.1

    # num_train = int(len(X) * train_ratio)
    # num_val = int(len(X) * val_ratio)

    # X_train, y_train = X[:num_train], y[:num_train]
    # X_val, y_val = X[num_train:num_train+num_val], y[num_train:num_train+num_val]
    # X_test, y_test = X[num_train+num_val:], y[num_train+num_val:]

    # Split Data randomly, training, validation, and test sets with the ratio 8:1:1
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Train Model
    model = create_model()
    model.summary()
    # print("Data point:", X_train.shape[0])
    model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
    
    # Evaluate Model
    print(f'Test Loss: {model.evaluate(X_test, y_test)}')

    # Predict and Plot
    predicted_noises = model.predict(X_test)
    plot_noises(y_test, predicted_noises)


if __name__ == '__main__':
    main()


