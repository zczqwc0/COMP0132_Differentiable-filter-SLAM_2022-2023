from typing import Tuple
import jaxlie
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


def simulate_trajectory(current_state: jaxlie.SE2, speed: float, angular_velocity: float, delta_T: float) -> Tuple[jaxlie.SE2, jaxlie.SE2, jnp.array]:
    """
    Simulates a single step of the robot's motion based on the process model.

    Args:
        current_state: Current state of the robot as a jaxlie.SE2 object.
        speed: Speed of the vehicle.
        angular_velocity: Angular velocity of the vehicle.
        delta_T: Length of the prediction interval.
        
    Returns:
        ground_truth_state: Ground truth state of the robot after applying the process model input and noise.
        noise: Noise added to the process_model input as a jnp.array.
    """

    # The process noise vk​ is zero mean, Gaussian, and additive on all three dimensions.
    # Define noise standard deviation
    noise_std = jnp.array([0.03, 0.03, 0.01])
    # Generate noise from a normal distribution
    noise = jnp.array([np.random.normal(0, std) for std in noise_std])

    # Control input
    u_k = jnp.array([speed, 0, angular_velocity])

    # Rotation matrix M(psi)
    theta = current_state.rotation().log()[0]
    M = jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0],
                   [jnp.sin(theta), jnp.cos(theta), 0],
                   [0, 0, 1]])

    # State update
    delta_state = delta_T * jnp.dot(M, (u_k + noise))
    T_delta = jaxlie.SE2.exp(delta_state)

    # Predicted new state
    noisy_state = current_state.multiply(T_delta)

    # Compute the difference between the current and actual new state to get the noise
    noise_vector = (current_state.inverse().multiply(noisy_state)).log()

    # Noise-free state update
    delta_state_noise_free = delta_T * jnp.dot(M, u_k)
    T_delta_noise_free = jaxlie.SE2.exp(delta_state_noise_free)
    ground_truth_state = current_state.multiply(T_delta_noise_free)

    return ground_truth_state,noisy_state, noise_vector


def generate_data(num_steps: int, speed: float, angular_velocity: float, delta_T: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative_displacements, ground_truth_displacements, errors = [], [], []
    current_state = jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)
    for _ in range(num_steps):
        ground_truth_state, noisy_state, noise_vector = simulate_trajectory(current_state, speed, angular_velocity, delta_T)

        # Calculate the relative displacement for noisy state
        displacement = current_state.inverse().multiply(noisy_state).log()
        relative_displacements.append(displacement)
        
        # Calculate the relative displacement for ground truth
        gt_displacement = current_state.inverse().multiply(ground_truth_state).log()
        ground_truth_displacements.append(gt_displacement)

        errors.append(noise_vector)
        current_state = ground_truth_state

    return (
        np.vstack(relative_displacements),
        np.vstack(ground_truth_displacements),
        np.vstack(errors)
    )
def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    variance_penalty = tf.square(tf.math.reduce_std(y_pred) - 1.0)
    return mse + variance_penalty


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(3,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(3)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=custom_loss)
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
    
    # Assuming the error e to be Gaussian distributed with zero mean and unit covariance matrix
    # Since Σ is the identity matrix, only need to minmize e'e
    
    # Define speed, angular_velocity, and delta_T
    speed = 1.0  
    angular_velocity = 0.0  
    delta_T = 1.0  

    # Generate Data
    X_noisy, X_ground_truth, y = generate_data(num_steps=10000, speed=speed, angular_velocity=angular_velocity, delta_T=delta_T)   

    # Split Data randomly, training, validation, and test sets with the ratio 8:1:1
    X_train, X_temp, y_train, y_temp = train_test_split(X_noisy, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Splitting the ground truth data
    X_train_gt, X_temp_gt = train_test_split(X_ground_truth, test_size=0.2, random_state=42)
    X_val_gt, X_test_gt = train_test_split(X_temp_gt, test_size=0.5, random_state=42)

    # Train Model
    model = create_model()
    model.summary()
    model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))
    
    # Evaluate Model
    print(f'Test Loss: {model.evaluate(X_test, y_test)}')

    # Compute the error e and e^T e
    e = y_test - model.predict(X_test)
    eTe = np.sum(e**2, axis=1)
    print("Average e^T e:", np.mean(eTe))

    
    
    # The values in "difference" are close to zero, especially for off-diagonal elements,
    # it indicates that the covariance matrix is close to the identity matrix.
    # Predict errors on the test set
    predicted_errors = model.predict(X_test)
    # Compute the sample covariance matrix
    cov_matrix = np.cov(predicted_errors, rowvar=False)
    print("Covariance matrix of predicted errors:\n", cov_matrix)
    identity_matrix = np.identity(cov_matrix.shape[0])
    difference = np.abs(cov_matrix - identity_matrix)
    print("Difference from identity matrix:\n", difference)
    
    # Compute the estimated state by subtracting the predicted errors from the noisy measurements
    estimated_states = X_test - predicted_errors
    # Compute the squared differences between the true state and the estimated state
    squared_differences = np.sum((estimated_states - X_test_gt)**2, axis=1)

    # Compute the MSE
    mse = np.mean(squared_differences)
    print("MSE between the mean of the belief and true state:", mse)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.title("Sample Covariance Matrix of Predicted Errors")
    plt.show()

if __name__ == '__main__':
    main()


