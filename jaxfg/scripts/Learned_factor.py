from typing import Tuple
import jaxlie
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# np.random.seed(42)
# tf.random.set_seed(42)

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
    noise_vector = (new_state.multiply(predicted_new_state.inverse())).log()


    return new_state, noise_vector


step_size = 1

num_step = 500

# Arrays to hold the collected data
initial_states = []
final_states = []
noises = []


# Simulate multiple steps
current_state = jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)
control = jaxlie.SE2.from_xy_theta(step_size, 0.0, 0.0)
for _ in range(num_step):
    new_state, noise_vector = simulate_step(current_state, control)
    initial_states.append(current_state)
    final_states.append(new_state)
    noises.append(noise_vector)
    current_state = new_state  # Update the current state for the next iteration

# Convert SE2 objects to arrays
initial_states = np.array([state.parameters() for state in initial_states])
final_states = np.array([state.parameters() for state in final_states])
noises = np.array(noises)

print("Initial State:", initial_states)
print("Final State", final_states)
print("Noise:", noises)

# Data preprocessing: flatten the data and combine initial and final states
X = np.hstack((initial_states, final_states))
y = noises
# print ("Model Input (Intial state & Final state for each step): ", X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3)  # Output layer with 3 units for the 3 noise values
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
# Evaluate the model
val_loss = model.evaluate(X_val, y_val)
# print(f'Validation Loss: {val_loss}')

# List to store the predicted noise for each step
predicted_noises = []

# Iterate through each step
for initial_state, final_state in zip(initial_states, final_states):
    input_data = np.hstack((initial_state, final_state)).reshape(1, -1)  # Prepare input data for prediction
    predicted_noise = model.predict(input_data)  # Predict the noise using the trained model
    predicted_noises.append(predicted_noise[0])  # Store the predicted noise

# Convert the list to an array for further analysis
predicted_noises = np.array(predicted_noises)

# Print the predicted noises
print("Predicted Noises for Each Step:")
print(predicted_noises)


# Extracting the actual noise values for comparison
actual_noises = noises

# Concatenate the data horizontally
data_to_save = np.hstack((initial_states, final_states, noises, predicted_noises))

# Define the header for the CSV file
header = "init_x,init_y,init_theta,final_x,final_y,final_theta,noise_x,noise_y,noise_theta,predicted_noise_x,predicted_noise_y,predicted_noise_theta"

# # Save the data to a CSV file
# np.savetxt("robot_data.csv", data_to_save, delimiter=",", header=header, comments='')
# print("Data saved to robot_data.csv")

# Setting up the plots
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Plotting x noise values
axs[0].plot(actual_noises[:, 0], label='Actual Noise')
axs[0].plot(predicted_noises[:, 0], label='Predicted Noise', linestyle='dashed')
axs[0].set_title('X Noise')
axs[0].legend()

# Plotting y noise values
axs[1].plot(actual_noises[:, 1], label='Actual Noise')
axs[1].plot(predicted_noises[:, 1], label='Predicted Noise', linestyle='dashed')
axs[1].set_title('Y Noise')
axs[1].legend()

# Plotting theta noise values
axs[2].plot(actual_noises[:, 2], label='Actual Noise')
axs[2].plot(predicted_noises[:, 2], label='Predicted Noise', linestyle='dashed')
axs[2].set_title('Theta Noise')
axs[2].legend()

plt.tight_layout()
plt.show()






