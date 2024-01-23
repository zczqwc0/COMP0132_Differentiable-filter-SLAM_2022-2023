import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import csv

# Dynamic model parameters
delta_t = 1  # time interval
Q_std_dev = np.diag([0.1, 0.1, 0.01])  # process noise covariance

# Observation model parameters
R_std_dev = np.diag([0.1, 0.01])  # observation noise covariance

# Rotation matrix function
def rotation_matrix(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

def normalize_theta(theta):
    if theta < -np.pi:
        theta += 2 * np.pi
    elif theta > np.pi:
        theta -= 2 * np.pi
    return theta

def save_to_csv(filename, poses, covariances, control_inputs, write_header):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        if write_header : 
            writer.writerow(['Step', 'u_k_x', 'u_k_y', 'u_k_theta', 'relative_displacement_x', 'relative_displacement_x', 'relative_displacement_theta', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta'])
        # Write data
        for step, (pose, covariance, control_inputs) in enumerate(zip(poses, covariances, control_inputs)):
            # Format covariance values with 4 decimal places
            formatted_covariance_x = f"{covariance[0]:.4f}"
            formatted_covariance_y = f"{covariance[1]:.4f}"
            formatted_covariance_theta = f"{covariance[2]:.4f}"
            writer.writerow([step, pose[0], pose[1], pose[2], formatted_covariance_x, formatted_covariance_y, formatted_covariance_theta, control_inputs[0], control_inputs[1], control_inputs[2]])

# Dynamic model function
def dynamic_model(x_k, psi_k, u_k, v_k):
    M_psi_k = rotation_matrix(psi_k)
    a = M_psi_k @ (u_k.T + v_k.T)
    c = delta_t * a
    d = x_k + c
    d[2] = normalize_theta(d[2])
    return d

# Observation model function
def observation_model(x_i, y_i, x_k, y_k, phi_k, R_std_dev):
    r_k_i = np.sqrt((x_i - x_k)**2 + (y_i - y_k)**2)+ np.random.normal(0, R_std_dev[0])
    beta_k_i = np.arctan2((y_i - y_k), (x_i - x_k)) - phi_k+ np.random.normal(0, R_std_dev[1])
    return np.array([r_k_i, beta_k_i])

def loop_closures_detection(poses, tolerance= 0.05):
    loop_closures = []
    for i in range(len(poses) - 1):
        for j in range(i + 1, len(poses)):
            # Check if the poses are the same within a tolerance
            if np.all(np.abs(poses[i] - poses[j]) < tolerance):
                loop_closure_data = (i, j, *poses[i])
                loop_closures.append(loop_closure_data)
    return loop_closures

def save_loop_closures_to_csv(loop_closures, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index1", "Index2", "X", "Y", "Theta"])
        for loop_closure in loop_closures:
            writer.writerow(loop_closure)

def on_scroll(event, ax):
    if event.button == 'up':
        zoom_factor = 1.1
    else:
        zoom_factor = 1 / 1.1

    ax.set_xlim(ax.get_xlim()[0] * zoom_factor, ax.get_xlim()[1] * zoom_factor)
    ax.set_ylim(ax.get_ylim()[0] * zoom_factor, ax.get_ylim()[1] * zoom_factor)

# Visualization parameters
# In this experiment, we will run the simulation for 500 steps in each trajectory with 120 times to collect 60000 training data.
runs  = int(input("Enter the number of simulations (runs): "))
num_steps = int(input("Enter the number of steps per simulation (num_steps): "))
num_landmarks = 10  # Number of landmarks

# Define maximum x and y for landmark scatter
max_x = 8
max_y = 8

header_written = False

# Main loop for simulation
for simulation in range(runs):
    # Initialize lists to store data for csv files
    poses_list = []  # To store robot poses
    covariances_list = [] # To store covariances
    control_inputs_list = []  # To store control inputs

    initial_phi = np.random.uniform(0, 2 * np.pi)
    angular_velocity = np.random.uniform(0.05, 0.2)

    # Generate random initial values for phi and angular velocity
    x_k = np.array([0, 0, initial_phi])
    trajectory = [x_k.copy()]  # Initialize with a copy of the initial state
    prev_pose = x_k.copy()

    # Save the current state of the random number generator
    rng_state = np.random.get_state()
    # Set a seed for reproducibility of landmark positions only
    np.random.seed(42)
    # Generate N fixed landmarks throughout the trajectory within the specified limits
    landmarks = np.random.uniform(low=-max_x, high=max_x, size=(num_landmarks, 1))
    landmarks = np.hstack([landmarks, np.random.uniform(low=-max_y, high=max_y, size=(num_landmarks, 1))])
    # Restore the previous state of the random number generator
    np.random.set_state(rng_state)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Robot Movement Over Time with Landmarks and Distances')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True)

    line, = ax.plot(trajectory[0][0], trajectory[0][1], label='Robot Trajectory', marker='o')
    scatter_landmarks = ax.scatter(landmarks[:, 0], landmarks[:, 1], color='red', marker='*', label='Landmarks')

    min_x, max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    min_y, max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    ax.set_xlim(min_x - 5, max_x + 5)
    ax.set_ylim(min_y - 5, max_y + 5)
    fig.canvas.draw()

    # Enable interactive zoom
    fig.canvas.mpl_connect('scroll_event', lambda event: on_scroll(event, ax))

    for step in range(num_steps):

        # Define velocities in x, y, and theta directions
        velocity_x = np.random.uniform(-1.0, 1.0)  # Random velocity in x direction
        velocity_y = np.random.uniform(-1.0, 1.0)  # Random velocity in y direction
        angular_velocity = np.random.uniform(-0.1, 0.1)  # Random rotational velocity

        # Generate control input and process noise
        u_k = np.array([velocity_x, velocity_y, angular_velocity])
        v_k = np.random.multivariate_normal([0, 0, 0], Q_std_dev)  # Ensure v_k has 3 components

        # Dynamic model
        x_k = dynamic_model(x_k, x_k[2], u_k, v_k)
        trajectory.append(x_k.copy())  # Append a copy of the updated state

        # Find all landmark within the detection radius of 3 units
        detection_radius = 5
        for landmark in landmarks:
            distance = np.sqrt((landmark[0] - x_k[0])**2 + (landmark[1] - x_k[1])**2)
            if distance <= detection_radius:
                # If within radius, use the observation model
                r_k_i, beta_k_i = observation_model(landmark[0], landmark[1], x_k[0], x_k[1], x_k[2], R_std_dev)

                # # Plot connection from robot to landmark
                # con = ConnectionPatch(xyA=(x_k[0], x_k[1]), xyB=(landmark[0], landmark[1]),
                #                       coordsA="data", coordsB="data", color="blue", arrowstyle="->", linewidth=2)
                # ax.add_patch(con)

        # Update plot data
        line.set_data([state[0] for state in trajectory], [state[1] for state in trajectory])

        relative_pose = np.array([x_k[0] - prev_pose[0], x_k[1] - prev_pose[1], normalize_theta(x_k[2] - prev_pose[2])])
        
        poses_list.append(relative_pose)
        covariances_list.append(v_k)
        control_inputs_list.append(u_k)

        prev_pose = x_k.copy()

        # Pause for a short duration to simulate real-time
        plt.pause(0.1)

    # save csv files
    loop_closures = loop_closures_detection(poses_list)
    print(f"Found {len(loop_closures)} loop closures.")
    if not header_written:
        save_loop_closures_to_csv(loop_closures, 'loop_closures.csv')
        save_to_csv('dataset.csv', poses_list, covariances_list, control_inputs_list, write_header=True)
        header_written = True
    else:
        # Append data without writing header
        save_loop_closures_to_csv(loop_closures, 'loop_closures.csv')
        save_to_csv('dataset.csv', poses_list, covariances_list, control_inputs_list, write_header=False)

    plt.close()