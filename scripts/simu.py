import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import csv

# Dynamic model parameters
delta_t = 1  # time interval

# Prompt user for noise type
dynamic_noise_type = input("Enter 'gaussian' or 'uniform' for dynamic model noise: ").lower()
if dynamic_noise_type not in ['gaussian', 'uniform']:
    raise ValueError("Invalid noise type. Please enter 'gaussian' or 'uniform'.")

Q_std_dev = np.diag([0.1, 0.1, 0.01])  # default values for Gaussian noise
Q_uniform_range  = 0.1  # set a common standard deviation for uniform noise

# Observation model parameters
observation_noise_type = input("Enter 'gaussian' or 'uniform' for observation model noise: ").lower()
if observation_noise_type not in ['gaussian', 'uniform']:
    raise ValueError("Invalid noise type. Please enter 'gaussian' or 'uniform'.")

R_std_dev = np.diag([0.1, 0.01])  # default values for Gaussian noise
R_uniform_range = 0.1  # Uniform noise range for observation model

 # Determine the CSV file name based on noise types
if dynamic_noise_type == 'gaussian' and observation_noise_type == 'gaussian':
     csv_filename = 'dataset.csv'
elif dynamic_noise_type == 'uniform' and observation_noise_type == 'uniform':
     csv_filename = 'testset.csv'
else:
     raise ValueError("Inconsistent noise types. Both should be either 'gaussian' or 'uniform'.")

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

def save_to_csv(filename, acc_step, trajectory, control_inputs, relative_poses_list, obs_list, covariances, write_header):
    mode = 'w' if write_header else 'a'  # Open in write mode if writing header, else append mode
    with open(filename, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        if write_header : 
            writer.writerow(['Step', 'pose_x', 'pose_y', 'pose_theta', 'u_k_x', 'u_k_y', 'u_k_theta', 'relative_X', 'relative_Y', 'relative_Theta', 'Obs_dist', 'Obs_tetha', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle'])
        # Write data
        for step, (st, relative_pose, obs, covariance, control_inputs, traj) in enumerate(zip(acc_step, relative_poses_list, obs_list, covariances, control_inputs, trajectory)):
            
            # Format covariance values with 4 decimal places
            formatted_covariance_x = f"{covariance[0][0]:.4f}"
            formatted_covariance_y = f"{covariance[0][1]:.4f}"
            formatted_covariance_theta = f"{covariance[0][2]:.4f}"
            formatted_covariance_d = f"{covariance[1][0]:.4f}"
            formatted_covariance_t = f"{covariance[1][1]:.4f}"

            obs_d = f"{obs[0]:.4f}"
            obs_t = f"{obs[1]:.4f}"

            control_inputs_x = f"{control_inputs[0]:.4f}"
            control_inputs_y = f"{control_inputs[1]:.4f}"
            control_inputs_theta = f"{control_inputs[2]:.4f}"

            relative_pose_x = f"{relative_pose[0]:.4f}"
            relative_pose_y = f"{relative_pose[1]:.4f}"
            relative_pose_t = f"{relative_pose[2]:.4f}"

            trajectory_x = f"{traj[0]:.4f}"
            trajectory_y = f"{traj[1]:.4f}"
            trajectory_theta = f"{traj[2]:.4f}"

            writer.writerow([st, trajectory_x, trajectory_y, trajectory_theta, control_inputs_x, control_inputs_y, control_inputs_theta , relative_pose_x, relative_pose_y, relative_pose_t,  obs_d , obs_t , formatted_covariance_x, formatted_covariance_y, formatted_covariance_theta,formatted_covariance_d, formatted_covariance_t])

# Dynamic model function
def dynamic_model(x_k, psi_k, u_k, v_k):
    if dynamic_noise_type == 'gaussian':
        v_k = np.random.multivariate_normal([0, 0, 0], Q_std_dev)
    else: # uniform noise
        v_k = np.random.uniform(-Q_uniform_range, Q_uniform_range, size=(3,))
    
    M_psi_k = rotation_matrix(psi_k)
    a = M_psi_k @ (u_k.T + v_k.T)
    c = delta_t * a
    d = x_k + c
    d[2] = normalize_theta(d[2])
    return d

# Observation model function
def observation_model(x_i, y_i, x_k, y_k, phi_k, w_k):
    if observation_noise_type == 'gaussian':
        w_k = np.random.multivariate_normal([0, 0], R_std_dev)
    else:
        w_k = np.random.uniform(-R_uniform_range, R_uniform_range, size=(2,))
    
    r_k_i = np.sqrt((x_i - x_k)**2 + (y_i - y_k)**2)  + w_k[0]
    beta_k_i = np.arctan2((y_i - y_k), (x_i - x_k)) - phi_k + w_k[1]  
    return np.array([r_k_i, beta_k_i])

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

# Initialise parameters
runs  = int(input("Enter the number of simulations (runs): ")) # 120 times run for training data, and 24 times for test data
num_steps = int(input("Enter the number of steps per simulation (num_steps): ")) # 500 steps in each trajectory
num_landmarks = 200  # Number of landmarks

# Define maximum x and y for landmark scatter
max_x = 40
max_y = 40

header_written = False

acc_step = 0
sim_step = 0 
# Main loop for simulation
for simulation in range(runs):
    
    # simulation times
    print(sim_step) 
    sim_step = sim_step +1 

    # Initialize lists to store data for csv files
    relative_poses_list = []  # To store robot poses
    covariances_list = [] # To store covariances
    control_inputs_list = []  # To store control inputs
    obs_list = [] # To store observations
    landmark_pos_list = []  # To store landmark positions
    true_range_list = []  # To store true range measurements between landmarks and robot's pose
    acc = [] # To store step number
    trajectory = [] # To store robot pose

    # Initial robot pose
    initial_phi = np.random.uniform(0, 2 * np.pi)
    x_k = np.array([0, 0, initial_phi])
    trajectory = [x_k.copy()]  # Initialize with a copy of the initial state
    prev_pose = x_k.copy()

    # Set a seed for reproducibility of fixed landmark positions only
    rng_state = np.random.get_state() # Save the current state of the random number generator
    # Set a seed for reproducibility of landmark positions only
    np.random.seed(42)
    # Generate N fixed landmarks throughout the trajectory within the specified limits
    landmarks = np.random.uniform(low=-max_x, high=max_x, size=(num_landmarks, 1))
    landmarks = np.hstack([landmarks, np.random.uniform(low=-max_y, high=max_y, size=(num_landmarks, 1))])
    # Restore the previous state of the random number generator
    np.random.set_state(rng_state)

    # Store landmark indices and positions
    for idx, landmark in enumerate(landmarks):
        landmark_pos_list.append((idx, landmark))

    # Uncomment the following lines to visualize animation of the robot motion and landmarks
    '''
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
    '''

    for step in range(num_steps):

        # Define velocities in x, y, and theta directions
        velocity_x = np.random.uniform(-1.0, 1.0)  # Random velocity in x direction
        velocity_y = np.random.uniform(-1.0, 1.0)  # Random velocity in y direction
        angular_velocity = np.random.uniform(-0.1, 0.1)  # Random rotational velocity

        # Generate control input and process noise
        u_k = np.array([velocity_x, velocity_y, angular_velocity])
        v_k = np.random.multivariate_normal([0, 0, 0], Q_std_dev)  # Ensure v_k has 3 components
        w_k = np.random.multivariate_normal([0, 0], R_std_dev)  # Ensure w_k has 3 components
        # Dynamic model
        x_k = dynamic_model(x_k, x_k[2], u_k, v_k)
        trajectory.append(x_k.copy())  # Append a copy of the updated state

        # Find all landmark within the detection radius of 3 units
        detection_radius = 5
        land_detect = 0

        for landmark in landmarks:
            distance = np.sqrt((landmark[0] - x_k[0])**2 + (landmark[1] - x_k[1])**2)
            if distance <= detection_radius:
                # If within radius, use the observation model
                r_k_i, beta_k_i = observation_model(landmark[0], landmark[1], x_k[0], x_k[1], x_k[2], w_k)

                land_detect = 1

                # # Plot connection from robot to landmark
                # con = ConnectionPatch(xyA=(x_k[0], x_k[1]), xyB=(landmark[0], landmark[1]),
                #                       coordsA="data", coordsB="data", color="blue", arrowstyle="->", linewidth=2)
                # ax.add_patch(con)

        # Uncomment the following lines to visualize animation of the robot motion and landmarks
        # line.set_data([state[0] for state in trajectory], [state[1] for state in trajectory]) # Update plot data
        # plt.pause(0.1) # Pause for a short duration to simulate real-time movement

        relative_pose = np.array([x_k[0] - prev_pose[0], x_k[1] - prev_pose[1], normalize_theta(x_k[2] - prev_pose[2])])
        
        if land_detect == 1 :
            obs_list.append((r_k_i, beta_k_i))
            covariances_list.append((v_k, w_k))
        else :
            obs_list.append((0, 0))
            covariances_list.append((np.zeros_like(v_k), np.zeros_like(w_k)))

            acc
        relative_poses_list.append(relative_pose)
        
        control_inputs_list.append(u_k)

        prev_pose = x_k.copy()

        acc.append(acc_step)
        acc_step  = acc_step + 1

    # save csv files
    if not header_written:
        save_to_csv(csv_filename, acc, trajectory, control_inputs_list, relative_poses_list , obs_list, covariances_list,  write_header=True)
        header_written = True
    else:
        save_to_csv(csv_filename, acc, trajectory, control_inputs_list, relative_poses_list , obs_list, covariances_list,  write_header=False)

    # plt.close()