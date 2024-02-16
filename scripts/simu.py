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
Q_uniform_range = 0.1 # range for uniform noise

# Observation model parameters
observation_noise_type = input("Enter 'gaussian' or 'uniform' for observation model noise: ").lower()
if observation_noise_type not in ['gaussian', 'uniform']:
    raise ValueError("Invalid noise type. Please enter 'gaussian' or 'uniform'.")

R_std_dev = np.diag([0.1, 0.01])  # default values for Gaussian noise
R_uniform_range = 0.1 # range for uniform noise

# Determine the CSV file name based on noise types
if dynamic_noise_type == 'gaussian' and observation_noise_type == 'gaussian':
     csv_filename = 'Dataset/dataset.csv'
elif dynamic_noise_type == 'uniform' and observation_noise_type == 'uniform':
     csv_filename = 'Dataset/testset.csv'
else:
     raise ValueError("Inconsistent noise types. Both should be either 'gaussian' or 'uniform'.")

# Rotation matrix function
def rotation_matrix(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])

# Function to normalize theta to be within [-pi, pi]
def normalize_theta(theta):
    if theta < -np.pi:
        theta += 2 * np.pi
    elif theta > np.pi:
        theta -= 2 * np.pi
    return theta

# Save data to a CSV file
def save_to_csv(filename, acc_step, control_inputs,abs_pose, poses, obs_list, covariances, write_header):
    mode = 'w' if write_header else 'a'  # Open in write mode if writing header, else append mode
    with open(filename, mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        if write_header : 
            writer.writerow(['Step', 'u_k_x', 'u_k_y','u_k_theta', 'x','y','theta',  'relative_X', 'relative_Y', 'relative_Theta', 'Obs_dist', 'Obs_tetha', 'Covariance_X', 'Covariance_Y', 'Covariance_Theta','Covariance_dis', 'Covariance_angle'])
        # Write data
        for step, (st,abs_pose, pose,obs,  covariance, control_inputs) in enumerate(zip(acc_step,abs_pose, poses,obs_list, covariances, control_inputs)):
             
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

            abs_pose_x = f"{abs_pose[0]:.4f}"
            abs_pose_y = f"{abs_pose[1]:.4f}"
            abs_pose_theta = f"{abs_pose[2]:.4f}"

            pose_x = f"{pose[0]:.4f}"
            pose_y = f"{pose[1]:.4f}"
            pose_t = f"{pose[0]:.4f}"
            
            writer.writerow([st, control_inputs_x, control_inputs_y, control_inputs_theta, abs_pose_x,abs_pose_y ,abs_pose_theta, pose_x, pose_y, pose_t,  obs_d , obs_t , formatted_covariance_x, formatted_covariance_y, formatted_covariance_theta,formatted_covariance_d, formatted_covariance_t])

# Dynamic model function
def dynamic_model(x_k, psi_k, u_k, v_k):

    max_delta_x = 0.5  # max change in x
    max_delta_y = 0.5  # max change in y
    max_delta_theta = np.radians(30)

    # Noise application
    if dynamic_noise_type == 'gaussian':
        v_k = np.random.multivariate_normal([0, 0, 0], Q_std_dev)
    else: # uniform noise
        v_k = np.random.uniform(-Q_uniform_range, Q_uniform_range, size=(3,))
    
    # Transformation matrix based on current orientation
    M_psi_k = rotation_matrix(psi_k)
    a = M_psi_k @ (u_k.T + v_k.T)
    c = delta_t * a  # Change due to control input and noise

    # New state calculation
    d = x_k + c
    d[2] = normalize_theta(d[2])  # Normalize orientation

    # Apply limits to changes in x, y, and theta
    d[0] = np.clip(d[0], x_k[0] - max_delta_x, x_k[0] + max_delta_x)
    d[1] = np.clip(d[1], x_k[1] - max_delta_y, x_k[1] + max_delta_y)
    d[2] = np.clip(d[2], normalize_theta(psi_k - max_delta_theta), normalize_theta(psi_k + max_delta_theta))

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

# Save loop closures to a CSV file
def save_loop_closures_to_csv(loop_closures, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index1", "Index2", "X", "Y", "Theta"])
        for loop_closure in loop_closures:
            writer.writerow(loop_closure)

# Function to handle zooming in and out of the plot
def on_scroll(event, ax):
    if event.button == 'up':
        zoom_factor = 1.1
    else:
        zoom_factor = 1 / 1.1

    ax.set_xlim(ax.get_xlim()[0] * zoom_factor, ax.get_xlim()[1] * zoom_factor)
    ax.set_ylim(ax.get_ylim()[0] * zoom_factor, ax.get_ylim()[1] * zoom_factor)

# Function to load landmarks from a file
def load_landmarks(filename):
    return np.loadtxt(filename, delimiter=',')

# Usage
filename = "landmarks.txt"
landmarks = load_landmarks(filename)

# Initialise parameters
runs  = int(input("Enter the number of simulations (runs): ")) # 120 times run for training data, and 24 times for test data
num_steps = int(input("Enter the number of steps per simulation (num_steps): ")) # 500 steps in each trajectory
 
header_written = False

acc_step = 0
sim_step = 0 

# Main loop for simulation
for simulation in range(runs):
    print( sim_step)
    sim_step = sim_step +1 
    
    # Initialize lists to store data for csv files
    poses_list = []  # To store transoformations
    abs_pose = [] # To store absolute robot poses
    covariances_list = [] # To store covariances
    control_inputs_list = []  # To store control inputs
    obs_list = [] # To store observations
    acc = [] # To store the step number
    
    # Generate random initial values for phi
    initial_phi = np.random.uniform(0, 2 * np.pi)
    
    # Initialize state vector with random steering angles
    x_k = np.array([0, 0, initial_phi])
    trajectory = [x_k.copy()]  # Initialize with a copy of the initial state
    prev_pose = x_k.copy()

    # Uncomment the following lines to visualize animation of the robot motion and landmarks and also at lines 236 and 237
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

    fig.canvas.mpl_connect('scroll_event', lambda event: on_scroll(event, ax))
    '''
    
    for step in range(num_steps):

        # Define velocities in x, y, and theta directions
        velocity_x = np.random.uniform(0.0, 2.0)  # Random velocity in x direction
        velocity_y = np.random.uniform(0.0, 2.0)  # Random velocity in y direction
        angular_velocity = np.random.uniform(0.0, 0.2)  # Random rotational velocity

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

                """
                # # Plot connection from robot to landmark
                con = ConnectionPatch(xyA=(x_k[0], x_k[1]), xyB=(landmark[0], landmark[1]),
                                      coordsA="data", coordsB="data", color="blue", arrowstyle="->", linewidth=2)
                ax.add_patch(con)
                """

        # Uncomment the following lines to visualize animation of the robot motion and landmarks
        # line.set_data([state[0] for state in trajectory], [state[1] for state in trajectory]) # Update plot data
        # plt.pause(0.1) # Pause for a short duration to simulate real-time movement

        relative_pose = np.array([x_k[0] - prev_pose[0], x_k[1] - prev_pose[1], normalize_theta(x_k[2] - prev_pose[2])])
        abs_po = np.array([x_k[0],x_k[1],x_k[2]])
        if land_detect == 1 :
            obs_list.append((r_k_i, beta_k_i))
            covariances_list.append((v_k, w_k))
        else :
            obs_list.append((0, 0))
            covariances_list.append((np.zeros_like(v_k), np.zeros_like(w_k)))

            acc
        poses_list.append(relative_pose)
        abs_pose.append(abs_po)
        control_inputs_list.append(u_k)

        prev_pose = x_k.copy()

        acc.append(acc_step)
        acc_step  = acc_step + 1
        
    # save csv files
    if not header_written:
        save_to_csv(csv_filename, acc, control_inputs_list, abs_pose, poses_list , obs_list, covariances_list,  write_header=True)
        header_written = True
    else:
        save_to_csv(csv_filename, acc, control_inputs_list,abs_pose,  poses_list , obs_list, covariances_list,  write_header=False)

    # plt.close()