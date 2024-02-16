# Import necessary libraries for SLAM, numerical operations, and plotting
import gtsam
import numpy as np
import matplotlib.pyplot as plt
from robot import SimulatedEnvironment, SimulatedRobot, SLAMNNIntegrator
import matplotlib.lines as mlines

# Function to calculate the smallest difference between two angles
def angle_difference(angle1, angle2):
    diff = (angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi
    return diff

# Simulation parameters
num_landmarks_to_generate = 500

# Initialize simulated environment and robot
environment = SimulatedEnvironment(num_landmarks=num_landmarks_to_generate)
robot = SimulatedRobot(environment)

# Initialize SLAM Neural Network Integrator with a pre-trained model
model_path = 'trained_model.keras'
slam_nn_integrator = SLAMNNIntegrator(model_path)

# Main function for executing SLAM with real-time plotting
def create_slam_with_real_time_plotting_corrected(num_poses=50):
    # Initialize SLAM components: factor graph, initial estimates, and noise models
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1]))

    # Setup for real-time plotting
    plt.figure(figsize=(6, 6))
    plt.ion()

    # Initialize true and estimated poses dictionaries
    landmarks = environment.landmarks
    true_poses = {}
    estimated_poses = {}

    # Ask user if they want to use the neural network for motion prediction
    use_nn = input("Use neural network? (yes/no): ").strip().lower() == 'yes'
    # Determine plot title based on the use of neural network
    title_suffix = " with Neural Network" if use_nn else " without Neural Network"
    
    # Add a prior on the first pose
    first_pose = gtsam.Pose2(0.0, 0.0, 0.0)
    graph.add(gtsam.PriorFactorPose2(1, first_pose, prior_noise))
    initial_estimate.insert(1, first_pose)

    # Loop through the poses
    for i in range(1, num_poses + 2):
        if i == 1:
            new_pose = first_pose
        else:
            # Predict next pose based on previous pose and control inputs
            prev_pose = initial_estimate.atPose2(i-1)
            u_k = np.array([1, 1, 0.1])  # Example control inputs
            odometry = robot.dynamic_model(prev_pose.x(), prev_pose.y(), prev_pose.theta(), u_k)
            if use_nn:
                odometry, land, noise_cov = slam_nn_integrator.predict_motion_and_noise(u_k, prev_pose.x(), prev_pose.y(), prev_pose.theta())
            x_k2 = robot.sense_odometry()
            new_pose = gtsam.Pose2(x_k2[0], x_k2[1], x_k2[2])
            graph.add(gtsam.BetweenFactorPose2(i-1, i, odometry, odometry_noise))
            initial_estimate.insert(i, new_pose)

        # Simulate landmark observation and update graph
        for lm_id, lm_pos in landmarks.items():
            obs = robot.observation_model(lm_pos[0], lm_pos[1], new_pose.x(), new_pose.y(), new_pose.theta())
            if obs[0] < 10:
                graph.add(gtsam.BearingRangeFactor2D(i, lm_id, gtsam.Rot2(obs[1]), obs[0], measurement_noise))
                lnd_po = robot.estimate_landmark_position(new_pose, obs[1], obs[0])
                if initial_estimate.exists(lm_id):
                    updates = gtsam.Values()
                    updates.insert(lm_id, lnd_po)
                    initial_estimate.update(updates)
                else:
                    initial_estimate.insert(lm_id, lnd_po)

        # Update dictionaries with true and estimated poses
        true_poses[i] = [new_pose.x(), new_pose.y(), new_pose.theta()]

        # Real-time plotting of true poses and landmarks
        plt.clf()
        for j in range(1, i + 1):
            if j in initial_estimate.keys():
                pose = initial_estimate.atPose2(j)
                plt.plot(pose.x(), pose.y(), 'ko')  # Black circles for true poses
        for lm_id, lm_pos in landmarks.items():
            plt.plot(lm_pos[0], lm_pos[1], 'gx')

        plt.axis('equal')
        plt.draw()
        plt.pause(.001)

    # Optimize the factor graph after all observations
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()

    # Final update and plot of estimated poses and landmarks
    for i in range(1, num_poses + 2):
        if i in result.keys():
            estimated_pose = result.atPose2(i)
            estimated_poses[i] = [estimated_pose.x(), estimated_pose.y(), estimated_pose.theta()]
            plt.plot(estimated_pose.x(), estimated_pose.y(), 'r+')  # Plot estimated pose as red pluses

    for lm_id in landmarks.keys():
        if lm_id in result.keys():
            estimated_landmark = result.atPoint2(lm_id)
            plt.plot(estimated_landmark[0], estimated_landmark[1], 'b+')  # Plot estimated landmarks as blue pluses

    # Add legend to the plot
    true_pose_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='True Pose')
    true_landmark_handle = mlines.Line2D([], [], color='green', marker='x', linestyle='None', markersize=10, label='True Landmarks')
    estimated_pose_handle = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=10, label='Estimated Pose')
    estimated_landmark_handle = mlines.Line2D([], [], color='blue', marker='+', linestyle='None', markersize=10, label='Estimated Landmarks')
    plt.legend(handles=[true_pose_handle, true_landmark_handle, estimated_pose_handle, estimated_landmark_handle])

    plt.title("SLAM Results" + title_suffix)  # Update title with the suffix based on neural network usage

    # Calculate and print RMSE for position and orientation
    position_rmse, orientation_rmse = robot.calculate_rmse(true_poses, estimated_poses)
    print("errors", position_rmse, orientation_rmse)

    # Plot differences in position and orientation for each pose
    plt.ioff()  # Turn off interactive mode
    plt.show()
    plt.pause(2)
    plt.clf()

    # Additional plots for position and orientation differences
    pose_ids = true_poses.keys()  # Assuming both dictionaries have the same keys
    position_differences = [np.sqrt((true_poses[id][0] - estimated_poses[id][0])**2 + (true_poses[id][1] - estimated_poses[id][1])**2) for id in pose_ids]
    orientation_differences = [angle_difference(true_poses[id][2], estimated_poses[id][2]) for id in pose_ids]

    # Position difference plot
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(list(pose_ids), position_differences, label='Position Difference', marker='o')
    plt.xlabel('Pose ID')
    plt.ylabel('Position Difference')
    plt.title('Position Differences per Pose' + title_suffix)
    plt.grid(True)
    plt.legend()

    # Orientation difference plot
    plt.subplot(1, 2, 2)
    plt.plot(list(pose_ids), orientation_differences, label='Orientation Difference', marker='x')
    plt.xlabel('Pose ID')
    plt.ylabel('Orientation Difference (radians)')
    plt.title('Orientation Differences per Pose' + title_suffix)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Comparison plots for X and Y coordinates
    pose_ids = list(true_poses.keys())  # List of pose IDs for indexing
    true_x = [true_poses[id][0] for id in pose_ids]
    true_y = [true_poses[id][1] for id in pose_ids]
    estimated_x = [estimated_poses[id][0] for id in pose_ids]
    estimated_y = [estimated_poses[id][1] for id in pose_ids]

    # Plotting
    plt.figure(figsize=(14, 6))

    # Subplot for x coordinates
    plt.subplot(1, 2, 1)
    plt.plot(pose_ids, true_x, 'ko-', label='True X')
    plt.plot(pose_ids, estimated_x, 'ro--', label='Estimated X')
    for i, pid in enumerate(pose_ids):
        plt.plot([pid, pid], [true_x[i], estimated_x[i]], 'k:', linewidth=1)  # Dotted line between true and estimated

    plt.xlabel('Pose ID')
    plt.ylabel('X Coordinate')
    plt.title('Comparison of X Coordinates' + title_suffix)
    plt.legend()
    plt.grid(True)

    # Subplot for y coordinates
    plt.subplot(1, 2, 2)
    plt.plot(pose_ids, true_y, 'ko-', label='True Y')
    plt.plot(pose_ids, estimated_y, 'ro--', label='Estimated Y')
    for i, pid in enumerate(pose_ids):
        plt.plot([pid, pid], [true_y[i], estimated_y[i]], 'k:', linewidth=1)  # Dotted line between true and estimated

    plt.xlabel('Pose ID')
    plt.ylabel('Y Coordinate')
    plt.title('Comparison of Y Coordinates' + title_suffix)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

create_slam_with_real_time_plotting_corrected()