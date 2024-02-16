import gtsam
import numpy as np
import tensorflow as tf
import gtsam
import numpy as np


class SimulatedEnvironment:
    def __init__(self, num_landmarks = 500, area_size=(200, 200)):
        self.landmarks = self.generate_landmarks(num_landmarks, area_size)

    def generate_landmarks(self, num_landmarks, area_size):
        # Generate landmarks within the specified area
        landmarks = {}
        filename = "landmarks.txt"
        with open(filename, 'r') as file:
            for i, line in enumerate(file):
                x, y = map(float, line.strip().split(','))
                landmarks[i+1000] = np.array([x, y])
        return landmarks

class SimulatedRobot:
    def __init__(self, environment):
        self.last_x = 0
        self.last_y = 0
        self.last_theta = 0
        self.environment = environment
        self.pose = gtsam.Pose2(0, 0, 0)  # Initial pose
        self.speed = 0.5  # Constant speed
        self.angular_velocity = 0.1  # Constant angular velocity
        self.delta_t = 1
        self.R_std_dev = np.diag([0.2, 0.05]) 
    
    def rotation_matrix(self, psi):
        return np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

    def normalize_theta(self, theta):
        if theta < -np.pi:
            theta += 2 * np.pi
        elif theta > np.pi:
            theta -= 2 * np.pi
        return theta

    def dynamic_model(self, x_k1, x_k2, psi_k, u_k):
        max_delta_x = 5  # max change in x
        max_delta_y = 5  # max change in y
        max_delta_theta = np.radians(30)
        Q_std_dev = np.diag([0.1, 0.1, 0.01])
        v_k = np.random.multivariate_normal([0, 0, 0], Q_std_dev)

        M_psi_k = self.rotation_matrix(psi_k)
        a = M_psi_k @ (u_k.T + v_k.T)
        delta = self.delta_t * a  # Calculate the change

        # Apply limits to delta
        delta[0] = np.clip(delta[0], -max_delta_x, max_delta_x)
        delta[1] = np.clip(delta[1], -max_delta_y, max_delta_y)
        delta[2] = np.clip(delta[2], -max_delta_theta, max_delta_theta)

        # Calculate new pose components
        new_x = x_k1 + delta[0]
        new_y = x_k2 + delta[1]
        new_psi = psi_k + delta[2]
        new_psi = self.normalize_theta(new_psi)  # Ensure psi remains within [-pi, pi]

        delta_x = new_x - self.last_x
        delta_y = new_y - self.last_y
        delta_theta = new_psi - self.last_theta
        delta_theta = self.normalize_theta(delta_theta)  # Ensure delta_theta is normalized

        # Apply movement limits
        delta_x = np.clip(delta_x, -max_delta_x, max_delta_x)
        delta_y = np.clip(delta_y, -max_delta_y, max_delta_y)
        delta_theta = np.clip(delta_theta, -max_delta_theta, max_delta_theta)

        # Calculate corrected new position based on limited deltas
        corrected_new_x = self.last_x + delta_x
        corrected_new_y = self.last_y + delta_y
        corrected_new_psi = self.normalize_theta(self.last_theta + delta_theta)

        # Update the robot's pose to the corrected new position
        self.pose = gtsam.Pose2(corrected_new_x, corrected_new_y, corrected_new_psi)

        # Update last position and orientation
        self.last_x = corrected_new_x
        self.last_y = corrected_new_y
        self.last_theta = corrected_new_psi

        return gtsam.Pose2(delta_x,delta_y,delta_theta)

    def sense_odometry(self):
        # Simulate odometry data with noise
        noise_x, noise_y, noise_theta = np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(0, 0.05)
        noisy_pose = gtsam.Pose2(self.pose.x() + noise_x , self.pose.y() + noise_y  , self.pose.theta() + noise_theta )
        return np.array([noisy_pose.x() , noisy_pose.y(), noisy_pose.theta()])

    def observation_model(self, x_i, y_i, x_k, y_k, phi_k):
        # Simulate landmark observation with noise
        w_k = np.random.multivariate_normal([0, 0], self.R_std_dev)
        # Calculate range and bearing
        r_k_i = np.sqrt((x_i - x_k)**2 + (y_i - y_k)**2)  + w_k[0]
        beta_k_i = np.arctan2((y_i - y_k), (x_i - x_k)) - phi_k + w_k[1]  
        return np.array([r_k_i, beta_k_i])
    
    def estimate_landmark_position(self, pose, bearing, landmark_range):
        # Implement landmark position estimation based on the pose, bearing, and landmark range
        x = pose.x() + landmark_range * np.cos(pose.theta() + bearing)
        y = pose.y() + landmark_range * np.sin(pose.theta() + bearing)
        return gtsam.Point2(x, y)
    
    def pose_error(self,true_pose, estimated_pose):
        """
        Calculate squared position and orientation errors between two poses.
        """
        
        position_error_sq = (true_pose[0] - estimated_pose[0])**2 + (true_pose[1]- estimated_pose[1])**2
        orientation_error = np.abs(true_pose[2] - estimated_pose[2])
        orientation_error = np.arctan2(np.sin(orientation_error), np.cos(orientation_error))  # Normalize to [-pi, pi]
        orientation_error_sq = orientation_error**2
        return position_error_sq, orientation_error_sq

    def calculate_rmse(self, true_poses, estimated_poses):
        """
        Calculate the RMSE for position and orientation across all poses.
        """
        position_error_sums = 0
        orientation_error_sums = 0
        n = len(true_poses)

        for id, true_pose in true_poses.items():
            if id in estimated_poses:
                estimated_pose = estimated_poses[id]
                pos_err_sq, ori_err_sq = self.pose_error(true_pose, estimated_pose)
                position_error_sums += pos_err_sq
                orientation_error_sums += ori_err_sq
            else:
                print(f"Missing estimated pose for ID: {id}")
                n -= 1  # Adjust count if any poses are missing

        position_rmse = np.sqrt(position_error_sums / n)
        orientation_rmse = np.sqrt(orientation_error_sums / n)
        return position_rmse, orientation_rmse

class SLAMNNIntegrator:
    def __init__(self, model_path):
      
        self.model = tf.keras.models.load_model(model_path)
      
    def predict_motion_and_noise(self, u_k, x, y, theta):
        """
        Predicts the motion and noise covariance based on robot control inputs and a single observation.
        u_k: Control inputs array containing [velocity_x, velocity_y, angular_velocity].
        x, y, theta: Current robot pose.
        """
        # Assuming the inputs are already scaled/normalized if necessary
        combined_inputs = np.concatenate((u_k, np.array([x, y, theta])))
        combined_inputs = combined_inputs.reshape(1, -1)  # Reshape for the neural network

        predictions = self.model.predict(combined_inputs)[0]
        motion = gtsam.Pose2(predictions[:3] ) # Assuming the first three predictions are motion (x, y, theta)
        land_b_d =  predictions[3:5] 
        noise_cov = predictions[5:]  # Assuming the rest are noise covariance
        return motion, land_b_d, noise_cov
 
 
 