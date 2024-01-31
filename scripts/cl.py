import gtsam
import numpy as np
import tensorflow as tf
import gtsam
from sklearn.preprocessing import StandardScaler
import numpy as np


class SimulatedEnvironment:
    def __init__(self, num_landmarks=10, area_size=(20, 20)):
        self.landmarks = self.generate_landmarks(num_landmarks, area_size)

    def generate_landmarks(self, num_landmarks, area_size):
        # Generate landmarks within the specified area
        landmarks = {}
        for i in range(1, num_landmarks + 1):
            x = np.random.uniform(-area_size[0]/2, area_size[0]/2)
            y = np.random.uniform(-area_size[1]/2, area_size[1]/2)
            landmarks[i] = np.array([x, y])
        return landmarks

class SimulatedRobot:
    def __init__(self, environment):
        self.environment = environment
        self.pose = gtsam.Pose2(0, 0, 0)  # Initial pose
        self.speed = 0.5  # Constant speed
        self.angular_velocity = 0.1  # Constant angular velocity
        self.delta_t = 0.1 
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


    def dynamic_model(self,x_k1,x_k2, psi_k, u_k): 
        
        Q_std_dev = np.diag([0.1, 0.1, 0.01]) 
 
        v_k = np.random.multivariate_normal([0, 0, 0], Q_std_dev)
 
        
        M_psi_k = self.rotation_matrix(psi_k)
        a = M_psi_k @ (u_k.T + v_k.T)
        c = self.delta_t * a
        d = [x_k1,x_k2,psi_k]+ c
        d[2] = self.normalize_theta(d[2])
        print("pose moving")
        print(d)
        self.pose = self.pose.compose(gtsam.Pose2(d[0], d[1], d[2]))

    def move(self, dt=1.0):
        # Update pose based on differential drive kinematics
        theta = self.pose.theta()
        dx = self.speed * np.cos(theta) * dt
        dy = self.speed * np.sin(theta) * dt
        dtheta = self.angular_velocity * dt
        self.pose = self.pose.compose(gtsam.Pose2(dx, dy, dtheta))

    def sense_landmarks(self):
        # Simulate sensing of landmarks with noise
        visible_landmarks = {}
        i = 0
        for id, position in self.environment.landmarks.items():
            if np.linalg.norm(self.pose.translation() - position) < 5:  # 10 units range
                # Calculate true bearing and range
                i = i+1
                true_bearing = np.arctan2(position[1] - self.pose.y(), position[0] - self.pose.x()) - self.pose.theta()
                true_range = np.linalg.norm(position - self.pose.translation())

                # Add noise to bearing and range
                noise_bearing = np.random.normal(0, 0.05)  # Adjust the standard deviation as needed
                noise_range = np.random.normal(0, 0.1)    # Adjust the standard deviation as needed

                bearing = true_bearing + noise_bearing
                range = true_range + noise_range

                visible_landmarks[id] = (bearing, range)
        print("landmarks")
        print(i)
        return visible_landmarks

    def sense_odometry(self):
        # Simulate odometry data with noise
        noise_x, noise_y, noise_theta = np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(0, 0.05)
        noisy_pose = gtsam.Pose2(self.pose.x() + noise_x, self.pose.y() + noise_y, self.pose.theta() + noise_theta)
        return noisy_pose

class SLAMNNIntegrator:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict_motion_and_noise(self, speed, angular_velocity, bearing, distance, additional_feature):
        """
        Predicts the motion and noise covariance based on robot control inputs and a single observation.
        speed: Robot's linear speed.
        angular_velocity: Robot's angular velocity.
        bearing: Bearing to a landmark.
        distance: Distance to a landmark.
        additional_feature: An additional feature relevant to the prediction.
        """
        control_inputs = np.array([speed, angular_velocity])
        observation = np.array([bearing, distance, additional_feature])
        combined_inputs = np.concatenate([control_inputs, observation])
        combined_inputs = combined_inputs.reshape(1, -1)  # Reshape for the neural network

        predictions = self.model.predict(combined_inputs)[0]
        motion = predictions[:3]
        noise_cov = predictions[3:]
        return motion, noise_cov




    def update_slam(self, graph, initial_estimate, control_inputs, observations, current_key, next_key):
        """
        Updates the SLAM graph and initial estimates using the NN predictions.
        graph: The current factor graph.
        initial_estimate: The current set of initial estimates.
        control_inputs: Current control inputs.
        observations: Current sensor observations.
        current_key, next_key: Keys for the current and next poses in the graph.
        """
        # Predict motion and noise
        motion, noise_cov = self.predict_motion_and_noise(control_inputs, observations)

        # Create a noise model for the motion factor
        noise_model = gtsam.noiseModel.Diagonal.Variances(noise_cov)

        # Create the motion factor (BetweenFactor)
        pose_current = initial_estimate.atPose2(current_key)
        pose_next_predicted = pose_current.compose(gtsam.Pose2(motion[0], motion[1], motion[2]))
        motion_factor = gtsam.BetweenFactorPose2(current_key, next_key, pose_next_predicted, noise_model)

        # Update the graph and initial estimates
        graph.push_back(motion_factor)
        initial_estimate.insert(next_key, pose_next_predicted)
