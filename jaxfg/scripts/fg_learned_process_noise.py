import jaxlie
import jaxfg
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from learn_process_noise import LearnedProcessNoise

# Constants

NUM_STEP = 200
STEP_SIZE = 1


def processModel(current_state: jaxlie.SE2, u_k: jnp.array, delta_T: float) -> Tuple[jaxlie.SE2, jnp.array]:
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
    noise_std = jnp.array([0.003, 0.003, 0.001])
    # Generate noise from a normal distribution
    noise = jnp.array([np.random.normal(0, std) for std in noise_std])
    
    # Control input
    u_k = jnp.array([u_k[0], u_k[1], u_k[2]])

    # Rotation matrix M(psi)
    theta = current_state.rotation().log()[0]
    M = jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0],
                   [jnp.sin(theta), jnp.cos(theta), 0],
                   [0, 0, 1]])
    print (M)

    # State update
    delta_state = delta_T * jnp.dot(M, (u_k + noise))
    T_delta = jaxlie.SE2.exp(delta_state)

    # Predicted new state
    noisy_state = current_state.multiply(T_delta)

    noise_vector = (current_state.inverse().multiply(noisy_state)).log()
    print (noisy_state, noise_vector)
    return noisy_state, noise_vector

def generate_square_trajectory_data(delta_T: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    
    L = 4  # Desired side length of the square
    speed = L / (4 * delta_T)  # Calculated speed based on desired side length and delta_T
    
    relative_displacements, errors = [], []
    current_state = jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)
    
    for i in range(1, NUM_STEP):
        # If i is a multiple of 5, only rotate
        if i % 5 == 0:
            speed = 0.0
            angular_velocity = np.pi / 2
            u_k = jnp.array([0, 0, angular_velocity])
        else:
            # Determine the current segment of the trajectory
            segment = (i-1) // 5 % 4

            if segment == 0:  # Steps 1-4
                speed = 1.0
                angular_velocity = 0.0
                u_k = jnp.array([speed, 0, angular_velocity])
            elif segment == 1:  # Steps 6-9
                speed = 1.0
                angular_velocity = 0.0
                u_k = jnp.array([0, -speed, angular_velocity])
            elif segment == 2:  # Steps 11-14
                speed = 1.0
                angular_velocity = 0.0
                u_k = jnp.array([-speed, 0, angular_velocity])
            elif segment == 3:  # Steps 16-19
                speed = 1.0
                angular_velocity = 0.0
                u_k = jnp.array([0, speed, angular_velocity])


        print(i)
        print(u_k)
        noisy_state, noise_vector  = processModel(current_state, u_k, delta_T)
        
        displacement = current_state.inverse().multiply(noisy_state).log()
        relative_displacements.append(displacement)
        
        errors.append(noise_vector)
        current_state = noisy_state

    return (
        np.vstack(relative_displacements),
        np.vstack(errors)
    )

def initialize_pose_and_factors(learned_noise_model: LearnedProcessNoise) -> Tuple[List, List]:
    
    pose_variables = [jaxfg.geometry.SE2Variable() for _ in range(NUM_STEP)]
    factors = [
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.array([0.03, 0.03, 0.01])),
        )
    ]
    
    for i in range(1, NUM_STEP):
        if i % 5 == 0:
            T = jaxlie.SE2.from_xy_theta(0, 0.0, np.pi / 2)
        else:
            T = jaxlie.SE2.from_xy_theta(STEP_SIZE, 0.0, 0.0)

        predicted_noise = learned_noise_model.predict_noise(np.array(T.log()))
        predicted_noise = jnp.squeeze(predicted_noise) 
        T_with_noise = T.multiply(jaxlie.SE2.exp(predicted_noise))

        # Using identity covariance for the predicted process noise
        noise_model = jaxfg.noises.DiagonalGaussian(jnp.array([1,1,1]))
        
        factors.append(
            jaxfg.geometry.BetweenFactor.make(
                variable_T_world_a=pose_variables[i-1],
                variable_T_world_b=pose_variables[i],
                T_a_b=T_with_noise,
                noise_model=noise_model,
            )
        )

        # Add a loop closure factor every 20 steps
        if i % 20 == 0:
            T_closure = jaxlie.SE2.identity()
            factors.append(
                jaxfg.geometry.BetweenFactor.make(
                    variable_T_world_a=pose_variables[0],
                    variable_T_world_b=pose_variables[i],
                    T_a_b=T_closure,
                    noise_model=jaxfg.noises.DiagonalGaussian(jnp.array([0.03, 0.03, 0.01])),
                )
            )

    return pose_variables, factors

def initialize_assignments(pose_variables: List[jaxfg.geometry.SE2Variable]) -> jaxfg.core.VariableAssignments:
    
    initial_assignments_dict = {}
    for i, variable in enumerate(pose_variables):
        step = i % 20
        if step == 5:
            initial_value = jaxlie.SE2.from_xy_theta(4, 0.0, np.pi / 2)
        elif step == 10:
            initial_value = jaxlie.SE2.from_xy_theta(4, 4, np.pi)
        elif step == 15:
            initial_value = jaxlie.SE2.from_xy_theta(0, 4, 3 * np.pi / 2)
        elif step < 5:
            initial_value = jaxlie.SE2.from_xy_theta(step, 0.0, 0.0)
        elif 5 < step < 10:
            initial_value = jaxlie.SE2.from_xy_theta(4, step - 5, np.pi / 2)
        elif 10 < step < 15:
            initial_value = jaxlie.SE2.from_xy_theta(4 - (step - 10), 4, np.pi)
        elif 15 < step < 20:
            initial_value = jaxlie.SE2.from_xy_theta(0, 4 - (step - 15), 3 * np.pi / 2)
    
        initial_assignments_dict[variable] = initial_value

    return jaxfg.core.VariableAssignments.make_from_dict(initial_assignments_dict)

def print_poses(assignment: jaxfg.core.VariableAssignments, pose_variables: List[jaxfg.geometry.SE2Variable]):
    
    for i, variable in enumerate(pose_variables):
        pose = assignment.get_value(variable)
        translation = pose.translation()
        rotation = pose.rotation().as_radians()
        angle_deg = np.degrees(rotation)

        print(f"Pose {i}:")
        print(f"Translation: {translation}")
        print(f"Rotation angle in degrees: {angle_deg}\n")

def plot_final_path(solution_assignments: jaxfg.core.VariableAssignments, pose_variables: List[jaxfg.geometry.SE2Variable]):
    
    x_coordinates = [solution_assignments.get_value(variable).translation()[0] for variable in pose_variables]
    y_coordinates = [solution_assignments.get_value(variable).translation()[1] for variable in pose_variables]

    plt.plot(x_coordinates, y_coordinates, 'o-', markersize=5)
    plt.title('Final path of the robot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def main():
    
    # Initialize the LearnedProcessNoise class
    learned_noise = LearnedProcessNoise()

    # Generate your training data based on the square trajectory
    X, y = generate_square_trajectory_data()
    learned_noise.train(X, y)
    learned_noise.save('Dataset of learned process noise')

    # Initialize pose variables and factors
    pose_variables, factors = initialize_pose_and_factors(learned_noise)
    graph = jaxfg.core.StackedFactorGraph.make(factors)

    # Create an initial guess for the poses
    initial_assignments = initialize_assignments(pose_variables)

    # Print initial guess
    print("Initial assignments:")
    print_poses(initial_assignments, pose_variables)

    # Solve the factor graph
    solution_assignments = graph.solve(initial_assignments)

    # Print and visualize the results
    print("Final poses:")
    print_poses(solution_assignments, pose_variables)
    plot_final_path(solution_assignments, pose_variables)

if __name__ == "__main__":
    main()
