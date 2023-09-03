import jaxlie
import jaxfg
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd
from io import StringIO

from learn_process_noise import NoiseAutoencoder
# Constants

NUM_STEP = 1000
STEP_SIZE = 1


def processModel(input_current_state: jaxlie.SE2, u_k: jnp.array, noise: jnp.array, delta_T: float) -> Tuple[jaxlie.SE2, jaxlie.SE2, jnp.array, jnp.array]:
    """
    Simulates a single step of the robot's motion based on the process model.

    Args:
        current_state: Current state of the robot.
        u_k: Control input.
        delta_T: Length of the prediction interval.
        noise: Noise to be added to the process model.
        
    Returns:
        current_state: Current state of the robot.
        noisy_state: Predicted state of the robot after applying the process model input and noise.
        u_k: Control input.
        noise: Noise added to the process_model input as an array.
    """
    current_state = input_current_state
    theta = current_state.rotation().log()[0]

    M = jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0],
                   [jnp.sin(theta), jnp.cos(theta), 0],
                   [0, 0, 1]])

    # State update
    delta_state = delta_T * jnp.dot(M, (u_k + noise))
    T_delta = jaxlie.SE2.exp(delta_state)

    # Predicted new state with noise
    noisy_state = current_state.multiply(T_delta)
    
    return current_state, noisy_state, u_k, noise



def generate_square_trajectory_data(input_current_state: jaxlie.SE2, input_u_k: jnp.array, input_noise: jnp.array, delta_T = 1) -> None:
    
    
    data = []
    current_state = input_current_state
    for i in range(1, NUM_STEP):
        # If i is a multiple of 5, only rotate
        if i % 5 == 0:
            u_k = jnp.array([0, 0, np.pi / 2])
        else:
            # Determine the current segment of the trajectory
            segment = (i-1) // 5 % 4

            if segment == 0:  # Steps 1-4
                # u_k = jnp.array([input_u_k.at[0], 0, 0])
                u_k = jnp.array([1, 0, 0])
            elif segment == 1:  # Steps 6-9
                # u_k = jnp.array([0, -input_u_k.at[0], 0])
                u_k = jnp.array([0, -1, 0])
            elif segment == 2:  # Steps 11-14
                # u_k = jnp.array([-input_u_k.at[0], 0, 0])
                u_k = jnp.array([-1, 0, 0])
            else:  # Steps 16-19
                # u_k = jnp.array([0, input_u_k.at[0], 0])
                u_k = jnp.array([0, 1, 0])


        current_state, noisy, control, noise  = processModel(current_state, u_k, input_noise, delta_T)
        
        # Collect the state, noisy prediction, and control input together
        x,y = current_state.unit_complex_xy[..., 2:]
        theta = current_state.rotation().log()[0]
        curState_array = np.array([x, y, theta])
        x,y = noisy.unit_complex_xy[..., 2:]
        theta = noisy.rotation().log()[0]
        noisy_array = np.array([x, y, theta])
        control = np.array([control[0], control[1], control[2]])


        combined_data = np.hstack([curState_array, noisy_array, control, noise])

        # X_data.append(combined_data)
        # y_data.append(noise)
        data.append(combined_data)
        current_state = noisy


    header = 'current_state_position_x,current_state_position_y,current_state_angle,noisy_state_position_x,noisy_state_position_y,noisy_state_angle,control_input_x,control_input_y,control_input_angle,noise_x,noise_y,noise_angle'
    end = 'end'
    file_path = 'data.csv'
    with open(file_path, 'a') as file:
        np.savetxt(file, data, delimiter=',', header=header, footer=end, comments='')
    return


def initialize_pose_and_factors(learned_noise_model: NoiseAutoencoder) -> Tuple[List, List]:

    current_state = jaxlie.SE2.from_xy_theta(0, 0.0, np.pi / 2)
    input_noise = jnp.array([0.01,0.01,0.01])
    noise_model = jaxfg.noises.DiagonalGaussian(input_noise)
    delta_T  = 1

    pose_variables = [jaxfg.geometry.SE2Variable() for _ in range(NUM_STEP)]
    factors = [
        jaxfg.geometry.PriorFactor.make(
            variable=pose_variables[0],
            mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
            noise_model=noise_model,
        )
    ]
    for i in range(1, NUM_STEP):
        if i % 5 == 0:
            u_k = jnp.array([0, 0, np.pi / 2])
        else:
            # Determine the current segment of the trajectory
            segment = (i-1) // 5 % 4

            if segment == 0:  # Steps 1-4
                # u_k = jnp.array([input_u_k.at[0], 0, 0])
                u_k = jnp.array([1, 0, 0])
            elif segment == 1:  # Steps 6-9
                # u_k = jnp.array([0, -input_u_k.at[0], 0])
                u_k = jnp.array([1, 0, 0])
            elif segment == 2:  # Steps 11-14
                # u_k = jnp.array([-input_u_k.at[0], 0, 0])
                u_k = jnp.array([1, 0, 0])
            else:  # Steps 16-19
                # u_k = jnp.array([0, input_u_k.at[0], 0])
                u_k = jnp.array([1, 0, 0])

        if i % 5 == 0:
            T = jaxlie.SE2.from_xy_theta(0, 0.0, np.pi / 2)
        else:
            T = jaxlie.SE2.from_xy_theta(STEP_SIZE, 0.0, 0.0)

        current_state, noisy, control, noise  = processModel(current_state, u_k, input_noise, delta_T)
        x,y = current_state.unit_complex_xy[..., 2:]
        theta = current_state.rotation().log()[0]
        current_state_array = np.array([x, y, theta])
        x,y = noisy.unit_complex_xy[..., 2:]
        theta = noisy.rotation().log()[0]
        noisy_array = np.array([x, y, theta])

        predicted_noise = learned_noise_model.predict_noise(np.hstack([current_state_array, noisy_array, control]).reshape(-1,9))
        predicted_noise = jnp.squeeze(predicted_noise) 
        T_with_noise = T.multiply(jaxlie.SE2.exp(predicted_noise))

        # Using identity covariance for the predicted process noise

        
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
                    noise_model=noise_model,
                )
            )
        current_state = noisy

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


def read_file(file_path):
    # Lists to hold data lines and columns
    data_lines = []
    header = 'current_state_position_x,current_state_position_y,current_state_angle,noisy_state_position_x,noisy_state_position_y,noisy_state_angle,control_input_x,control_input_y,control_input_angle,noise_x,noise_y,noise_angle'
    end = 'end'
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check if line is not a header or end line
            if not line.startswith((header, end)):
                data_lines.append(line)

    # Convert list of data lines back to a string and use StringIO to make it file-like
    data_str = '\n'.join(data_lines)
    
    # Use pandas to read the CSV data
    df = pd.read_csv(StringIO(data_str), header=None)
    
    # Only take the first 9 columns
    df = df.iloc[:, :9]
    
    return df



def main():
    input_current_state_array = jnp.array([0.01, 0.002, 0])
    input_current_state = jaxlie.SE2.from_xy_theta(input_current_state_array[0], input_current_state_array[1], input_current_state_array[2])
    input_u_k = jnp.array([1, 0, 1])    
    input_noise = jnp.array([0.02, 0.01, 0.03])
    delta_T = 1
    generate_square_trajectory_data(input_current_state, input_u_k, input_noise, delta_T)
    df = read_file('data.csv')
    print(df.shape)
    # Initialize the LearnedProcessNoise class
    learned_noise = NoiseAutoencoder()
    learned_noise.train(df)
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
