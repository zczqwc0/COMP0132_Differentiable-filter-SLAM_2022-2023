import jaxlie
import jaxfg
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from learn_process_noise import LearnedProcessNoise
from Learned_factor import simulate_ground_truth_trajectory

# Constants

# the trajectory optimization performs well when there are a sufficient number of steps (e.g., 120~140) 
# but fails to converge to a reasonable solution with fewer steps, and from the plot of the final path, 
# we can find out that when number of step has a remainder of 0 when divided by 20,, this would give the optimization 
# a strong constraint to refine the entire trajectory due to the loop closure which is set every 20 steps the robot
# will return back to origin. For trajectories with a number of steps not divisble by 20, the optimization would be 
# missing this final loop closure. As a result, any drift accumulated during the trajectory would reamin uncorrected,
# leading to a worse result, especially when the number of steps is close to the number that is divisble by 20. 

NUM_STEP = 140
STEP_SIZE = 1

def generate_square_trajectory_data(num_steps: int, step_size: float) -> Tuple[np.ndarray, np.ndarray]:
    
    relative_displacements, errors = [], []
    current_state = jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)
    
    for i in range(num_steps):
        if i % 5 == 0 and i != 0:
            T = jaxlie.SE2.from_xy_theta(0, 0.0, np.pi / 2)
        else:
            T = jaxlie.SE2.from_xy_theta(step_size, 0.0, 0.0)

        ground_truth_state, noise_vector  = simulate_ground_truth_trajectory(current_state, T)
        
        displacement = current_state.inverse().multiply(ground_truth_state).log()
        relative_displacements.append(displacement)
        
        errors.append(noise_vector)
        current_state = ground_truth_state

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

        
        factors.append(
            jaxfg.geometry.BetweenFactor.make(
                variable_T_world_a=pose_variables[i-1],
                variable_T_world_b=pose_variables[i],
                T_a_b=T_with_noise,
                noise_model=jaxfg.noises.DiagonalGaussian(predicted_noise),
            )
        )

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
    X, y = generate_square_trajectory_data(NUM_STEP, STEP_SIZE)
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
