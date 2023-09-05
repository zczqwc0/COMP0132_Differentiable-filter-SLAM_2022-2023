import jaxlie
import jaxfg
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# Number of poses (steps in the simulation)
num_step = 1000
step_size = 1

# Create pose variables
pose_variables = [jaxfg.geometry.SE2Variable() for _ in range(num_step)]

# Create prior factor for the first pose
factors = [
    jaxfg.geometry.PriorFactor.make(
        variable=pose_variables[0],
        mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),  # Start at the origin
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.array([0.01, 0.01, 0.01])),
    )
]

# Create between factors for each step
for i in range(1, num_step):
    if i % 5 == 0:
        # Turn 90 degrees counterclockwise
        T = jaxlie.SE2.from_xy_theta(0, 0.0, np.pi / 2)
    else:
        # Drive step_size units to the right
        T = jaxlie.SE2.from_xy_theta(step_size, 0.0, 0.0)

    # Add noise to the motion
    noise = jnp.array([np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)])
    T_noise = jaxlie.SE2.from_xy_theta(*T.translation() + noise[:2], T.rotation().as_radians() + noise[2])


    factors.append(
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[i-1],
            variable_T_world_b=pose_variables[i],
            T_a_b= T_noise,
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.array([0.01, 0.01, 0.01])),
        )
    )

    if i % 20 == 0 and i != 0:
        # Add a loop closure constraint
        T_closure = jaxlie.SE2.identity() # No transformation
        factors.append(
            jaxfg.geometry.BetweenFactor.make(
                variable_T_world_a = pose_variables[0],  # Always close the loop to the first pose
                variable_T_world_b = pose_variables[i],
                T_a_b = T_closure,
                noise_model = jaxfg.noises.DiagonalGaussian(jnp.array([0.01, 0.01, 0.01])),
            )
        )


# Create factor graph
graph = jaxfg.core.StackedFactorGraph.make(factors)


initial_assignments_dict = {}

# Initialize all pose variables
for i, variable in enumerate(pose_variables):
    # Determine the initial position based on the current step
    step = i % 20
    if step == 5:  # First corner of the square
        initial_value = jaxlie.SE2.from_xy_theta(4, 0.0, np.pi / 2)
    elif step == 10:  # Second corner of the square
        initial_value = jaxlie.SE2.from_xy_theta(4, 4, np.pi)
    elif step == 15:  # Third corner of the square
        initial_value = jaxlie.SE2.from_xy_theta(0, 4, 3 * np.pi / 2)
    elif step < 5:  # First side of the square
        initial_value = jaxlie.SE2.from_xy_theta(step, 0.0, 0.0)
    elif 5 < step < 10:  # Second side of the square
        initial_value = jaxlie.SE2.from_xy_theta(4, step - 5, np.pi / 2)
    elif 10 < step < 15:  # Third side of the square
        initial_value = jaxlie.SE2.from_xy_theta(4 - (step - 10), 4, np.pi)
    elif 15 < step < 20:  # Fourth side of the square
        initial_value = jaxlie.SE2.from_xy_theta(0, 4 - (step - 15), 3 * np.pi / 2)
    
    # Add the initial assignment to the dictionary
    initial_assignments_dict[variable] = initial_value

# Create the VariableAssignments object
initial_assignments = jaxfg.core.VariableAssignments.make_from_dict(initial_assignments_dict)

# Initialize all poses to the identity
# initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(pose_variables)

print("Initial assignments:")
# Print out the initial poses
for i, variable in enumerate(pose_variables):
    pose = initial_assignments.get_value(variable)

    # Access the pose's translation
    translation = pose.translation()

    # Convert the pose's rotation to an angle
    rotation = pose.rotation().as_radians()
    angle_deg = np.degrees(rotation)

    print(f"Pose {i}:")
    print(f"Translation: {translation}")
    print(f"Rotation angle in degrees: {angle_deg}\n")


# Solve the graph
solution_assignments = graph.solve(initial_assignments)

# Print out the final poses
for i, variable in enumerate(pose_variables):
    pose = solution_assignments.get_value(variable)
    
    # Convert the pose's rotation to an angle
    translation = pose.translation()
    rotation = pose.rotation().as_radians()
    angle_deg = np.degrees(rotation)

    print(f"Pose {i}")
    print(f"Translation: {translation}")
    print(f"Rotation angle in degrees: {angle_deg}\n")


# Plot the 2D planer drive bot simulation
# Extract the x and y coordinates of each pose
x_coordinates = [solution_assignments.get_value(variable).translation()[0] for variable in pose_variables]
y_coordinates = [solution_assignments.get_value(variable).translation()[1] for variable in pose_variables]


plt.plot(x_coordinates, y_coordinates, 'o-', markersize=5)
plt.title('Final path of the robot')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()




