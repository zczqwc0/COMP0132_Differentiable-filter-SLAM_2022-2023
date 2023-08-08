import jaxlie
import jaxfg
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Number of poses (steps in the simulation)
N = 20

# Create pose variables
pose_variables = [jaxfg.geometry.SE2Variable() for _ in range(N)]

# Assume the robot moves at a constant speed of 1 unit per time step
step = 1.0

# Create prior factor for the first pose
factors = [
    jaxfg.geometry.PriorFactor.make(
        variable=pose_variables[0],
        mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),  # Start at the origin
        noise_model=jaxfg.noises.DiagonalGaussian(jnp.array([0.3, 0.3, 0.1])),
    )
]

# Create between factors for each step
for i in range(1, N):
    # The robot is expected to move `speed` units forward
    T = jaxlie.SE2.from_xy_theta(step, 0.0, 0.0)

    factors.append(
        jaxfg.geometry.BetweenFactor.make(
            variable_T_world_a=pose_variables[i-1],
            variable_T_world_b=pose_variables[i],
            T_a_b=T,
            noise_model=jaxfg.noises.DiagonalGaussian(jnp.array([0.3, 0.3, 0.1])),
        )
    )

# Create factor graph
graph = jaxfg.core.StackedFactorGraph.make(factors)

# Initialize all poses to the identity
initial_assignments = jaxfg.core.VariableAssignments.make_from_defaults(pose_variables)

# Solve the graph
solution_assignments = graph.solve(initial_assignments)

# Extract the x coordinates of each pose
x_coordinates = [solution_assignments.get_value(variable).translation()[0] for variable in pose_variables]

plt.plot(x_coordinates, 'o-', markersize=5)
plt.title('Path of the robot')
plt.xlabel('Time step')
plt.ylabel('x position')
plt.grid(True)
plt.show()