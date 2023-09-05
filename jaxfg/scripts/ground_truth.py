import jaxlie
import jaxfg
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

class GroundTruthTrajectory:
    def __init__(self, num_step=1000, step_size=1):
        self.num_step = num_step
        self.step_size = step_size
        self.pose_variables = []
        self.factors = []
        self.graph = None
        self.initial_assignments_dict = {}
        self.initial_assignments = None
        self.solution_assignments = None

    def create_trajectory(self):
        # Create pose variables
        self.pose_variables = [jaxfg.geometry.SE2Variable() for _ in range(self.num_step)]

        # Create prior factor for the first pose
        self.factors = [
            jaxfg.geometry.PriorFactor.make(
                variable=self.pose_variables[0],
                mu=jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0),
                noise_model=jaxfg.noises.DiagonalGaussian(jnp.array([0.01, 0.01, 0.01])),
            )
        ]

        for i in range(1, self.num_step):
            if i % 5 == 0:
                # Turn 90 degrees counterclockwise
                T = jaxlie.SE2.from_xy_theta(0, 0.0, np.pi / 2)
            else:
                # Drive step_size units to the right
                T = jaxlie.SE2.from_xy_theta(self.step_size, 0.0, 0.0)

            # Add noise to the motion
            noise = jnp.array([np.random.normal(0, 0.01), np.random.normal(0, 0.01), np.random.normal(0, 0.01)])
            T_noise = jaxlie.SE2.from_xy_theta(*T.translation() + noise[:2], T.rotation().as_radians() + noise[2])


            self.factors.append(
                jaxfg.geometry.BetweenFactor.make(
                    variable_T_world_a=self.pose_variables[i-1],
                    variable_T_world_b=self.pose_variables[i],
                    T_a_b= T_noise,
                    noise_model=jaxfg.noises.DiagonalGaussian(jnp.array([0.01, 0.01, 0.01])),
                )
            )

            if i % 20 == 0 and i != 0:
                # Add a loop closure constraint
                T_closure = jaxlie.SE2.identity() # No transformation
                self.factors.append(
                    jaxfg.geometry.BetweenFactor.make(
                        variable_T_world_a = self.pose_variables[0],  # Always close the loop to the first pose
                        variable_T_world_b = self.pose_variables[i],
                        T_a_b = T_closure,
                        noise_model = jaxfg.noises.DiagonalGaussian(jnp.array([0.01, 0.01, 0.01])),
                    )
                )

    def create_factor_graph(self):
        self.graph = jaxfg.core.StackedFactorGraph.make(self.factors)

    def create_initial_assignments(self):

        # Initialize all pose variables
        for i, variable in enumerate(self.pose_variables):
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
            self.initial_assignments_dict[variable] = initial_value

            # Create the VariableAssignments object
            self.initial_assignments = jaxfg.core.VariableAssignments.make_from_dict(self.initial_assignments_dict)

    def solve(self):
        # Solve the factor graph
        self.solution_assignments = self.graph.solve(self.initial_assignments)

    def plot(self):
        # Plot the 2D planer drive bot simulation

        # Extract the x and y coordinates of each pose
        x_coordinates = [self.solution_assignments.get_value(variable).translation()[0] for variable in self.pose_variables]
        y_coordinates = [self.solution_assignments.get_value(variable).translation()[1] for variable in self.pose_variables]

        print("ground_truth_trajectory")
        plt.plot(x_coordinates, y_coordinates, 'o-', markersize=5)
        plt.title('Ground Truth Trajectory')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()
        
    
    def save_to_csv(self, filename="ground_truth_trajectory.csv"):
        # Save the ground truth trajectory to a CSV file
        directory = "./scripts/Data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Prepare data for CSV
        csv_data = []
        for i, variable in enumerate(self.pose_variables):
            pose = self.solution_assignments.get_value(variable)
            translation = pose.translation()
            rotation = pose.rotation().as_radians()
            angle_deg = np.degrees(rotation)
            csv_data.append([i, translation[0], translation[1], angle_deg])
        # Write to CSV
        csv_file_path = os.path.join(directory, filename)
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Pose Index", "X", "Y", "Rotation (Degrees)"])  # Writing the headers
            writer.writerows(csv_data)

    def run(self):
        self.create_trajectory()
        self.create_factor_graph()
        self.create_initial_assignments()
        self.solve()
        self.plot()
        self.save_to_csv()

