import re
import numpy as np
import random

class STLParser:
    
    @staticmethod
    def remove_spaces(formula):
        return formula.replace(" ","")  # Removes all spaces in the formula
    @staticmethod
    def count_and_operators(formula):
        return formula.count('&')  # Counts each instance of '&' 
    @staticmethod
    def split_formula(formula):
        return formula.split('&')  # Splits the formula at each '&'
    @staticmethod
    def find_max_agent(formula):
        # Regex to find patterns like x1, x2, x30, etc.
        agent_pattern = r'x(\d+)'
        agents = re.findall(agent_pattern, formula)
        agent_numbers = map(int, agents)
        return max(agent_numbers)
    @staticmethod
    def extract_time_intervals(formula):
        # Pattern to find temporal operators followed by intervals in square brackets
        pattern = r'(G|F)_\[(\d+),(\d+)\]'
        matches = re.finditer(pattern, formula)

        intervals = []
        for match in matches:
            operator = match.group(1)
            t1 = int(match.group(2))
            t2 = int(match.group(3))
            intervals.append((operator, t1, t2))
        return intervals
    def time_horizon(intervals):
    # Extract numbers using a generator and find the max
        return max(max(t[1], t[2]) for t in intervals)
    @staticmethod
    def return_function(formula):
        remove_spaces = STLParser.remove_spaces(formula)
        count_and_operators = STLParser.count_and_operators(remove_spaces)
        split_formula = STLParser.split_formula(remove_spaces)
        find_max_agent = STLParser.find_max_agent(remove_spaces)
        time_intervals = STLParser.extract_time_intervals(remove_spaces)
        time_horizon = STLParser.time_horizon(time_intervals)
        return remove_spaces, count_and_operators, split_formula, find_max_agent, time_intervals, time_horizon
        
class UserInput:
    
    def get_space_dimensions():
        # Ask for the dimensions of the space (e.g., 2D or 3D and their respective sizes)
        dimensions = int(input("Enter the number of dimensions for the space (e.g., 2 for 2D, 3 for 3D): "))
        return dimensions
    
    def get_initial_conditions(number_of_robots, dimensions):
        # Initialize a dictionary to store the initial conditions of each robot
        initial_conditions = {}
        for i in range(1, number_of_robots + 1):
            # For each robot, ask for their initial positions in all dimensions
            while True:
                input_str = input(f"Enter the initial conditions for robot {i} as a list (e.g., [1, 2] for 2D): ")
                try:
                    # Parse the input string to a list of floats
                    conditions = eval(input_str)
                    if isinstance(conditions, list) and len(conditions) == dimensions:
                        conditions = [float(x) for x in conditions]  # Convert each element to float
                        initial_conditions[f'Robot {i}'] = conditions
                        break
                    else:
                        print(f"Error: Please enter a list of {dimensions} numbers.")
                except (SyntaxError, TypeError, ValueError):
                    print("Invalid input. Please enter a proper list format, e.g., [1, 2].")
        return np.array(list(initial_conditions.values())).flatten()


def create_gradient_functions(num_agents):
    for i in range(1, num_agents + 1):
        # Create a function definition as a string
        func_definition = f"""
def grad{i}(params):
    print('Calculating gradient for agent {i} with parameters:', params)
    # Implement the gradient calculation logic here
    return params  # Placeholder return
"""
        # Execute the function definition
        exec(func_definition, globals())
        

# Create the parse tree
formula = 'G_[1,2](||x1-x2||<=2) & F_[5,6](x2**2-x3**3||<=1) & G_[3,4](||x1-x3||<=3)'
remove_spaces, count_and_operators, split_formula, number_of_robots, extract_time_intervals, time_horizon = STLParser.return_function(formula)
dimensions = UserInput.get_space_dimensions()
initial_conditions = UserInput.get_initial_conditions(number_of_robots, dimensions)

t_0 = 0
t_end = time_horizon+0.01

goal_condition = []
initial_condition = np.insert(initial_conditions,0, t_0)
goal_condition =    np.insert(initial_conditions,0, t_end)

N = np.vstack((initial_condition, goal_condition))

print(N)
exit()
# Example usage: creating 10 gradient functions
create_gradient_functions(10)

# Test one of the created functions
grad7([1, 2, 3])


state_dimensions = 3

