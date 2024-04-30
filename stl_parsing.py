import re
import numpy as np
import random
import numpy.linalg as la
from sympy import symbols, diff, sympify


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
    
    @staticmethod
    def time_horizon(intervals):
    # Extract numbers using a generator and find the max

        return max(max(t[1], t[2]) for t in intervals)
    
    @staticmethod
    def extract_predicates(formula):
        # This pattern matches expressions inside parentheses
        pattern = r'\((.*?)\)'  # Non-greedy match for any characters inside parentheses
        predicates = re.findall(pattern, formula)
        predicate_standard = []
        for predicate in predicates:
            # Check if the predicate contains <= or >=
            if '<=' in predicate:
                parts = predicate.split('<=')
                new_predicate = f"{parts[0]}-({parts[1]})"
            elif '>=' in predicate:
                parts = predicate.split('>=')
                new_predicate = f"{parts[1]}-({parts[0]})"
            if '-(-' in new_predicate:
                new_predicate = new_predicate.replace('-(-', '+(')
            predicate_standard.append(new_predicate)

        return predicate_standard
    
    @staticmethod
    def tau_nums(formula, predicates):
        print(predicates)
        tau_nums = 1+ len(predicates) + formula.count('G') + formula.count('F')
        print(tau_nums)
        return tau_nums

    @staticmethod
    def return_function(formula):
        formula = STLParser.remove_spaces(formula)
        count_and_operators = STLParser.count_and_operators(formula)
        split_formula = STLParser.split_formula(formula)
        find_max_agent = STLParser.find_max_agent(formula)
        time_intervals = STLParser.extract_time_intervals(formula)
        time_horizon = STLParser.time_horizon(time_intervals)
        predicates = STLParser.extract_predicates(formula)
        tau_nums = STLParser.tau_nums(formula, predicates)
        return formula, count_and_operators, split_formula, find_max_agent, time_intervals, time_horizon, predicates, tau_nums
        
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

class MAPS2:
    
    def __init__(self, N, sv, tstar, av, robust):
        self.N = N
        self.sv = sv
        self.tstar = tstar
        self.av = av
        self.robust = robust

    def create_gradient_functions(num_agents):
        for i in range(1, num_agents + 1):
            # Create a function definition as a string
            func_definition = f"""
def grad{i}(predicate, params, num_agents, dimensions):
    print('Calculating gradient for agent {i}')
    symbol_names = ' '.join('x'+str(j) for j in range(1,num_agents+1))
    symbols_tuple = symbols(symbol_names, real=True)
    print(symbols_tuple[1])
    expr = sympify(predicate)
    print(expr)
    diff_expr = diff(expr, "x{i}").subs({{'x'+str(j): params[j-1] for j in range(1,num_agents+1)}})
    print(diff_expr)
    print('Derivative with respect to x{i}:', diff_expr)
    return diff_expr
"""
            # Execute the function definition
            exec(func_definition, globals())

    def TrajectoryShaping(self):
        for j in range(10000):
        #while self.sv(0)<1:
            # random.seed(j) # Generates same random number for all agents
            midTime = 100*random.random() # Uses the random number generator to generate 'time'
            idx = self.N[:,0].searchsorted(midTime) # Searchsorted has complexity O(N)
            N_mid = self.Interpolate(midTime, idx) # Run linear interpolation here to give initial conditions for EXTRA
            self.ActiveVariable(midTime)
            N_gradient = np.hstack((midTime, self.GradientDescent(N_mid[1:5], midTime)))
            self.N = np.concatenate((self.N[:idx,], [N_gradient], self.N[idx:, ]))
            self.av = np.zeros(8)
        
    def GradientDescent(self, x0, midTime):
        t = 1
        alpha = 0.3 # alpha \in (0,0.5)
        beta = 0.5 # beta \in (0,1)
        k = 0
        while True:
            grad = np.array([self.gradf1(x0, midTime), self.gradf2(x0, midTime), self.gradf3(x0, midTime), self.gradf4(x0, midTime)])
            deltax = - grad 
            while self.f(x0 + t * deltax, midTime) > self.f(x0, midTime) - alpha * t * la.norm(grad)**2:
                t = beta * t 
            x0 = x0 + t * deltax 
            k += 1
            if  la.norm(grad) <= 0.001 or k >= 100:
                break
        return x0
    
    def ActiveVariable(self, midTime):
        self.av[0] = 1
        t11, t12 =  self.ValidityDomain_branch1()
        t21, t22 =  self.ValidityDomain_branch2()
        if t12>=60:
            self.sv[0] = 1
        if t22>=60:
            self.sv[1] = 1
        if midTime <= t12 and midTime >= t11:
            self.av[1] = 1
        if midTime <= t22 and midTime >= t21:
            self.av[2] = 1
        if self.av[1] > 0.5 and self.av[2] > 0.5:
            self.av[1] = np.random.choice([0,1])
            if self.av[1] > 0.5:
                self.av[2] = 0
            else:
                self.av[2] = 1
        if midTime <= 30 and midTime >= 10:
            self.av[3] = 1
        if midTime <= 60 and midTime >= 20:
            self.av[4] = 1
        if midTime <= 100 and midTime >= 80:
            self.av[5] = 1
        if midTime <= 100 and midTime >= 80:
            self.av[6] = 1
        if midTime <= 100 and midTime >= 80:
            self.av[7] = 1



MAPS2.create_gradient_functions(4)
grad4('x1**2+x2**2+x3**2+x4**2', [1,2,3,4, 5, 6, 7, 8], 4, 2)

# Create the parse tree
formula = 'G_[1,2](||x1-x2||<=2) & F_[5,6](x2**2-x3**3<=1) & G_[3,4](-||x1-x3||>=-3) & F_[7,8](x1-x2>=1)'
formula, count_and_operators, split_formula, number_of_robots, extract_time_intervals, time_horizon, predicates, tau_nums = STLParser.return_function(formula)
dimensions = UserInput.get_space_dimensions()
initial_conditions = UserInput.get_initial_conditions(number_of_robots, dimensions)

t_0 = 0
t_end = time_horizon+0.01

goal_condition = []
initial_condition = np.insert(initial_conditions,0, t_0)
goal_condition =    np.insert(initial_conditions,0, t_end)

N = np.vstack((initial_condition, goal_condition))

tau = -1*np.ones(tau_nums)
print(tau)
print(N)

# Example usage: creating 10 gradient functions
#create_gradient_functions(10)

# Test one of the created functions
#grad7([1, 2, 3])


#state_dimensions = 3

