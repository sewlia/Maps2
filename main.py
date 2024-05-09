import re
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy.linalg as la
from sympy import symbols, diff, sympify, Max


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
    # Find the maximum time horizon from the intervals
        return max(max(t[1], t[2]) for t in intervals)
    
    @staticmethod
    def extract_predicates(formula):
        # Pattern to find predicates inside parentheses
        pattern = r'\((.*?)\)'  
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
        # Number of satisfaction variables
        tau_nums = 1+ len(predicates) + formula.count('G') + formula.count('F')
        return tau_nums
    
    def format_predicates(predicates):
        #Bringing the predicates into optimisation function form
        formatted_predicates = [f'Max(0,{pred})**2' for pred in predicates]
        lambdas = [f'lambda{i+1}' for i in range(len(predicates))]
        lambdas_predicates = [f'{lambdas[i]}({formatted_predicates[i]})' for i in range(len(formatted_predicates))]
        optimisation_function = '+'.join(lambdas_predicates)

        return optimisation_function


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
        optimisation_function = STLParser.format_predicates(predicates)
        return formula, count_and_operators, split_formula, find_max_agent, time_intervals, time_horizon, predicates, tau_nums, optimisation_function
        
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
    
    def __init__(self, N, optimisation_function, tau, Tstar, tstar, number_of_robots, extract_time_intervals, predicates, time_horizon):
        self.N = N
        self.optimisation_function = optimisation_function
        self.tau = tau
        self.Tstar = Tstar
        self.tstar = tstar
        self.number_of_robots = number_of_robots
        self.extract_time_intervals = extract_time_intervals
        self.predicates = predicates
        self.time_horizon = time_horizon

    def Interpolate(self, midTime, idx):
        N_mid = np.hstack((midTime, np.zeros(self.number_of_robots)))
        for i in range(1, number_of_robots+1):
            N_mid[i] = (self.N[idx,i] - self.N[idx-1,i]) * (midTime - self.N[idx-1,0]) / (self.N[idx,0]-self.N[idx-1,0]) + self.N[idx-1,i]
        return N_mid 
    
    def validity_domain(self):
        vd = np.zeros(shape=(len(self.extract_time_intervals), 2))
        new_tau = tau[-4:]
        for i, interval in enumerate(self.extract_time_intervals):
            if interval[0] == 'G' and new_tau[i] == -1:
                vd[i] = [interval[1], interval[2]]
            elif interval[0] == 'F' and new_tau[i] == -1:
                vd[i] = [interval[1], interval[2]]
            elif interval[0] == 'G' and new_tau[i] == 1:
                vd[i] = [0, 0]
            elif interval[0] == 'F' and new_tau[i] == 1:
                vd[i] = [0, 0]
        return vd

    def activation_variable(self, midTime, agent):
        lambda1 = np.zeros(len(extract_time_intervals)+1)
        lambda2 = np.zeros(len(extract_time_intervals)+1)
        agent_state = f'x{agent}'
        # Decides which predicates are active at sampled time midTime
        for i, interval in enumerate(extract_time_intervals):
            if interval[1] <= midTime <= interval[2]:
                lambda1[i] = 1
        for i, predicate in enumerate(predicates):
            if agent_state in predicate:
                lambda2[i] = 1
        lambda_ = np.logical_and(lambda1, lambda2)
        lambda_ = lambda_.astype(int)    
        return lambda_

    def activation_substitution(self, activation_lambdas):
        active_optimisation_function = optimisation_function
        for i, value in enumerate(activation_lambdas, 1):  # start counting from 1 for lambda1, lambda2, etc.
            active_optimisation_function = active_optimisation_function.replace(f'lambda{i}', str(value)+'*')
        return active_optimisation_function

    def grad(self, x0, midTime, agent):
        activation_lambdas = self.activation_variable(midTime, agent)
        active_optimisation_function = self.activation_substitution(activation_lambdas)
        symbol_names = ' '.join('x'+str(j) for j in range(1,number_of_robots+1))
        symbols_tuple = symbols(symbol_names, real=True)
        expr = sympify(active_optimisation_function)
        diff_expr = diff(expr, "x"+str(agent)).subs({'x'+str(j): x0[j-1] for j in range(1,number_of_robots+1)})
        return diff_expr
    
    def TrajectoryShaping(self):
        for j in range(10000):
        #while self.sv(0)<1:
            # random.seed(j) # Generates same random number for all agents
            midTime = time_horizon*random.random() # Uses the random number generator to generate 'time'
            idx = self.N[:,0].searchsorted(midTime) # Searchsorted has complexity O(N)
            N_mid = self.Interpolate(midTime, idx) # Run linear interpolation here to give initial conditions for EXTRA
            N_gradient = np.hstack((midTime, self.GradientDescent(N_mid[1:], midTime)))
            self.N = np.concatenate((self.N[:idx,], [N_gradient], self.N[idx:, ]))
            print(j)
        self.plotGraph()

    def GradientDescent(self, x0, midTime):
        t = 0.5
        alpha = 0.3 # alpha \in (0,0.5)
        beta = 0.5 # beta \in (0,1)
        k = 0
        while True:
            grad = np.array([self.grad(x0, midTime, j) for j in range(1,number_of_robots+1)])
            deltax = - grad 
            x0 = x0 + t * deltax 
            k += 1
            grad = np.array([float(value) for value in grad])
            #print('grad', grad)
            if  la.norm(grad) <= 0.001 or k >= 100:
                break
        return x0
    
    def plotGraph(self):
        # Plots the graph

        start_marker = "xr"
        end_marker = "bo"
        line_styles = ['b-', 'g-', 'm-', 'k-', 'c-', 'y-', 'r-', 'purple-']  
        
        num_agents = self.N.shape[1] - 1  
        for i in range(1, num_agents + 1):
            # Plot start and end points
            plt.plot(self.N[0, 0], self.N[0, i], start_marker)
            plt.plot(self.N[-1, 0], self.N[-1, i], end_marker)
            
            plt.plot(self.N[:, 0], self.N[:, i], line_styles[i % len(line_styles)], label=f'Agent {i}')

        # Set plot limits, labels, and legend
        plt.axis([0, self.N[-1,0], 0, 10])
        plt.xlabel('time(s)')
        plt.ylabel('x')
        plt.legend()
        plt.grid(True)
        plt.show()


# Create the parse tree
formula = 'G_[1,2](x1<=2) & F_[5,6](x2>=8) & G_[3,4](x3>=3) & F_[7,8](x4<=8) & G_[2,3](x4>=8) & F_[8,9](x5<=1)'
formula, count_and_operators, split_formula, number_of_robots, extract_time_intervals, time_horizon, predicates, tau_nums, optimisation_function = STLParser.return_function(formula)
G_nums = formula.count('G')
F_nums = formula.count('F')
#dimensions = UserInput.get_space_dimensions()
#initial_conditions = UserInput.get_initial_conditions(number_of_robots, dimensions)
initial_conditions = [3, 5, 2, 9, 8]
#print(formula)
#print(count_and_operators)
#print(split_formula)
#print(number_of_robots)
#print(extract_time_intervals)
#print(time_horizon)
#print(predicates)
#print(tau_nums)
#print(G_nums)
#print(F_nums)
#print(optimisation_function)
t_0 = 0
t_end = time_horizon+0.01

goal_condition = []
initial_condition = np.insert(initial_conditions,0, t_0)
goal_condition =    np.insert(initial_conditions,0, t_end)
tau = [-1]*tau_nums
Tstar = np.zeros(F_nums)
tsar = np.zeros(F_nums)

N = np.vstack((initial_condition, goal_condition))
A = MAPS2(N,optimisation_function, tau,Tstar,tsar, number_of_robots, extract_time_intervals, predicates, time_horizon)
#A.create_gradient_functions(3)
#diff_expr = A.grad(optimisation_function, [1,2,3], 3, 1, 10, 1)
#print(diff_expr)
A.TrajectoryShaping()


