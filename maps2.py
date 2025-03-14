import re
import matplotlib.pyplot as plt
import numpy as np
import random
import numpy.linalg as la
from sympy import symbols, diff, sympify, Max
import copy

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
        parts = [part.strip() for part in re.split(r'\s*&\s*', formula)]

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
    def extract_time_intervals2(formula):
        parts = [part.strip() for part in re.split(r'\s*&\s*', formula)]
        intervals = []
        for part in parts:
            # Pattern to find temporal operators followed by intervals in square brackets
            pattern = r'(G|F)_\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]'
            matches = re.findall(pattern, part)
            intervals.append(matches)
        
        return intervals
    
    @staticmethod
    def rename_F_operators(formula_intervals):
        renamed_formula_intervals = []
        f_count = 0  # Counter for 'F' operators

        # Iterate through each row
        for row in formula_intervals:
            new_row = []
            # Iterate through each tuple in the row
            for operator, start, end in row:
                if operator == 'F':
                    f_count += 1  # Increment for each 'F'
                    new_operator = f'F{f_count}'  # Rename 'F' to 'F1', 'F2', etc.
                    new_row.append((new_operator, start, end))
                else:
                    new_row.append((operator, start, end))
            renamed_formula_intervals.append(new_row)

        return renamed_formula_intervals

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
    
    @staticmethod
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
        time_intervals2 = STLParser.extract_time_intervals2(formula)
        renamed_formula_intervals = STLParser.rename_F_operators(time_intervals2)
        time_horizon = STLParser.time_horizon(time_intervals)
        predicates = STLParser.extract_predicates(formula)
        tau_nums = STLParser.tau_nums(formula, predicates)
        optimisation_function = STLParser.format_predicates(predicates)
        return formula, count_and_operators, split_formula, find_max_agent, time_intervals, time_intervals2,  renamed_formula_intervals, time_horizon, predicates, tau_nums, optimisation_function
        
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
    
    def __init__(self, N, optimisation_function, tau, Tstar, tstar, number_of_robots, extract_time_intervals, extract_time_intervals2, renamed_formula_intervals, predicates, time_horizon, robust):
        self.N = N
        self.optimisation_function = optimisation_function
        self.tau = tau
        self.Tstar = Tstar
        self.tstar = tstar
        self.number_of_robots = number_of_robots
        self.extract_time_intervals = extract_time_intervals
        self.extract_time_intervals2 = extract_time_intervals2
        self.renamed_formula_intervals = renamed_formula_intervals
        self.predicates = predicates
        self.time_horizon = time_horizon
        self.robust = robust
        self.last_vd = np.zeros(shape=(len(self.extract_time_intervals), 2))
        

    def Interpolate(self, midTime, idx):
        N_mid = np.hstack((midTime, np.zeros(self.number_of_robots)))
        for i in range(1, self.number_of_robots+1):
            N_mid[i] = (self.N[idx,i] - self.N[idx-1,i]) * (midTime - self.N[idx-1,0]) / (self.N[idx,0]-self.N[idx-1,0]) + self.N[idx-1,i]
        return N_mid 
    
    def InitialSamples(self):
        for j in range(1000):
            midTime = self.time_horizon*random.random()
            idx = self.N[:,0].searchsorted(midTime)
            N_mid = self.Interpolate(midTime, idx)
            self.N = np.concatenate((self.N[:idx,], [N_mid], self.N[idx:, ]))
        return self.N 

    def vd1(self):
        self.vd = np.zeros(shape=(len(self.extract_time_intervals2), 2))
        F_index = 0
        index = 0
        for row_index, row in enumerate(self.extract_time_intervals2):
            if len(row) == 1:
                operator = row[0][0]
                start_time = row[0][1]
                end_time = row[0][2]
                if operator == 'G':
                    self.vd[row_index] = [float(start_time), float(end_time)]
                    index += 1
                elif operator.startswith('F'):
                    F_index += 1
                    if self.tau[index] == 1:
                        self.vd[row_index] = [0,0]
                    else:
                        self.vd[row_index] = [float(start_time), float(end_time)]
                    index += 1
            else:
                operator = row[0][0]
                start_time = row[0][1]
                end_time = row[0][2]
                if operator == 'G':
                    if self.tstar[F_index] == 0:
                        a = float(start_time)
                    else:
                        a = self.tstar[F_index]
                    self.vd[row_index] = [a+float(row[1][1]), a+float(row[1][2])]
                    if a+float(row[1][2]) > float(end_time)+float(row[1][2]):
                        self.vd[row_index] = [0, 0] 
                    F_index += 1
                    index += 2
                elif operator.startswith('F'):
                    if self.tstar[F_index] == 0:
                        self.vd[row_index] = [float(start_time), float(end_time)+float(row[1][2])]
                    else:
                        self.vd[row_index] = [self.tstar[F_index],self.tstar[F_index] +float(row[1][2])]
                    F_index += 1
                    index += 2
        return self.vd

                
    def SatisfactionVariable(self, midTime, x0):
        index = 0
        F_index = 0
        for row_index, row in enumerate(self.extract_time_intervals2):
            if len(row) == 1:
                operator = row[0][0]
                active = sympify(self.predicates[row_index])
                active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                if operator == 'G':
                    if self.vd[row_index][0] <= midTime <= self.vd[row_index][1]:
                        if active_subs <= 0.05:
                            pass
                elif operator.startswith('F'):
                    if self.vd[row_index][0] <= midTime <= self.vd[row_index][1]:
                        active = sympify(self.predicates[row_index])
                        active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                        if active_subs <= 0.05:
                            self.tau[index] = 1
                            self.tstar[F_index] = midTime
                    F_index += 1
                index += 1
            else:
                operator = row[0][0]
                active = sympify(self.predicates[row_index])
                active_subs = active.subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
                if operator == 'G':
                    if self.vd[row_index][0] <= midTime <= self.vd[row_index][1]:
                        if active_subs <= 0.05:
                            self.tstar[F_index] = midTime
                    F_index += 1
                elif operator.startswith('F'):
                    if self.vd[row_index][0] <= midTime <= self.vd[row_index][1]:
                        if active_subs <= 0.05:
                            pass
                    F_index += 1
                

    def activation_variable(self, midTime, agent):
        vd = self.vd1()
        lambda1 = np.zeros(len(self.extract_time_intervals)+1)
        lambda2 = np.zeros(len(self.extract_time_intervals)+1)
        agent_state = f'x{agent}'
        # Decides which predicates are active at sampled time midTime

        for i, interval in enumerate(vd):
            if interval[0] <= midTime <= interval[1]:
                lambda1[i] = 1
        for i, predicate in enumerate(self.predicates):
            if agent_state in predicate:
                lambda2[i] = 1
        lambda_ = np.logical_and(lambda1, lambda2)
        lambda_ = lambda_.astype(int)    
        return lambda_

    def activation_substitution(self, activation_lambdas):
        active_optimisation_function = self.optimisation_function
        for i, value in enumerate(activation_lambdas, 1):  # start counting from 1 for lambda1, lambda2, etc.
            active_optimisation_function = active_optimisation_function.replace('lambda' + str(i), str(value)+'*', 1)
        return active_optimisation_function

    def grad(self, x0, midTime, agent):
        activation_lambdas = self.activation_variable(midTime, agent)
        active_optimisation_function = self.activation_substitution(activation_lambdas)
        symbol_names = ' '.join('x'+str(j) for j in range(1,self.number_of_robots+1))
        symbols_tuple = symbols(symbol_names, real=True)
        expr = sympify(active_optimisation_function)
        diff_expr = diff(expr, "x"+str(agent)).subs({'x'+str(j): x0[j-1] for j in range(1,self.number_of_robots+1)})
        return diff_expr
    
    def TrajectoryShaping(self):
        for j in range(100):
            self.robust = [10/(j+1)]*(len(self.predicates))
            midTime = self.time_horizon*random.random() # Uses the random number generator to generate 'time'
            idx = self.N[:,0].searchsorted(midTime) # Searchsorted has complexity O(N)
            N_mid = self.Interpolate(midTime, idx) # Run linear interpolation here to give initial conditions for EXTRA
            N_gradient = np.hstack((midTime, self.GradientDescent(N_mid[1:], midTime)))
            self.N = np.concatenate((self.N[:idx,], [N_gradient], self.N[idx:, ]))
            print(j)
        # Uncomment the line below to save the trajectory and plot it
        return self.N
        self.plotGraph()

    def GradientDescent(self, x0, midTime):
        t = 0.01
        k = 0
        while True:
            grad = np.array([self.grad(x0, midTime, j) for j in range(1,self.number_of_robots+1)])
            deltax = - grad 
            x0 = x0 + t * deltax 
            k += 1
            grad = np.array([float(value) for value in grad])
            if  la.norm(grad) <= 0.001 or k >= 100:
                if la.norm(grad) <= 0.001:
                    self.SatisfactionVariable(midTime, x0)
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
        plt.axis([0, self.N[-1,0], -5, 5])
        plt.xlabel('time(s)')
        plt.ylabel('x')
        plt.legend()
        plt.grid(True)
        plt.show()
        np.savetxt("/Users/mayanksewlia/My Drive/NonlinearMPCManipulation/HullLeaderFollower/trajectories.csv", self.N, delimiter = ",")

