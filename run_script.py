import numpy as np
import random
from maps2 import STLParser, UserInput, MAPS2

#formula = 'F_[12,13](x4<=8) & G_[1,6] F_[0,1](x1<=2) & G_[5,6](x3>=3) & F_[7,8](x4<=8) & G_[9,10](x5>=8)'
formula = 'G_[1,2](x1**2<=2) & G_[1,2](x1**2>=1.5) & G_[1,2](x1**2-x2**2>=0.01)  & F_[3,4](x3>=3) & G_[5,10]F_[0,1](x4<=4) & F_[7,8](x5>=6) & G_[9,10](x6<=8)'
formula, count_and_operators, split_formula, number_of_robots, extract_time_intervals, extract_time_intervals2, renamed_formula_intervals, time_horizon, predicates, tau_nums, optimisation_function = STLParser.return_function(formula)
G_nums = formula.count('G')
F_nums = formula.count('F')

initial_conditions = [5, 4, 1, 7, 3, 9]
print('STL Formula', formula)
print('Total number of robots', number_of_robots)
print('All Branches of the STL formula', extract_time_intervals2)
print('Time horizon', time_horizon)
print('Total number of always operators', G_nums)
print('Total number of always operators', F_nums)
print('Optimisation Function', optimisation_function)

t_0 = 0
t_end = time_horizon+0.01

goal_condition = []
initial_condition = np.insert(initial_conditions,0, t_0)
goal_condition =    np.insert(initial_conditions,0, t_end)
tau = [-1]*(G_nums+F_nums)
robust = [None]*(len(predicates))
Tstar = np.zeros(F_nums)
tsar = np.zeros(F_nums)

N = np.vstack((initial_condition, goal_condition))

A = MAPS2(N,optimisation_function, tau,Tstar,tsar, number_of_robots, extract_time_intervals, extract_time_intervals2, renamed_formula_intervals, predicates, time_horizon, robust)

A.TrajectoryShaping()


