import re

def remove_spaces(formula):
    return formula.replace(" ","")  # Removes all spaces in the formula

def count_and_operators(formula):
    return formula.count('&')  # Counts each instance of '&' 

def split_formula(formula):
    return formula.split('&')  # Splits the formula at each '&'

def find_max_agent(formula):
    # Regex to find patterns like x1, x2, x30, etc.
    agent_pattern = r'x(\d+)'
    agents = re.findall(agent_pattern, formula)
    agent_numbers = map(int, agents)
    return max(agent_numbers)

# Create the parse tree
formula = 'G_[1,2](||x1-x2||<=2) & F_[5,6](||x2-x3||<=1) & G_[3,4]F_[4,5](||x1-x3||<=3)'

formula = remove_spaces(formula)
print(formula)
subformula = split_formula(formula)
print(count_and_operators(formula))  
print(find_max_agent(formula))

state_dimensions = 3

