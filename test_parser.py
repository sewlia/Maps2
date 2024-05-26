import re

def extract_time_intervals(formula):
    # Initialize list to hold the structured intervals
    structured_intervals = []

    # Split the formula on the logical AND operator
    parts = [part.strip() for part in re.split(r'\s*∧\s*', formula)]

    for part in parts:
        # Extract nested temporal operators with their intervals
        pattern = r'(G|F)_\[(\d+),(\d+)\]'
        matches = re.findall(pattern, part)
        structured_intervals.append(matches)

    return structured_intervals

formula = 'G_[1,5]F_[3,7] ∧ F_[8,12]'
nested_intervals = extract_time_intervals(formula)
print(nested_intervals)


from graphviz import Digraph

def build_tree_from_intervals(intervals, graph, parent_id):
    """ Build the tree from nested interval tuples """
    for seq in intervals:
        last_node_id = parent_id
        for operator, t1, t2 in reversed(seq):
            current_id = f"{operator}_{t1}_{t2}"
            label = f"{operator}_[{t1},{t2}]"
            graph.node(current_id, label)
            graph.edge(last_node_id, current_id)
            last_node_id = current_id

def main():
    formula = 'G_[1,5]F_[3,7] ∧ F_[8,12]'
    intervals = extract_time_intervals(formula)
    
    # Initialize the graph with a root node
    dot = Digraph(comment='STL Parse Tree')
    root_id = "root"
    dot.node(root_id, "\\land")

    # Build the tree for each conjunction part
    for interval in intervals:
        build_tree_from_intervals(interval, dot, root_id)
    
    # Render and view the graph
    dot.render('stl_parse_tree', format='png', view=True)

if __name__ == "__main__":
    main()
