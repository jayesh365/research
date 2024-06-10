from graphviz import Source

with open('Digraph.gv', 'r') as file:
    dot_graph = file.read()

graph = Source(dot_graph)
graph.render('output', format='png', view=True)
