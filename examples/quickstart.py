"""Minimal example of the research graph tools."""

from research_graph_tools.graphs import path_graph
from research_graph_tools.metrics import accessibility_index, cheeger_approximation, fiedler_vector, kemeny_constant
from research_graph_tools.search import find_counterexample
from research_graph_tools.visualization import draw_graph


graph = path_graph(6)
print("Kemeny:", kemeny_constant(graph))
print("Accessibility:", accessibility_index(graph))
print("Fiedler:", fiedler_vector(graph))
print("Cheeger approximation:", cheeger_approximation(graph))

counterexample = find_counterexample([path_graph(n) for n in range(3, 10)], lambda g: kemeny_constant(g) > 5, limit=7)
print("Found counterexample:", counterexample is not None)

draw_graph(graph, color_values=accessibility_index(graph)).show()
