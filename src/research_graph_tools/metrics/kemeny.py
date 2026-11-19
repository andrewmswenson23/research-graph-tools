"""Kemeny's constant for the simple random walk on a graph."""

import networkx as nx
import numpy as np
from research_graph_tools.metrics._validation import require_connected


def kemeny_constant(graph: nx.Graph) -> float:
    """Return Kemeny's constant for the graph's simple random walk."""
    require_connected(graph)
    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    degrees = adjacency.sum(axis=1)
    transition = adjacency / degrees[:, None]
    stationary = degrees / degrees.sum()
    fundamental = np.linalg.inv(np.eye(len(nodes)) - transition + np.outer(np.ones(len(nodes)), stationary))
    return float(np.trace(fundamental) - 1.0)
