"""Kemeny's constant for the simple random walk on a graph."""

import networkx as nx
import numpy as np
from research_graph_tools.metrics._validation import require_connected


def kemeny_constant(graph: nx.Graph) -> float:
    """Return Kemeny's constant for the graph's simple random walk."""
    require_connected(graph)
    # Keep a node list so matrix rows and columns map back to graph nodes.
    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    # A random walk chooses uniformly among a node's neighbors.
    degrees = adjacency.sum(axis=1)
    transition = adjacency / degrees[:, None]
    # The long-run visit probability is proportional to node degree.
    stationary = degrees / degrees.sum()
    # This matrix form lets us compute Kemeny's constant from one inverse.
    fundamental = np.linalg.inv(np.eye(len(nodes)) - transition + np.outer(np.ones(len(nodes)), stationary))
    return float(np.trace(fundamental) - 1.0)
