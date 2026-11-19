"""Shared matrix inputs for graph metrics."""

import networkx as nx
import numpy as np

from research_graph_tools.metrics._validation import require_connected


def graph_data(graph: nx.Graph) -> tuple[list, np.ndarray, np.ndarray]:
    """Return nodes, adjacency matrix, and degrees in a consistent node order."""
    require_connected(graph)
    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    degrees = adjacency.sum(axis=1)
    return nodes, adjacency, degrees
