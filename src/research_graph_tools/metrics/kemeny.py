"""Kemeny's constant for the simple random walk on a graph."""

import networkx as nx
import numpy as np

from research_graph_tools.metrics._graph_data import graph_data


def kemeny_constant(graph: nx.Graph) -> float:
    """Return Kemeny's constant for the graph's simple random walk."""
    _, adjacency, degrees = graph_data(graph)
    inverse_sqrt = 1.0 / np.sqrt(degrees)
    normalized_laplacian = np.eye(len(degrees)) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    eigenvalues = np.linalg.eigvalsh(normalized_laplacian)
    return float(np.reciprocal(eigenvalues[1:]).sum())
