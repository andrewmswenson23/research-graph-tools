"""Per-vertex accessibility for the simple random walk."""

import networkx as nx
import numpy as np

from research_graph_tools.metrics._graph_data import graph_data


def accessibility_index(graph: nx.Graph) -> dict:
    """Return accessibility indexed by node, using stationary-weighted mean hitting time."""
    nodes, adjacency, degrees = graph_data(graph)
    stationary = degrees / degrees.sum()
    transition = adjacency / degrees[:, None]
    fundamental = np.linalg.inv(np.eye(len(nodes)) - transition + np.outer(np.ones(len(nodes)), stationary))
    hitting = (np.diag(fundamental)[None, :] - fundamental) / stationary[None, :]
    np.fill_diagonal(hitting, 0.0)
    return dict(zip(nodes, stationary @ hitting))
