"""Per-vertex accessibility for the simple random walk."""

import networkx as nx
import numpy as np
from research_graph_tools.metrics._validation import require_connected


def accessibility_index(graph: nx.Graph) -> dict:
    """Return accessibility indexed by node, using the stationary-weighted mean hitting time."""
    require_connected(graph)
    # Keep the matrix order consistent so results can be labeled by node later.
    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    degrees = adjacency.sum(axis=1)
    # Stationary probabilities describe where the walk spends its long-run time.
    stationary = degrees / degrees.sum()
    transition = adjacency / degrees[:, None]
    # The fundamental matrix is used to derive expected hitting times.
    fundamental = np.linalg.inv(np.eye(len(nodes)) - transition + np.outer(np.ones(len(nodes)), stationary))
    hitting = (np.diag(fundamental)[None, :] - fundamental) / stationary[None, :]
    # A node's hitting time to itself is zero by definition.
    np.fill_diagonal(hitting, 0.0)
    # Average each target's hitting times over possible starting nodes.
    return dict(zip(nodes, stationary @ hitting))
