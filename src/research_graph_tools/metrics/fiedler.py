"""Fiedler vectors for graph Laplacians."""

import networkx as nx
import numpy as np
from research_graph_tools.metrics._validation import require_connected


def fiedler_vector(graph: nx.Graph, normalized: bool = False) -> dict:
    """Return the second Laplacian eigenvector indexed by node.

    Set ``normalized=True`` to use the symmetric normalized Laplacian.
    """
    require_connected(graph)
    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    degrees = adjacency.sum(axis=1)
    if normalized:
        inv = 1.0 / np.sqrt(degrees)
        laplacian = np.eye(len(nodes)) - inv[:, None] * adjacency * inv[None, :]
    else:
        laplacian = np.diag(degrees) - adjacency
    values, vectors = np.linalg.eigh(laplacian)
    vector = vectors[:, 1].copy()
    first = np.flatnonzero(np.abs(vector) > 1e-12)
    if first.size and vector[first[0]] > 0:
        vector *= -1
    return dict(zip(nodes, vector))
