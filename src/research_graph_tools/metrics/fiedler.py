"""Fiedler vectors for graph Laplacians."""

import networkx as nx
import numpy as np
from research_graph_tools.metrics._validation import require_connected


def fiedler_vector(graph: nx.Graph, normalized: bool = False) -> dict:
    """Return the second Laplacian eigenvector indexed by node.

    Set ``normalized=True`` to use the symmetric normalized Laplacian.
    """
    require_connected(graph)
    # Matrix row i always corresponds to nodes[i].
    nodes = list(graph.nodes)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    degrees = adjacency.sum(axis=1)
    if normalized:
        # Scaling by degrees reduces the influence of high-degree vertices.
        inv = 1.0 / np.sqrt(degrees)
        laplacian = np.eye(len(nodes)) - inv[:, None] * adjacency * inv[None, :]
    else:
        laplacian = np.diag(degrees) - adjacency
    # Eigenvalues are returned in order; column 1 is the second eigenvector.
    values, vectors = np.linalg.eigh(laplacian)
    vector = vectors[:, 1].copy()
    # Eigenvectors may point either way; choose one sign for stable examples.
    first = np.flatnonzero(np.abs(vector) > 1e-12)
    if first.size and vector[first[0]] > 0:
        vector *= -1
    return dict(zip(nodes, vector))
