"""Fiedler vectors for graph Laplacians."""

import networkx as nx
import numpy as np

from research_graph_tools.metrics._graph_data import graph_data


def _fiedler_from_data(nodes: list, adjacency: np.ndarray, degrees: np.ndarray, normalized: bool) -> dict:
    if normalized:
        inv = 1.0 / np.sqrt(degrees)
        laplacian = np.eye(len(nodes)) - inv[:, None] * adjacency * inv[None, :]
    else:
        laplacian = np.diag(degrees) - adjacency
    _, vectors = np.linalg.eigh(laplacian)
    vector = vectors[:, 1].copy()
    first = np.flatnonzero(np.abs(vector) > 1e-12)
    if first.size and vector[first[0]] > 0:
        vector *= -1
    return dict(zip(nodes, vector))


def fiedler_vector(graph: nx.Graph, normalized: bool = False) -> dict:
    """Return the second Laplacian eigenvector indexed by node.

    Set ``normalized=True`` to use the symmetric normalized Laplacian.
    """
    return _fiedler_from_data(*graph_data(graph), normalized)
