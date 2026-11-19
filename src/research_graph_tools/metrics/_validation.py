"""Shared input validation for graph metrics."""

import networkx as nx


def require_connected(graph: nx.Graph) -> None:
    # These calculations model a simple random walk, so directed and parallel
    # edges would change the meaning of the result.
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("expected an undirected simple graph")
    if len(graph) < 2 or not nx.is_connected(graph):
        raise ValueError("graph must be connected and have at least 2 vertices")
    # Isolated vertices cannot take a random-walk step (their degree is zero).
    if any(degree == 0 for _, degree in graph.degree()):
        raise ValueError("graph must have no isolated vertices")
