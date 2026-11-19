"""Minimal constructors for common undirected graph families."""

import networkx as nx


def path_graph(n: int) -> nx.Graph:
    """Return a path on ``n`` vertices."""
    if n < 1:
        raise ValueError("n must be at least 1")
    return nx.path_graph(n)


def cycle_graph(n: int) -> nx.Graph:
    """Return a cycle on ``n`` vertices."""
    if n < 3:
        raise ValueError("a cycle requires at least 3 vertices")
    return nx.cycle_graph(n)
