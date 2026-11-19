"""Fiedler sweep approximation to graph conductance and its cut."""

import networkx as nx
import numpy as np

from research_graph_tools.metrics._graph_data import graph_data
from research_graph_tools.metrics.fiedler import _fiedler_from_data


def _sweep(graph: nx.Graph) -> tuple[float, dict]:
    nodes, adjacency, degrees_array = graph_data(graph)
    degrees = dict(zip(nodes, degrees_array))
    fiedler = _fiedler_from_data(nodes, adjacency, degrees_array, normalized=True)
    order = sorted(nodes, key=lambda node: fiedler[node] / np.sqrt(degrees[node]))
    total_volume = float(degrees_array.sum())
    selected = set()
    volume = 0.0
    cut = 0.0
    best_value = float("inf")
    best_side = set()
    for node in order[:-1]:
        # Adding a node creates one cut edge per incident edge, then removes
        # two for each edge that was already crossing into the selected set.
        cut += degrees[node] - 2 * sum(neighbor in selected for neighbor in graph.neighbors(node))
        selected.add(node)
        volume += degrees[node]
        value = cut / min(volume, total_volume - volume)
        if value < best_value:
            best_value, best_side = value, set(selected)
    return float(best_value), {node: node in best_side for node in nodes}


def cheeger_cut(graph: nx.Graph) -> dict:
    """Return the side of the best normalized-Fiedler sweep cut for each node."""
    return _sweep(graph)[1]


def cheeger_approximation(graph: nx.Graph) -> float:
    """Return conductance of the best sweep cut along the normalized Fiedler vector."""
    return _sweep(graph)[0]
