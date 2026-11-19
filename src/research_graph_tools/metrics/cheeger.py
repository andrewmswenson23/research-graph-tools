"""Fiedler sweep approximation to graph conductance."""

import networkx as nx
import numpy as np
from research_graph_tools.metrics.fiedler import fiedler_vector
from research_graph_tools.metrics._validation import require_connected


def cheeger_approximation(graph: nx.Graph) -> float:
    """Return conductance of the best sweep cut along the normalized Fiedler vector."""
    require_connected(graph)
    nodes = list(graph.nodes)
    degrees = dict(graph.degree())
    vector = fiedler_vector(graph, normalized=True)
    # Sweep through vertices in spectral order; each prefix is a possible cut.
    order = sorted(nodes, key=lambda node: vector[node] / np.sqrt(degrees[node]))
    best = float("inf")
    selected = set()
    total_volume = sum(degrees.values())
    for node in order[:-1]:
        selected.add(node)
        # Conductance compares edges leaving this group with its smaller side.
        volume = sum(degrees[v] for v in selected)
        cut = sum(1 for u in selected for v in graph.neighbors(u) if v not in selected)
        best = min(best, cut / min(volume, total_volume - volume))
    return float(best)
