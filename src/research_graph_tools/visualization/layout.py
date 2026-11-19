"""Deterministic default graph layout."""

import networkx as nx


def graph_positions(graph: nx.Graph, positions: dict | None = None) -> dict:
    """Return supplied positions or a deterministic spring layout."""
    if positions is not None:
        missing = set(graph) - set(positions)
        if missing:
            raise ValueError(f"positions missing nodes: {missing}")
        return positions
    return nx.spring_layout(graph, seed=0)
