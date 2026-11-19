"""Plotly graph drawing with optional per-node metric coloring."""

import networkx as nx
import plotly.graph_objects as go

from research_graph_tools.visualization.layout import graph_positions


def draw_graph(graph: nx.Graph, color_values: dict | None = None, positions: dict | None = None, node_size: int = 14) -> go.Figure:
    """Return a Plotly figure for an undirected graph."""
    pos = graph_positions(graph, positions)
    edge_x, edge_y = [], []
    for left, right in graph.edges:
        edge_x.extend([pos[left][0], pos[right][0], None])
        edge_y.extend([pos[left][1], pos[right][1], None])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"color": "#777", "width": 1}, hoverinfo="none"))
    nodes = list(graph.nodes)
    marker = {"size": node_size, "color": [color_values.get(node, 0) for node in nodes] if color_values else "#4c78a8", "colorscale": "RdBu", "showscale": color_values is not None}
    figure.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=[str(n) for n in nodes], textposition="top center", marker=marker, hovertext=[str(n) for n in nodes], hoverinfo="text"))
    figure.update_layout(showlegend=False, margin={"l": 10, "r": 10, "t": 20, "b": 10}, xaxis={"visible": False}, yaxis={"visible": False}, template="plotly_white")
    return figure
