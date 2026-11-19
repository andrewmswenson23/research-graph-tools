"""Counterexample Atlas style graph comparison views."""

from types import SimpleNamespace
from typing import Optional

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research_graph_tools.metrics import accessibility_index, cheeger_approximation, cheeger_cut, kemeny_constant
from research_graph_tools.metrics._graph_data import graph_data
from research_graph_tools.visualization.layout import graph_positions

METRICS = {
    "accessibility": "accessibility",
    "fiedler_L": "Fiedler (L = D - A)",
    "fiedler_Lsym": "Fiedler (L_sym)",
    "cheeger_side": "Cheeger cut",
}
ADDED_EDGE_COLOR = "#e08a3c"
CUT_SIDE_COLORS = ("#3b6ea5", "#c0392b")


def _color_options(view: str, metric: str) -> dict:
    if view == "delta" and metric == "cheeger_side":
        return dict(colorscale="RdBu_r", cmin=-1.0, cmax=1.0)
    if view == "delta":
        return dict(colorscale="RdBu_r", cmid=0.0)
    if metric == "cheeger_side":
        return dict(colorscale=[[0.0, CUT_SIDE_COLORS[0]], [0.5, CUT_SIDE_COLORS[0]],
                                [0.5, CUT_SIDE_COLORS[1]], [1.0, CUT_SIDE_COLORS[1]]],
                    cmin=0.0, cmax=1.0)
    if metric != "accessibility":
        return dict(colorscale="RdBu_r", cmid=0.0)
    return dict(colorscale="Viridis")


def _snapshot(graph: nx.Graph) -> SimpleNamespace:
    nodes, adjacency, degrees = graph_data(graph)
    laplacian = np.diag(degrees) - adjacency
    inverse_sqrt = 1.0 / np.sqrt(degrees)
    normalized = np.eye(len(nodes)) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    values_l, vectors_l = np.linalg.eigh(laplacian)
    values_sym, vectors_sym = np.linalg.eigh(normalized)

    def orient(vector):
        vector = vector.copy()
        nonzero = np.flatnonzero(np.abs(vector) > 1e-12)
        if nonzero.size and vector[nonzero[0]] > 0:
            vector *= -1
        return vector

    accessibility = accessibility_index(graph)
    cut_side = cheeger_cut(graph)
    return SimpleNamespace(
        kemeny=kemeny_constant(graph),
        accessibility=np.array([accessibility[node] for node in nodes]),
        lambda2_L=float(values_l[1]),
        fiedler_L=orient(vectors_l[:, 1]),
        lambda2_Lsym=float(values_sym[1]),
        fiedler_Lsym=orient(vectors_sym[:, 1]),
        diameter=nx.diameter(graph),
        cheeger=cheeger_approximation(graph),
        cheeger_side=np.array([cut_side[node] for node in nodes], dtype=float),
        cheeger_exact=False,
    )


def _title(values: SimpleNamespace, view: str) -> str:
    d = "Δ" if view == "delta" else ""
    return ("<b>{}</b><br>{}K = {:.5f}  {}λ₂(L) = {:.5f}  "
            "{}λ₂(L_sym) = {:.5f}  {}h (sweep) = {:.5f}  {}diam = {}".format(
                view, d, values.kemeny, d, values.lambda2_L, d, values.lambda2_Lsym,
                d, values.cheeger, d,
                "{:+d}".format(values.diameter) if view == "delta" else values.diameter))


def _normalize_graph(graph, added_edge, positions, use_default_edge=True):
    """Accept adjacency matrices, NetworkX graphs, and Atlas graph records."""
    if isinstance(graph, np.ndarray):
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("adjacency matrix must be square")
        if not np.array_equal(graph, graph.T):
            raise ValueError("adjacency matrix must be symmetric")
        if not np.isin(graph, (0, 1)).all() or np.diag(graph).any():
            raise ValueError("adjacency matrix must contain only 0/1 values and no self-loops")
        graph = nx.from_numpy_array(graph)
        return graph, added_edge, positions, "Graph"

    if isinstance(graph, nx.Graph):
        metadata = graph.graph
        if positions is None:
            positions = metadata.get("layout", metadata.get("pos"))
        name = metadata.get("name", "") or "Graph"
        if added_edge is None and use_default_edge:
            added_edge = metadata.get("default_added_edge")
            defaults = metadata.get("default_added_edges", ())
            if added_edge is None and defaults:
                if len(defaults) != 1:
                    raise ValueError("pass added_edge explicitly when the graph has multiple default added edges")
                added_edge = defaults[0]
        return graph, added_edge, positions, name

    if all(hasattr(graph, name) for name in ("vertex_ids", "edges", "layout")):
        converted = nx.Graph()
        converted.add_nodes_from(graph.vertex_ids)
        converted.add_edges_from(graph.edges)
        if positions is None:
            positions = graph.layout
        if added_edge is None and use_default_edge:
            defaults = getattr(graph, "default_added_edges", ())
            if len(defaults) > 1:
                raise ValueError("pass added_edge explicitly when the graph has multiple default added edges")
            added_edge = defaults[0] if defaults else None
        return converted, added_edge, positions, getattr(graph, "name", "Graph")
    raise TypeError("expected a NetworkX graph or Counterexample Atlas graph")


def draw_graph(
    graph,
    added_edge: tuple | None = None,
    *,
    show_augmented: bool = False,
    show_delta: bool = False,
    metric: str = "accessibility",
    positions: dict | None = None,
    panel_height: int = 420,
    width: Optional[int] = None,
    stack: str = "rows",
    node_size: Optional[float] = None,
    show_labels: Optional[bool] = None,
    label_size: int = 9,
    edge_width: float = 1.0,
    added_edge_width: float = 2.5,
) -> go.Figure:
    """Render an Atlas style graph, or compare baseline/augmented/delta views.

    A single dropdown colors every panel by accessibility, either Fiedler vector,
    or Cheeger cut side. Hovering a node shows all four per-node metrics.
    """
    graph, added_edge, positions, graph_name = _normalize_graph(
        graph, added_edge, positions, use_default_edge=show_augmented or show_delta)
    if graph.is_directed() or graph.is_multigraph() or not nx.is_connected(graph):
        raise ValueError("graph must be a connected, undirected simple graph")
    if metric not in METRICS:
        raise ValueError("metric must be one of {}".format(tuple(METRICS)))
    if stack not in ("rows", "columns"):
        raise ValueError("stack must be 'rows' or 'columns'")
    if added_edge is not None:
        if len(added_edge) != 2:
            raise ValueError("added_edge must contain two vertices")
        left, right = added_edge
        if left == right or left not in graph or right not in graph or graph.has_edge(left, right):
            raise ValueError("added_edge must connect two distinct existing non-adjacent vertices")
    else:
        left = right = None
        if show_augmented or show_delta:
            raise ValueError("show_augmented or show_delta requires an added edge")

    nodes = list(graph.nodes)
    pos = graph_positions(graph, positions)
    baseline = _snapshot(graph)
    augmented_graph = graph.copy()
    if added_edge is not None:
        augmented_graph.add_edge(left, right)
    augmented = _snapshot(augmented_graph)
    # Match Fiedler signs before taking differences.
    for name in ("fiedler_L", "fiedler_Lsym"):
        if np.dot(getattr(baseline, name), getattr(augmented, name)) < 0:
            setattr(augmented, name, -getattr(augmented, name))

    views = ["baseline"] if added_edge is None else ["baseline", "augmented"]
    if show_delta:
        views.append("delta")
    snapshots = {"baseline": baseline, "augmented": augmented}
    if show_delta:
        delta = SimpleNamespace(**{
            key: (getattr(augmented, key) - getattr(baseline, key)
                  if key not in {"cheeger_exact"} else False)
            for key in ("kemeny", "accessibility", "lambda2_L", "fiedler_L", "lambda2_Lsym",
                        "fiedler_Lsym", "diameter", "cheeger", "cheeger_side", "cheeger_exact")
        })
        snapshots["delta"] = delta

    if node_size is None:
        node_size = 14 if len(nodes) <= 30 else 8 if len(nodes) <= 100 else 5
    if show_labels is None:
        show_labels = len(nodes) <= 30
    rows, cols = (len(views), 1) if stack == "rows" else (1, len(views))
    titles = [_title(snapshots[view], view) for view in views]
    figure = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                           vertical_spacing=0.12 / max(rows - 1, 1) if rows > 1 else 0.0,
                           horizontal_spacing=0.12 if cols > 1 else 0.0)
    node_x = [pos[node][0] for node in nodes]
    node_y = [pos[node][1] for node in nodes]
    metric_names = list(METRICS)
    trace_metrics = []
    for index, view in enumerate(views):
        row, col = (index + 1, 1) if stack == "rows" else (1, index + 1)
        values = snapshots[view]
        axis = "" if index == 0 else str(index + 1)
        x_domain = figure.layout["xaxis" + axis].domain
        y_domain = figure.layout["yaxis" + axis].domain

        edge_x, edge_y = [], []
        for u, v in graph.edges:
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])
        figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="skip",
                                    line=dict(color="#bbbbbb", width=edge_width), showlegend=False),
                         row=row, col=col)
        if added_edge is None:
            figure.add_trace(go.Scatter(x=[], y=[], mode="lines", hoverinfo="skip", showlegend=False),
                             row=row, col=col)
        else:
            figure.add_trace(go.Scatter(
                x=[pos[left][0], pos[right][0]], y=[pos[left][1], pos[right][1]],
                mode="lines", hoverinfo="skip", showlegend=False,
                line=dict(color=ADDED_EDGE_COLOR, width=added_edge_width, dash="dash"),
                opacity=0.45 if view == "baseline" else 1.0), row=row, col=col)
        trace_metrics += [None, None]

        hover = [
            "<b>{}</b><br>accessibility: {:.5g}<br>Fiedler L: {:.5g}<br>Fiedler L_sym: {:.5g}"
            "<br>Cheeger side: {:+.0f}".format(
                node, values.accessibility[i], values.fiedler_L[i], values.fiedler_Lsym[i],
                values.cheeger_side[i])
            for i, node in enumerate(nodes)
        ]
        for name in metric_names:
            colorbar = dict(title=METRICS[name] if view != "delta" else "Δ " + METRICS[name],
                            x=x_domain[1] + 0.01, xanchor="left",
                            y=(y_domain[0] + y_domain[1]) / 2, yanchor="middle",
                            len=y_domain[1] - y_domain[0])
            if name == "cheeger_side" and view != "delta":
                colorbar.update(tickvals=[0.25, 0.75], ticktext=["side 0", "side 1"])
            figure.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode="markers+text" if show_labels else "markers",
                text=[str(node) for node in nodes] if show_labels else None,
                textposition="top center", textfont=dict(size=label_size),
                hovertext=hover, hoverinfo="text", showlegend=False,
                visible=name == metric,
                marker=dict(size=node_size, color=np.asarray(getattr(values, name)),
                            line=dict(color="white", width=0.8), colorbar=colorbar,
                            **_color_options(view, name)),
            ), row=row, col=col)
            trace_metrics.append(name)

        figure.update_xaxes(visible=False, fixedrange=True, row=row, col=col)
        figure.update_yaxes(visible=False, fixedrange=True, scaleanchor="x" + axis, row=row, col=col)

    buttons = []
    for name in metric_names:
        visible = [owner is None or owner == name for owner in trace_metrics]
        buttons.append(dict(label=METRICS[name], method="update", args=[{"visible": visible}]))
    figure.update_annotations(font=dict(size=12 if stack == "rows" else 11))
    figure.update_layout(
        title=dict(text="<b>{}</b>".format(graph_name),
                   font=dict(size=14), x=0.5, xanchor="center", yref="container", y=0.995,
                   yanchor="top", pad=dict(t=8)),
        updatemenus=[dict(buttons=buttons, active=metric_names.index(metric), x=0.0,
                          xanchor="left", y=1.0, yanchor="bottom", pad=dict(b=40))],
        dragmode=False, plot_bgcolor="white", margin=dict(t=110, l=10, r=90, b=10),
        height=panel_height * rows + 110, width=width,
    )
    return figure
