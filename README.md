# Research Graph Tools

Small calculation, search, and visualization tools for research on undirected, unweighted graphs.

## Install

```bash
uv sync
```

## Quick start

```python
import numpy as np
from research_graph_tools.visualization import draw_graph

# A 4-cycle, with rows and columns identifying vertices 0 through 3.
adjacency = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
])

draw_graph(adjacency).show()  # one graph
# Add edge 0--2; show baseline, augmented, and delta panels.
draw_graph(adjacency, (0, 2), show_delta=True).show()
```

`draw_graph` accepts a symmetric 0/1 NumPy adjacency matrix, a NetworkX graph, or a Counterexample Atlas graph record. It assigns a layout automatically; pass `positions=` to set one yourself. Hover over nodes to see accessibility, both Fiedler values, and Cheeger-cut side. The metric dropdown recolors all panels together.

## Tools

- `research_graph_tools.metrics`: Kemeny's constant, accessibility index, Fiedler vectors, and Cheeger sweep approximation.
- `research_graph_tools.graphs`: small path and cycle constructors.
- `research_graph_tools.search`: first-match search over caller-provided candidates.
- `research_graph_tools.visualization`: single-graph and baseline/augmented/delta views.

See the [notebook walkthrough](notebooks/walkthrough.ipynb) for examples.

## Correctness checks

```bash
uv run python -m unittest discover -s tests
```
