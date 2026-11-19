# Research Graph Tools

A small Python toolkit for calculations and visualizations in Kempton graph research. Version 0.1 focuses on undirected, unweighted graphs and keeps each tool in a small, separate module.

## Install

```bash
python -m pip install -e .
```

## Quick start

```python
from research_graph_tools.graphs import path_graph
from research_graph_tools.metrics import kemeny_constant, accessibility_index
from research_graph_tools.visualization import draw_graph

graph = path_graph(6)
print(kemeny_constant(graph))
print(accessibility_index(graph))
draw_graph(graph, color_values=accessibility_index(graph)).show()
```

See [`examples/quickstart.py`](examples/quickstart.py) for a minimal example including all metrics and a bounded search.
