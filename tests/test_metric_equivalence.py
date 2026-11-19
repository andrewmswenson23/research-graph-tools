import unittest

import networkx as nx
import numpy as np

from research_graph_tools.metrics import accessibility_index, cheeger_approximation, fiedler_vector, kemeny_constant
from research_graph_tools.metrics._graph_data import graph_data


class MetricEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.graphs = [
            nx.path_graph(4),
            nx.cycle_graph(5),
            nx.star_graph(4),
            nx.complete_graph(4),
            nx.barbell_graph(3, 1),
        ]

    def test_graph_data_returns_consistent_shared_inputs(self):
        graph = nx.path_graph(4)
        nodes, adjacency, degrees = graph_data(graph)
        self.assertEqual(nodes, list(graph.nodes))
        np.testing.assert_array_equal(adjacency, nx.to_numpy_array(graph, nodelist=nodes))
        np.testing.assert_array_equal(degrees, adjacency.sum(axis=1))

    def test_spectral_kemeny_matches_fundamental_matrix(self):
        for graph in self.graphs:
            with self.subTest(graph=graph):
                nodes = list(graph.nodes)
                adjacency = nx.to_numpy_array(graph, nodelist=nodes)
                degrees = adjacency.sum(axis=1)
                transition = adjacency / degrees[:, None]
                stationary = degrees / degrees.sum()
                fundamental = np.linalg.inv(
                    np.eye(len(nodes)) - transition + np.outer(np.ones(len(nodes)), stationary)
                )
                reference = np.trace(fundamental) - 1.0
                self.assertAlmostEqual(kemeny_constant(graph), reference, places=9)

    def test_accessibility_matches_original_hitting_time_calculation(self):
        for graph in self.graphs:
            with self.subTest(graph=graph):
                nodes = list(graph.nodes)
                adjacency = nx.to_numpy_array(graph, nodelist=nodes)
                degrees = adjacency.sum(axis=1)
                stationary = degrees / degrees.sum()
                transition = adjacency / degrees[:, None]
                fundamental = np.linalg.inv(
                    np.eye(len(nodes)) - transition + np.outer(np.ones(len(nodes)), stationary)
                )
                hitting = (np.diag(fundamental)[None, :] - fundamental) / stationary[None, :]
                np.fill_diagonal(hitting, 0.0)
                expected = stationary @ hitting
                actual = accessibility_index(graph)
                np.testing.assert_allclose([actual[node] for node in nodes], expected, atol=1e-10)

    def test_cheeger_sweep_matches_direct_cut_counts(self):
        for graph in self.graphs:
            with self.subTest(graph=graph):
                nodes = list(graph.nodes)
                degree = dict(graph.degree())
                fiedler = fiedler_vector(graph, normalized=True)
                order = sorted(nodes, key=lambda node: fiedler[node] / np.sqrt(degree[node]))
                total_volume = sum(degree.values())
                selected = set()
                direct_best = float("inf")
                for node in order[:-1]:
                    selected.add(node)
                    cut = sum(1 for u, v in graph.edges if (u in selected) != (v in selected))
                    volume = sum(degree[v] for v in selected)
                    direct_best = min(direct_best, cut / min(volume, total_volume - volume))
                self.assertAlmostEqual(cheeger_approximation(graph), direct_best, places=12)


if __name__ == "__main__":
    unittest.main()
