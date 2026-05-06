"""Graph feature extraction for astrocyte networks."""

import numpy as np
from typing import Any

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from neuros_astro.metadata.schema import AstroGraph


def compute_graph_summary_features(graph: AstroGraph) -> dict[str, Any]:
    """
    Compute summary features from an astrocyte functional graph.

    Features:
    - n_nodes: Number of nodes
    - n_edges: Number of edges
    - density: Edge density
    - mean_edge_weight: Mean edge weight
    - max_edge_weight: Max edge weight
    - degree_mean: Mean node degree
    - degree_max: Max node degree
    - n_connected_components: Number of connected components (if networkx available)

    Args:
        graph: AstroGraph object

    Returns:
        Dictionary of graph features
    """
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

    # Basic features
    features = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
    }

    # Density
    if n_nodes > 1:
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0.0
    else:
        density = 0.0

    features["density"] = density

    # Edge weight statistics
    if n_edges > 0:
        features["mean_edge_weight"] = float(np.mean(graph.edge_weights))
        features["max_edge_weight"] = float(np.max(graph.edge_weights))
        features["std_edge_weight"] = float(np.std(graph.edge_weights))
    else:
        features["mean_edge_weight"] = 0.0
        features["max_edge_weight"] = 0.0
        features["std_edge_weight"] = 0.0

    # Degree statistics
    if n_nodes > 0 and n_edges > 0:
        # Compute degree for each node
        node_degrees = {node: 0 for node in graph.nodes}

        for edge in graph.edges:
            node_degrees[edge[0]] += 1
            node_degrees[edge[1]] += 1

        degrees = list(node_degrees.values())
        features["degree_mean"] = float(np.mean(degrees))
        features["degree_max"] = int(np.max(degrees))
        features["degree_std"] = float(np.std(degrees))
    else:
        features["degree_mean"] = 0.0
        features["degree_max"] = 0
        features["degree_std"] = 0.0

    # Connected components (requires networkx)
    if HAS_NETWORKX and n_nodes > 0:
        G = nx.Graph()
        G.add_nodes_from(graph.nodes)
        G.add_edges_from(graph.edges)

        n_components = nx.number_connected_components(G)
        features["n_connected_components"] = n_components

        # Largest component size
        if n_components > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            features["largest_component_size"] = len(largest_cc)
        else:
            features["largest_component_size"] = 0
    else:
        # Fallback: assume all nodes in one component if any edges, else each node is a component
        if n_edges > 0:
            features["n_connected_components"] = 1
            features["largest_component_size"] = n_nodes
        else:
            features["n_connected_components"] = n_nodes
            features["largest_component_size"] = 1 if n_nodes > 0 else 0

    return features


def graphs_to_feature_matrix(graphs: list[AstroGraph]) -> tuple[np.ndarray, list[str]]:
    """
    Convert list of graphs to feature matrix for analysis or modeling.

    Args:
        graphs: List of AstroGraph objects

    Returns:
        feature_matrix: Array [n_graphs, n_features]
        feature_names: List of feature names
    """
    if len(graphs) == 0:
        return np.array([]), []

    # Compute features for each graph
    all_features = [compute_graph_summary_features(g) for g in graphs]

    # Extract feature names (assume all graphs have same features)
    feature_names = list(all_features[0].keys())

    # Build matrix
    feature_matrix = np.array([[f[name] for name in feature_names] for f in all_features])

    return feature_matrix, feature_names
