"""
Advanced network analysis metrics for astrocyte functional connectivity.

Provides temporal network analysis, stability metrics, community detection,
and motif analysis.
"""

import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from neuros_astro.metadata.schema import AstroGraph

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("networkx not installed. Install with: pip install networkx")


@dataclass
class NetworkMetrics:
    """Comprehensive network metrics for astrocyte networks."""

    # Basic metrics (required fields first)
    n_nodes: int
    n_edges: int
    density: float

    # Centrality metrics
    mean_degree: float
    max_degree: int
    degree_centrality_std: float

    # Clustering and community
    global_clustering: float
    transitivity: float

    # Connectivity
    n_components: int
    largest_component_size: int

    # Strength (weighted)
    mean_strength: float
    max_strength: float

    # Time window
    window_start_s: float
    window_end_s: float

    # Optional fields (with defaults) come last
    n_communities: Optional[int] = None
    mean_path_length: Optional[float] = None


def compute_temporal_network_metrics(
    graphs: List[AstroGraph],
) -> List[NetworkMetrics]:
    """
    Compute comprehensive metrics for a sequence of temporal networks.

    Args:
        graphs: List of AstroGraph objects in temporal order

    Returns:
        List of NetworkMetrics objects, one per graph

    Example:
        >>> from neuros_astro.networks import build_event_coactivation_graph
        >>> graphs = build_event_coactivation_graph(events, session_id="demo",
        ...                                         frame_rate_hz=10.0)
        >>> metrics = compute_temporal_network_metrics(graphs)
        >>> for m in metrics:
        ...     print(f"t=[{m.window_start_s:.1f}, {m.window_end_s:.1f}]s: "
        ...           f"density={m.density:.3f}, n_communities={m.n_communities}")
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required. Install with: pip install networkx")

    metrics_list = []

    for graph in graphs:
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(graph.nodes)

        # Add weighted edges
        for (source, target), weight in zip(graph.edges, graph.edge_weights):
            G.add_edge(source, target, weight=weight)

        # Basic metrics
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes > 1:
            density = nx.density(G)
        else:
            density = 0.0

        # Degree metrics
        degrees = dict(G.degree())
        if degrees:
            mean_degree = np.mean(list(degrees.values()))
            max_degree = max(degrees.values())
            degree_centrality = nx.degree_centrality(G)
            degree_centrality_std = np.std(list(degree_centrality.values()))
        else:
            mean_degree = 0.0
            max_degree = 0
            degree_centrality_std = 0.0

        # Clustering
        if n_nodes > 2:
            global_clustering = nx.average_clustering(G)
            transitivity = nx.transitivity(G)
        else:
            global_clustering = 0.0
            transitivity = 0.0

        # Components
        if n_nodes > 0:
            components = list(nx.connected_components(G))
            n_components = len(components)
            largest_component_size = len(max(components, key=len))

            # Mean path length (only for connected graphs)
            if n_components == 1 and n_nodes > 1:
                mean_path_length = nx.average_shortest_path_length(G)
            else:
                mean_path_length = None
        else:
            n_components = 0
            largest_component_size = 0
            mean_path_length = None

        # Community detection (Louvain if enough nodes/edges)
        if n_nodes > 3 and n_edges > 0:
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G)
                n_communities = len(set(communities.values()))
            except ImportError:
                n_communities = None
        else:
            n_communities = None

        # Weighted strength
        strengths = []
        for node in G.nodes():
            strength = sum(G[node][neighbor].get('weight', 1.0)
                          for neighbor in G.neighbors(node))
            strengths.append(strength)

        if strengths:
            mean_strength = np.mean(strengths)
            max_strength = np.max(strengths)
        else:
            mean_strength = 0.0
            max_strength = 0.0

        # Create metrics object
        metrics = NetworkMetrics(
            n_nodes=n_nodes,
            n_edges=n_edges,
            density=density,
            mean_degree=mean_degree,
            max_degree=max_degree,
            degree_centrality_std=degree_centrality_std,
            global_clustering=global_clustering,
            transitivity=transitivity,
            n_communities=n_communities,
            n_components=n_components,
            largest_component_size=largest_component_size,
            mean_path_length=mean_path_length,
            mean_strength=mean_strength,
            max_strength=max_strength,
            window_start_s=graph.window_start_s,
            window_end_s=graph.window_end_s,
        )

        metrics_list.append(metrics)

    return metrics_list


def compute_network_stability(
    graphs: List[AstroGraph],
    method: str = "jaccard",
) -> np.ndarray:
    """
    Compute pairwise stability between consecutive networks.

    Args:
        graphs: List of AstroGraph objects in temporal order
        method: Similarity method ('jaccard', 'edge_overlap', 'node_overlap')

    Returns:
        Array of stability scores between consecutive graphs

    Example:
        >>> stability = compute_network_stability(graphs, method='jaccard')
        >>> print(f"Mean stability: {np.mean(stability):.3f}")
        >>> print(f"Stable network" if np.mean(stability) > 0.5 else "Dynamic network")
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required. Install with: pip install networkx")

    if len(graphs) < 2:
        return np.array([])

    stability_scores = []

    for i in range(len(graphs) - 1):
        g1 = graphs[i]
        g2 = graphs[i + 1]

        if method == "node_overlap":
            # Jaccard similarity of node sets
            nodes1 = set(g1.nodes)
            nodes2 = set(g2.nodes)

            if len(nodes1) == 0 and len(nodes2) == 0:
                score = 1.0
            elif len(nodes1 | nodes2) == 0:
                score = 0.0
            else:
                score = len(nodes1 & nodes2) / len(nodes1 | nodes2)

        elif method == "edge_overlap":
            # Jaccard similarity of edge sets
            edges1 = set(g1.edges)
            edges2 = set(g2.edges)

            if len(edges1) == 0 and len(edges2) == 0:
                score = 1.0
            elif len(edges1 | edges2) == 0:
                score = 0.0
            else:
                score = len(edges1 & edges2) / len(edges1 | edges2)

        elif method == "jaccard":
            # Combined Jaccard (nodes and edges)
            nodes1 = set(g1.nodes)
            nodes2 = set(g2.nodes)
            edges1 = set(g1.edges)
            edges2 = set(g2.edges)

            if len(nodes1 | nodes2) > 0:
                node_jaccard = len(nodes1 & nodes2) / len(nodes1 | nodes2)
            else:
                node_jaccard = 0.0

            if len(edges1 | edges2) > 0:
                edge_jaccard = len(edges1 & edges2) / len(edges1 | edges2)
            else:
                edge_jaccard = 0.0

            score = (node_jaccard + edge_jaccard) / 2

        else:
            raise ValueError(f"Unknown method: {method}")

        stability_scores.append(score)

    return np.array(stability_scores)


def detect_network_communities(
    graph: AstroGraph,
    algorithm: str = "louvain",
) -> Dict[str, int]:
    """
    Detect communities in astrocyte functional network.

    Args:
        graph: AstroGraph object
        algorithm: Community detection algorithm ('louvain', 'label_propagation', 'greedy')

    Returns:
        Dict mapping node_id to community_id

    Example:
        >>> communities = detect_network_communities(graph, algorithm='louvain')
        >>> print(f"Detected {len(set(communities.values()))} communities")
        >>> for node, comm in communities.items():
        ...     print(f"Node {node} -> Community {comm}")
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required. Install with: pip install networkx")

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)

    for (source, target), weight in zip(graph.edges, graph.edge_weights):
        G.add_edge(source, target, weight=weight)

    if algorithm == "louvain":
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G, weight='weight')
        except ImportError:
            raise ImportError(
                "python-louvain is required. Install with: "
                "pip install python-louvain"
            )

    elif algorithm == "label_propagation":
        communities_gen = nx.community.label_propagation_communities(G)
        communities = {}
        for i, comm_nodes in enumerate(communities_gen):
            for node in comm_nodes:
                communities[node] = i

    elif algorithm == "greedy":
        communities_gen = nx.community.greedy_modularity_communities(G, weight='weight')
        communities = {}
        for i, comm_nodes in enumerate(communities_gen):
            for node in comm_nodes:
                communities[node] = i

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return communities


def compute_network_motifs(
    graph: AstroGraph,
    motif_size: int = 3,
) -> Dict[str, int]:
    """
    Count network motifs (small subgraph patterns).

    Args:
        graph: AstroGraph object
        motif_size: Size of motifs to count (3 or 4)

    Returns:
        Dict mapping motif name to count

    Example:
        >>> motifs = compute_network_motifs(graph, motif_size=3)
        >>> print(f"Triangles: {motifs['triangle']}")
        >>> print(f"Open triplets: {motifs['open_triplet']}")
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required. Install with: pip install networkx")

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)

    for (source, target), _ in zip(graph.edges, graph.edge_weights):
        G.add_edge(source, target)

    motif_counts = {}

    if motif_size == 3:
        # Count triangles
        triangles = nx.triangles(G)
        motif_counts['triangle'] = sum(triangles.values()) // 3  # Each triangle counted 3 times

        # Count open triplets (paths of length 2)
        # This is more complex - approximate using clustering
        n_nodes = G.number_of_nodes()
        if n_nodes > 2:
            degrees = dict(G.degree())
            possible_triplets = sum(d * (d - 1) // 2 for d in degrees.values())
            motif_counts['open_triplet'] = max(0, possible_triplets - motif_counts['triangle'] * 3)
        else:
            motif_counts['open_triplet'] = 0

    elif motif_size == 4:
        # Count 4-node motifs (expensive!)
        # For simplicity, just count 4-cliques and 4-paths
        if G.number_of_nodes() >= 4:
            cliques = list(nx.find_cliques(G))
            motif_counts['4-clique'] = sum(1 for c in cliques if len(c) == 4)

            # 4-paths are harder to count efficiently
            motif_counts['4-path'] = None  # Placeholder
        else:
            motif_counts['4-clique'] = 0
            motif_counts['4-path'] = 0

    else:
        raise ValueError("motif_size must be 3 or 4")

    return motif_counts


def compute_network_modularity(
    graph: AstroGraph,
    communities: Optional[Dict[str, int]] = None,
) -> float:
    """
    Compute network modularity score.

    Higher modularity indicates stronger community structure.

    Args:
        graph: AstroGraph object
        communities: Optional community assignments (if None, will detect automatically)

    Returns:
        Modularity score (range: -0.5 to 1.0)

    Example:
        >>> modularity = compute_network_modularity(graph)
        >>> print(f"Modularity: {modularity:.3f}")
        >>> if modularity > 0.3:
        ...     print("Strong community structure")
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx is required. Install with: pip install networkx")

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(graph.nodes)

    for (source, target), weight in zip(graph.edges, graph.edge_weights):
        G.add_edge(source, target, weight=weight)

    # Detect communities if not provided
    if communities is None:
        communities = detect_network_communities(graph, algorithm='louvain')

    # Convert to list of sets format
    community_sets = {}
    for node, comm_id in communities.items():
        if comm_id not in community_sets:
            community_sets[comm_id] = set()
        community_sets[comm_id].add(node)

    communities_list = list(community_sets.values())

    # Compute modularity
    modularity = nx.community.modularity(G, communities_list, weight='weight')

    return modularity
