#!/usr/bin/env python3
"""
qubo_mapping.py

Map a small vehicle-routing / simplified TSP-like instance to a QUBO.

Notes:
- Binary variable x_e = 1 if edge e is chosen.
- Objective: minimize total weighted edge cost (we place edge weights on diagonal).
- Degree constraints enforced with quadratic penalties:
    * Non-depot nodes: degree == 2 (for single-route TSP-like tours)
    * Depot node: degree == 2 * vehicle_count
- Subtour elimination / connectivity NOT enforced here (small instances only).
"""

from typing import Dict, Tuple, List
import itertools
import json
import networkx as nx
import numpy as np

Edge = Tuple[int, int]
Qubo = Dict[Tuple[int, int], float]


def edge_index_map(G: nx.Graph) -> Tuple[Dict[Edge, int], List[Edge]]:
    """Return mapping edge -> index and list of edges (consistent ordering)."""
    edges = sorted([tuple(sorted(e)) for e in G.edges()])
    idx = {e: k for k, e in enumerate(edges)}
    return idx, edges


def build_qubo_for_routing(
    G: nx.Graph,
    depot: int = 0,
    vehicle_count: int = 1,
    penalty_degree: float = 10.0,
) -> Tuple[Qubo, List[Edge]]:
    """
    Build a QUBO for a simplified routing problem (single tour per vehicle approximation).

    Returns:
      Q: dict mapping (i,j) -> coefficient (upper-triangular convention)
      edges: list of edges corresponding to variable indices
    """
    idx, edges = edge_index_map(G)
    m = len(edges)
    Q: Qubo = {}

    def add_qubo(i: int, j: int, coeff: float) -> None:
        # store using upper-triangular ordering (i <= j)
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + float(coeff)

    # Objective: minimize total weighted edge cost
    # We place weights on diagonal so minimization solver sees cost ~ sum w*x
    for k, e in enumerate(edges):
        u, v = e
        w = float(G[u][v].get("weight", 1.0))
        add_qubo(k, k, w)

    # Build node -> incident edge indices map
    node_edge_map = {n: [] for n in G.nodes()}
    for k, e in enumerate(edges):
        u, v = e
        node_edge_map[u].append(k)
        node_edge_map[v].append(k)

    # Degree constraints for non-depot nodes: (sum_i x_i - 2)^2
    for n in G.nodes():
        if n == depot:
            continue
        inds = node_edge_map.get(n, [])
        if not inds:
            continue
        # Expand (sum x_i - 2)^2 = sum x_i^2 + 2*sum_{i<j} x_i x_j -4*sum x_i + 4
        for i in inds:
            add_qubo(i, i, penalty_degree * 1.0)  # x_i^2 coefficient
        for (i, j) in itertools.combinations(inds, 2):
            add_qubo(i, j, penalty_degree * 2.0)  # cross terms
        for i in inds:
            add_qubo(i, i, -penalty_degree * 4.0)  # linear correction term

    # Depot degree constraint: (sum_i x_i - K)^2 where K = 2 * vehicle_count
    depot_inds = node_edge_map.get(depot, [])
    K = 2 * vehicle_count
    if depot_inds:
        for i in depot_inds:
            add_qubo(i, i, penalty_degree * 1.0)
        for (i, j) in itertools.combinations(depot_inds, 2):
            add_qubo(i, j, penalty_degree * 2.0)
        for i in depot_inds:
            add_qubo(i, i, -penalty_degree * 2.0 * K)

    return Q, edges


def qubo_to_matrix(Q: Qubo) -> np.ndarray:
    """Convert upper-triangular Q dict to dense symmetric matrix."""
    if not Q:
        return np.zeros((0, 0))
    max_index = max(max(i, j) for i, j in Q.keys()) + 1
    M = np.zeros((max_index, max_index))
    for (i, j), v in Q.items():
        if i == j:
            M[i, i] += v
        else:
            # Q stored upper-triangular; distribute to symmetric matrix
            M[i, j] += v
            M[j, i] += v
    return M


def sample_grid_graph(n: int = 5) -> nx.Graph:
    """
    Create a small grid-like weighted graph for testing.
    n = number of nodes (will be arranged in an approximate square grid).
    """
    side = int(np.ceil(np.sqrt(n)))
    G = nx.grid_2d_graph(side, side)
    G = nx.convert_node_labels_to_integers(G)

    # Deterministic layout for distance-based weights
    pos = nx.spring_layout(G, seed=42)
    for u, v in G.edges():
        w = float(np.linalg.norm(np.array(pos[u]) - np.array(pos[v])))
        G[u][v]["weight"] = w
    return G


def save_edges_json(edges: List[Edge], filename: str = "sample_edges.json") -> None:
    # Convert edges (tuples) to lists for JSON serializability
    serializable = {"edges": [list(e) for e in edges]}
    with open(filename, "w") as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    # Example usage
    G = sample_grid_graph(9)  # creates a 3x3 grid
    Q, edges = build_qubo_for_routing(G, depot=0, vehicle_count=1, penalty_degree=30.0)
    M = qubo_to_matrix(Q)
    print("Edges count:", len(edges))
    print("Dense QUBO matrix shape:", M.shape)
    # Show a small portion of the matrix for sanity
    print("QUBO matrix (first 6x6 block):")
    print(M[:6, :6])
    save_edges_json(edges, "sample_edges.json")
    print("Saved sample_edges.json")
