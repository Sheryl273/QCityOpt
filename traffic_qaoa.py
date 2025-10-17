# traffic_qaoa.py
# Quantum Optimization for Traffic Routing in Smart Cities using QAOA
import networkx as nx
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def draw_graph(graph, cut_edges=None, filename="graph.png", title="Traffic Graph"):
    """Draws and saves a graph with optional highlighted cut edges."""
    pos = nx.spring_layout(graph, seed=42)

    # Draw base graph
    nx.draw_networkx_edges(graph, pos, edge_color="gray", width=2)
    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=600)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")

    # Highlight cut edges if available
    if cut_edges:
        nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, edge_color="red", width=3)

    # Edge labels
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10)

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.savefig(filename, dpi=300)
    plt.show()


def solve_traffic_qaoa():
    """Solve a traffic optimization problem using QAOA and visualize results."""

    # 1. Build a sample traffic graph (nodes = intersections, edges = roads)
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        (0, 1, 2),  # road between node 0 and 1 with congestion cost 2
        (1, 2, 3),
        (2, 3, 1),
        (3, 0, 4),
        (0, 2, 2)
    ])

    # Draw original traffic graph
    draw_graph(graph, filename="traffic_original.png", title="Original Traffic Network")

    # 2. Convert to MaxCut problem
    maxcut = Maxcut(graph)
    problem = maxcut.to_quadratic_program()

    # 3. Setup QAOA
    sampler = Sampler(backend=AerSimulator())
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=2)

    # 4. Solve optimization problem
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(problem)

    print("\n=== Quantum Optimization Result (QAOA for Traffic Routing) ===")
    print(result)

    cut = maxcut.get_graph_solution(result)
    cut_edges = maxcut.get_graph_solution(result, True)

    print("Cut (traffic partitioning):", cut)
    print("Edges in the cut:", cut_edges)

    # Draw optimized solution
    draw_graph(graph, cut_edges=cut_edges,
               filename="traffic_qaoa_solution.png",
               title="Optimized Traffic Routing (QAOA)")

# Allow direct execution
if __name__ == "__main__":
    solve_traffic_qaoa()
