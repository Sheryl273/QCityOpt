import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
# Removed the problematic line: CplexMaxcutOptimizer. 
# We use nx.approximation.one_exchange as the classical benchmark.

# --- Your QAOA Function Placeholder ---
def run_qaoa(G, p=1, shots=1024):
    """
    Simulates QAOA for MaxCut on the given weighted graph G.
    
    NOTE: Replace this body with your actual Qiskit QAOA implementation.
    The current implementation simulates a result close to the classical heuristic.
    """
    # Use the NetworkX approximation result as a basis for the QAOA placeholder
    classical_solver = nx.approximation.one_exchange
    classical_cut_value, _ = classical_solver(G, weight='weight')
    
    # Simulate a QAOA result (e.g., 90% to 105% of the classical heuristic)
    qaoa_result = classical_cut_value * random.uniform(0.90, 1.05) 
    
    # Cap the QAOA result to prevent unrealistically exceeding the classical heuristic
    return min(qaoa_result, classical_cut_value * 1.05)


# --- Graph Generation Functions ---

def create_weighted_grid_graph(n_rows, n_cols, seed=42):
    G = nx.grid_2d_graph(n_rows, n_cols)
    random.seed(seed)
    weights = {(u, v): random.randint(1, 10) for u, v in G.edges()}
    nx.set_edge_attributes(G, weights, 'weight')
    # Convert to integer nodes for Qiskit compatibility
    return nx.convert_node_labels_to_integers(G)

def create_weighted_ba_graph(n, m, seed=42):
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    random.seed(seed)
    weights = {(u, v): random.randint(5, 15) for u, v in G.edges()}
    nx.set_edge_attributes(G, weights, 'weight')
    return G

def create_weighted_random_graph(n, p, seed=42):
    random.seed(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Ensure the graph is connected (re-seed until connected)
    attempt = 0
    while not nx.is_connected(G) and G.number_of_nodes() > 1 and attempt < 100:
        G = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 10000))
        attempt += 1
        
    weights = {(u, v): random.randint(1, 10) for u, v in G.edges()}
    nx.set_edge_attributes(G, weights, 'weight')
    return G

# --- Main Comparison Workflow ---

def run_comparison():
    
    n_nodes = 16 
    
    comparison_graphs = {
        f"Grid (4x4)": create_weighted_grid_graph(4, 4, seed=10),
        f"Scale-Free (BA, m=2)": create_weighted_ba_graph(n=n_nodes, m=2, seed=20),
        f"Random G(n=16, p=0.4)": create_weighted_random_graph(n=n_nodes, p=0.4, seed=30),
        f"Smaller Random G(n=10, p=0.6)": create_weighted_random_graph(n=10, p=0.6, seed=40),
    }

    results_list = []

    for name, G in comparison_graphs.items():
        print(f"Analyzing: {name} (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})...")

        # A. Classical Solver (NetworkX Greedy Heuristic) - The Benchmark
        classical_cut_value, _ = nx.approximation.one_exchange(G, weight='weight')

        # B. Quantum Solver (QAOA Call)
        qaoa_cut_value = run_qaoa(G)

        # C. Calculate Metrics
        approx_ratio = qaoa_cut_value / classical_cut_value
        
        results_list.append({
            "Graph Type": name,
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Classical Cut Value": classical_cut_value,
            "QAOA Cut Value": qaoa_cut_value,
            "Approximation Ratio (QAOA/Classical)": approx_ratio
        })

    # 2. Process Results
    df_results = pd.DataFrame(results_list)
    print("\n" + "="*50)
    print("FINAL COMPARISON RESULTS SUMMARY")
    print("="*50)
    print(df_results.to_string())

    # 3. Visualization
    plot_results(df_results, comparison_graphs)

# --- Visualization Function (Containing the fix) ---

def plot_results(df_results, graphs):
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # A. Plot Approximation Ratio Bar Chart
    axes[0, 0].bar(df_results["Graph Type"], df_results["Approximation Ratio (QAOA/Classical)"], 
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[0, 0].set_title('Approximation Ratio (QAOA / Classical Benchmark) by Graph Type', fontsize=14)
    axes[0, 0].set_ylabel('Approximation Ratio (Cut_QAOA / Cut_Classical)', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].axhline(1.0, color='gray', linestyle='--', linewidth=0.8, label='Classical Heuristic Parity')
    axes[0, 0].legend()
    
    # B. Plot Max Cut Values Comparison Bar Chart
    df_results[['Classical Cut Value', 'QAOA Cut Value']].plot(
        kind='bar', ax=axes[0, 1], rot=45, colormap='viridis'
    )
    axes[0, 1].set_title('Weighted MaxCut Value Comparison', fontsize=14)
    axes[0, 1].set_ylabel('Cut Value (Sum of Edge Weights)', fontsize=12)
    axes[0, 1].set_xlabel('Graph Type', fontsize=12)
    axes[0, 1].legend(loc='upper right')

    # C. Visualize Grid Graph (Structured Traffic)
    G_grid = graphs["Grid (4x4)"]
    
    # FIX: Use spring_layout since node labels were converted to single integers (0, 1, 2,...)
    pos_grid = nx.spring_layout(G_grid, seed=0) 
    
    edge_weights_grid = nx.get_edge_attributes(G_grid, 'weight')
    edge_width_grid = [w * 0.4 for w in edge_weights_grid.values()]

    nx.draw_networkx(G_grid, pos_grid, ax=axes[1, 0], with_labels=False, node_color='skyblue', node_size=300, 
                     width=edge_width_grid, edge_color='gray')
    nx.draw_networkx_edge_labels(G_grid, pos_grid, edge_labels=edge_weights_grid, ax=axes[1, 0], font_color='red', font_size=8)
    axes[1, 0].set_title('Visualization: Grid Graph (Weighted Edges)', fontsize=14)
    
    # D. Visualize Barabasi-Albert Graph (Hub-and-Spoke Traffic)
    G_ba = graphs["Scale-Free (BA, m=2)"]
    pos_ba = nx.spring_layout(G_ba, seed=1)
    edge_weights_ba = nx.get_edge_attributes(G_ba, 'weight')
    edge_width_ba = [w * 0.25 for w in edge_weights_ba.values()]
    
    # Highlight high-degree nodes (hubs)
    node_degrees = dict(G_ba.degree())
    node_colors = ['red' if node_degrees[n] > 4 else 'lightcoral' for n in G_ba.nodes()]

    nx.draw_networkx(G_ba, pos_ba, ax=axes[1, 1], with_labels=True, node_color=node_colors, node_size=500, 
                     width=edge_width_ba, edge_color='gray', font_size=8)
    axes[1, 1].set_title('Visualization: Scale-Free Graph (Hubs are Red)', fontsize=14)

    plt.suptitle("QAOA MaxCut Performance Across Various Weighted Graph Topologies", fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

if __name__ == "__main__":
    run_comparison()