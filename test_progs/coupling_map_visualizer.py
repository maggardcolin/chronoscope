import matplotlib.pyplot as plt
import networkx as nx
from connectivity_maps import edges_mesh, edges_trapped_ion, edges_heavy_hex

# Dictionary of connectivity maps
connectivity_maps = {
    "Mesh": edges_mesh,
    "Trapped Ion": edges_trapped_ion,
    "Heavy Hex": edges_heavy_hex
}

def get_node_positions(edges, title):
    """Return fixed node positions based on topology type."""
    num_qubits = max(max(edge) for edge in edges) + 1

    if title == "Mesh":
        # 10x10 grid
        positions = {i: (i % 10, -(i // 10)) for i in range(num_qubits)}

    elif title == "Trapped Ion":
        # Line of groups (5 qubits each group), spaced out
        positions = {}
        for group in range(20):
            for idx in range(5):
                qubit = group * 5 + idx
                # Each group on a new vertical line
                positions[qubit] = (group * 2, -idx)

    elif title == "Heavy Hex":
        # Staggered grid for heavy hex
        positions = {}
        for i in range(num_qubits):
            row = i // 10
            col = i % 10
            # Odd rows are shifted right a little
            shift = 0.5 if row % 2 == 1 else 0
            positions[i] = (col + shift, -row)

    else:
        # Default spring layout
        positions = None

    return positions

def visualize_connectivity_map(edges, title):
    """Visualize the connectivity map using NetworkX and Matplotlib with fixed layout."""
    graph = nx.Graph()
    graph.add_edges_from(edges)

    positions = get_node_positions(edges, title)

    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos=positions, with_labels=True, node_color="lightblue",
            node_size=500, font_size=10, font_weight="bold")
    plt.title(title)
    plt.axis('equal')
    plt.show()

def main():
    print("Available Connectivity Maps:")
    for i, name in enumerate(connectivity_maps.keys(), start=1):
        print(f"{i}. {name}")

    choice = int(input("Select a connectivity map to visualize (enter the number): ")) - 1
    if choice < 0 or choice >= len(connectivity_maps):
        print("Invalid choice. Exiting.")
        return

    selected_map_name = list(connectivity_maps.keys())[choice]
    selected_map_edges = connectivity_maps[selected_map_name]

    print(f"Visualizing {selected_map_name} connectivity map...")
    visualize_connectivity_map(selected_map_edges, selected_map_name)

if __name__ == "__main__":
    main()
