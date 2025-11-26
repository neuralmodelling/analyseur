import pickle
import blosc # allows to compress the lists

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.parameters import SimulationParams

simparam = SimulationParams()

class ThalToCtx(object):
    """
    from analyseur.cbgt.visual.connections.thalamus_to_cortex import ThalToCtx
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = ThalToCtx.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)
    ConnThalCtx.view(conn_i)
    """
    def __init__(self, rootfolder=None):
        self.conn_i, self.conn_j = self.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder,
                                                                               verbose=True)
        self.populations_names = list(self.conn_i.keys())


    @staticmethod
    def fetch_connection_lists_Thalamus_inputs(rootfolder=None, verbose=False):
        """
        Allows to fetch the synapses connection lists between the thalamus and cortex
        """
        CORTEX_CONNECTION_LISTS_i_Thalamus_inputs = {}
        CORTEX_CONNECTION_LISTS_j_Thalamus_inputs = {}

        folder_name = rootfolder + 'CORTEX/connection_lists/Thalamus_inputs_nbpops=' + str(
            int(simparam.size_info["cortex"]["TOTAL_NUMBER_OF_POPULATIONS"])) + '/'

        with open(folder_name + "connection_lists_i.dat", "rb") as f:
            compressed_pickle_CORTEX_CONNECTION_LISTS_i = f.read()
        CORTEX_CONNECTION_LISTS_i_pickle = pickle.loads(blosc.decompress(compressed_pickle_CORTEX_CONNECTION_LISTS_i))

        for name in CORTEX_CONNECTION_LISTS_i_pickle:
            if verbose:
                print(name)
            CORTEX_CONNECTION_LISTS_i_Thalamus_inputs[name] = CORTEX_CONNECTION_LISTS_i_pickle[name]

        with open(folder_name + "connection_lists_j.dat", "rb") as f:
            compressed_pickle_CORTEX_CONNECTION_LISTS_j = f.read()
        CORTEX_CONNECTION_LISTS_j_pickle = pickle.loads(blosc.decompress(compressed_pickle_CORTEX_CONNECTION_LISTS_j))

        for name in CORTEX_CONNECTION_LISTS_j_pickle:
            if verbose:
                print(name)
            CORTEX_CONNECTION_LISTS_j_Thalamus_inputs[name] = CORTEX_CONNECTION_LISTS_j_pickle[name]

        if verbose:
            print("\n=====> Connection lists fetched in folder: " + folder_name + " \n")

        return CORTEX_CONNECTION_LISTS_i_Thalamus_inputs, CORTEX_CONNECTION_LISTS_j_Thalamus_inputs

    @staticmethod
    def view(CORTEX_CONNECTION_LISTS_i_Thalamus_inputs):
        """
        Create a network graph showing thalamus-cortex connections.
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for thalamus and cortical populations
        thalamus_node = "Thalamus"
        G.add_node(thalamus_node, type="input", color="red")

        # Add cortical population nodes
        for pop_name in CORTEX_CONNECTION_LISTS_i_Thalamus_inputs.keys():
            G.add_node(pop_name, type="cortex", color="blue")

        # Add edges based on connection data
        for pop_name in CORTEX_CONNECTION_LISTS_i_Thalamus_inputs.keys():
            num_connections = len(CORTEX_CONNECTION_LISTS_i_Thalamus_inputs[pop_name])
            G.add_edge(thalamus_node, pop_name, weight=num_connections)

        # Create layout and plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]["weight"]/1000 for u,v in edges]  # scale for visibility
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=weights)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Thalamus to Cortex Connections")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Print connection statistics
        print("Connection Statistics:")
        for pop_name in CORTEX_CONNECTION_LISTS_i_Thalamus_inputs.keys():
            num_conn = len(CORTEX_CONNECTION_LISTS_i_Thalamus_inputs[pop_name])
            print(f"  {pop_name}: {num_conn} connections")

    def graph_from_multiple_views(self):
        """
        Create multiple views of the network.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        G = nx.DiGraph()

        # Build the graph
        thalamus_node = "Thalamus"
        G.add_node(thalamus_node, type="input", color="red")

        for pop_name in self.conn_i.keys():
            G.add_node(pop_name, type="cortex", color="blue")
            num_connections = len(self.conn_i[pop_name])
            G.add_edge(thalamus_node, pop_name, weight=num_connections)

        # Different layouts for different views
        layouts = {
            "Spring Layout": nx.spring_layout(G),
            "Circular Layout": nx.circular_layout(G),
            "Shell Layout": nx.shell_layout(G),
            "Random Layout": nx.random_layout(G),
        }

        for (layout_name, pos), ax in zip(layouts.items(), axes):
            node_colors = [G.nodes[node]["color"] for node in G.nodes()]

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=600, ax=ax)
            # Draw edges
            edges = G.edges()
            weights = [G[u][v]["weight"] / 1000 for u, v in edges]
            nx.draw_networkx_edges(G, pos, edge_color="gray",
                                   width=weights, ax=ax)
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

            ax.set_title(f"{layout_name}\nThalamus to Cortex Connections")
            ax.axis("off")

        # Hide unused subplots if any
        for i in range(len(layouts), len(axes)):
            axes[i].axis["off"]

        plt.tight_layout()
        plt.show()

    def view_actual_network(self, pop_name=None, max_neurons=100):
        """
        Visualize actual connections between Thalamus and Cortical neurons
        for a specific population (sampled if too large).
        """
        if pop_name not in self.conn_i:
            print(f"Population {pop_name} not found!")
            return

        i_neurons = self.conn_i[pop_name]
        j_neurons = self.conn_j[pop_name]

        # Sample if too many connections
        if len(i_neurons) > max_neurons:
            indices = np.random.choice(len(i_neurons), max_neurons, replace=False)
            i_neurons = [i_neurons[i] for i in indices]
            j_neurons = [j_neurons[i] for i in indices]
            print(f"Sampled {max_neurons} connections from {len(i_neurons)} total")

        G = nx.DiGraph()

        # Add Thalamus neurons
        thalamus_nodes = set(i_neurons)
        for node in thalamus_nodes:
            G.add_node(f"Th_{node}", type="thalamus", color="red")

        # Add Cortical neurons
        cortex_nodes = set(j_neurons)
        for node in cortex_nodes:
            G.add_node(f"Cx_{node}", type="cortex", color="blue")

        # Add Edges
        for i, j in zip(i_neurons, j_neurons):
            G.add_edge(f"Th_{i}", f"Cx_{j}")

        all_nodes = [node for node in G.nodes]

        # Create layout
        plt.figure(figsize=(15, 10))

        # Separate Thalamus and Cortex nodes spatially
        pos = {}
        thalamus_x = 0
        cortex_x = 1

        # Position Thalamus nodes
        thalamus_list = sorted([n for n in G.nodes() if n.startswith("Th_")])
        for i, node in enumerate(thalamus_list):
            pos[node] = (thalamus_x, i / max(1, len(thalamus_list)))

        # Position Cortex nodes
        cortex_list = sorted([n for n in G.nodes() if n.startswith("Cx_")])
        for i, node in enumerate(cortex_list):
            pos[node] = (cortex_x, i / max(1, len(cortex_list)))

        # Draw with different colors
        node_colors = [G.nodes[node].get("color", "gray") for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6, arrowsize=10)

        plt.title(f"Thalamus→Cortex Connections: {pop_name}\n"
                  f"({len(thalamus_list)} thalamus neurons → {len(cortex_list)} cortical neurons)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def view_actual_network_simple(self, pop_name=None, max_neurons=100):
        """
        Visualize actual connections between Thalamus and Cortical neurons
        for a specific population (sampled if too large).
        """
        if pop_name not in self.conn_i:
            print(f"Population {pop_name} not found!")
            return

        i_neurons = self.conn_i[pop_name]
        j_neurons = self.conn_j[pop_name]

        # Sample if too many connections
        if len(i_neurons) > max_neurons:
            indices = np.random.choice(len(i_neurons), max_neurons, replace=False)
            i_neurons = [i_neurons[i] for i in indices]
            j_neurons = [j_neurons[i] for i in indices]
            print(f"Sampled {max_neurons} connections from {len(i_neurons)} total")

        #G = nx.DiGraph()
        # Alternative to NetworkX
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create artificial coordinates
        thalamus_nodes = list(set(i_neurons))
        cortex_nodes = list(set(j_neurons))

        # Position nodes manually
        thalamus_x = [0] * len(thalamus_nodes)
        thalamus_y = np.linspace(0, 1, len(thalamus_nodes))

        cortex_x = [1] * len(cortex_nodes)
        cortex_y = np.linspace(0, 1, len(cortex_nodes))

        # Plot Nodes
        ax.scatter(thalamus_x, thalamus_y, c="red", s=50, label="Thalamus")
        ax.scatter(cortex_x, cortex_y, c="blue", s=50, label="Cortex")

        # Plot Edges
        for i, j in zip(i_neurons, j_neurons):
            thalamus_idx = thalamus_nodes.index(i)
            cortex_idx = cortex_nodes.index(j)
            ax.plot([0, 1], [thalamus_y[thalamus_idx], cortex_y[cortex_idx]],
                    "gray", alpha=0.3, linewidth=0.5)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Thalamus→Cortex Connections: {pop_name}\n"
                     f"({len(thalamus_nodes)} thalamus → {len(cortex_nodes)} cortex neurons")
        ax.legend()
        ax.axis("off")

        plt.tight_layout()
        plt.show()

    def view_graph_in_ax(self, ax):
        pass

    @classmethod
    def view_graph(cls, rootfolder=None):
        conn_i, conn_j = cls.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)

    @classmethod
    def view_summary(cls, rootfolder=None):
        """
        Show summary of connections at population level
        """
        conn_i, conn_j = cls.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)

        plt.figure(figsize=(12, 6))

        populations = list(conn_i.keys())
        connection_counts = [len(conn_i[pop]) for pop in populations]

        plt.bar(populations, connection_counts, color="skyblue")
        plt.title("Title Thalamus→Cortex Connections per Population")
        plt.xlabel("Cortical Populations")
        plt.ylabel("Number of Connections")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(connection_counts):
            plt.text(i, v, str(v), ha="center", va="bottom")

        plt.tight_layout()
        plt.show()

        print("Connection Statistics:")
        for pop in populations:
            i_conn = len(conn_i[pop])
            j_conn = len(conn_j[pop])
            print(f" {pop}: {i_conn} connections (should equal {j_conn})")

    @classmethod
    def compare_populations(cls, rootfolder=None):
        """
        Compare connection patterns across all populations
        """
        conn_i, conn_j = cls.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        populations = list(conn_i.keys())

        # Plot 1: Total connections
        totals = [len(conn_i[pop]) for pop in populations]
        axes[0,0].bar(populations, totals, color="lightblue")
        axes[0,0].set_title("Total Connections")
        axes[0,0].tick_params(axis="x", rotation=45)

        # Plot 2: Unique neurons
        unique_thalamus = [len(set(conn_i[pop])) for pop in populations]
        unique_cortex = [len(set(conn_j[pop])) for pop in populations]

        x = np.arange(len(populations))
        width = 0.35
        axes[0, 1].bar(x - width / 2, unique_thalamus, width, label="Thalamus", alpha=0.7)
        axes[0, 1].bar(x + width / 2, unique_cortex, width, label="Cortex", alpha=0.7)
        axes[0, 1].set_title("Unique Neurons")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(populations, rotation=45)
        axes[0, 1].legend()

        # Plot 3: Convergence ratio
        convergence = [totals[i] / unique_cortex[i] if unique_cortex[i] > 0 else 0
                       for i in range(len(populations))]
        axes[1, 0].bar(populations, convergence, color="orange")
        axes[1, 0].set_title("Average Convergence (conns/cortical neurons)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Divergence ratio
        divergence = [totals[i] / unique_thalamus[i] if unique_thalamus[i] > 0 else 0
                      for i in range(len(populations))]
        axes[1, 1].bar(populations, divergence, color="green")
        axes[1, 1].set_title("Average Divergence (conns/thalamus neurons)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()