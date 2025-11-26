import pickle
import blosc # allows to compress the lists

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.parameters import SimulationParams

simparam = SimulationParams()

class BGToThal(object):
    """
    from analyseur.cbgt.visual.connections.basalganglia_to_thalamus import BGToThal
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = BGToThal.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)
    ConnThalCtx.view(conn_i)
    """
    def __init__(self, rootfolder=None):
        self.conn_i, self.conn_j = self.fetch_connection_lists_BG_inputs(rootfolder=rootfolder,
                                                                         verbose=True)
        self.populations_names = list(self.conn_i.keys())


    @staticmethod
    def fetch_connection_lists_BG_inputs(rootfolder=None, verbose=False):
        """
        Allows to fetch the synapses connection lists between the basal ganglia and thalamus
        """
        THALAMUS_CONNECTION_LISTS_i_BG_inputs = {}
        THALAMUS_CONNECTION_LISTS_j_BG_inputs = {}

        folder_name = rootfolder + 'THALAMUS/connection_lists/BG_inputs_nbpops=' + str(
            int(simparam.size_info["thalamus"]['TOTAL_NUMBER_OF_POPULATIONS'])) + '/'

        with open(folder_name + "connection_lists_i.dat", "rb") as f:
            compressed_pickle_THALAMUS_CONNECTION_LISTS_i = f.read()
        THALAMUS_CONNECTION_LISTS_i_pickle = pickle.loads(
            blosc.decompress(compressed_pickle_THALAMUS_CONNECTION_LISTS_i))

        for name in THALAMUS_CONNECTION_LISTS_i_pickle:
            if verbose:
                print(name)
            THALAMUS_CONNECTION_LISTS_i_BG_inputs[name] = THALAMUS_CONNECTION_LISTS_i_pickle[name]

        with open(folder_name + "connection_lists_j.dat", "rb") as f:
            compressed_pickle_THALAMUS_CONNECTION_LISTS_j = f.read()
        THALAMUS_CONNECTION_LISTS_j_pickle = pickle.loads(
            blosc.decompress(compressed_pickle_THALAMUS_CONNECTION_LISTS_j))

        for name in THALAMUS_CONNECTION_LISTS_j_pickle:
            if verbose:
                print(name)
            THALAMUS_CONNECTION_LISTS_j_BG_inputs[name] = THALAMUS_CONNECTION_LISTS_j_pickle[name]

        if verbose:
            print("\n=====> Connection lists fetched in folder: " + folder_name + " \n")

        return THALAMUS_CONNECTION_LISTS_i_BG_inputs, THALAMUS_CONNECTION_LISTS_j_BG_inputs


    def view_summary(self):
        """
        Show summary of connections at population level
        """
        plt.figure(figsize=(12, 6))

        populations = list(self.conn_i.keys())
        connection_counts = [len(self.conn_i[pop]) for pop in populations]

        plt.bar(populations, connection_counts, color="skyblue")
        plt.title("Title BasalGanglia→Thalamus Connections per Population")
        plt.xlabel("Thalamus Populations")
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
            i_conn = len(self.conn_i[pop])
            j_conn = len(self.conn_j[pop])
            print(f" {pop}: {i_conn} connections (should equal {j_conn})")


    def compare_populations(self):
        """
        Compare connection patterns across all populations
        """

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        populations = list(self.conn_i.keys())

        # Plot 1: Total connections
        totals = [len(self.conn_i[pop]) for pop in populations]
        axes[0,0].bar(populations, totals, color="lightblue")
        axes[0,0].set_title("Total Connections")
        axes[0,0].tick_params(axis="x", rotation=45)

        # Plot 2: Unique neurons
        unique_basalganglia = [len(set(self.conn_i[pop])) for pop in populations]
        unique_thalamus = [len(set(self.conn_j[pop])) for pop in populations]

        x = np.arange(len(populations))
        width = 0.35
        axes[0, 1].bar(x - width / 2, unique_basalganglia, width, label="BasalGanglia", alpha=0.7)
        axes[0, 1].bar(x + width / 2, unique_thalamus, width, label="Thalamus", alpha=0.7)
        axes[0, 1].set_title("Unique Neurons")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(populations, rotation=45)
        axes[0, 1].legend()

        # Plot 3: Convergence ratio
        convergence = [totals[i] / unique_thalamus[i] if unique_thalamus[i] > 0 else 0
                       for i in range(len(populations))]
        axes[1, 0].bar(populations, convergence, color="orange")
        axes[1, 0].set_title("Average Convergence (conns/thalamus neurons)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Divergence ratio
        divergence = [totals[i] / unique_basalganglia[i] if unique_basalganglia[i] > 0 else 0
                      for i in range(len(populations))]
        axes[1, 1].bar(populations, divergence, color="green")
        axes[1, 1].set_title("Average Divergence (conns/basalganglia neurons)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()


    def view_actual_network(self, pop_name=None, max_neurons=100):
        """
        Visualize actual connections between BasalGanglia and Thalamus neurons
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

        # Add BasalGanglia neurons
        basalganglia_nodes = set(i_neurons)
        for node in basalganglia_nodes:
            G.add_node(f"Bg_{node}", type="basalganglia", color="red")

        # Add Thalamus neurons
        thalamus_nodes = set(j_neurons)
        for node in thalamus_nodes:
            G.add_node(f"Th_{node}", type="thalamus", color="blue")

        # Add Edges
        for i, j in zip(i_neurons, j_neurons):
            G.add_edge(f"Bg_{i}", f"Th_{j}")

        # Create layout
        plt.figure(figsize=(15, 10))

        # Separate BasalGanglia and Thalamus nodes spatially
        pos = {}
        basalganglia_x = 0
        thalamus_x = 1

        # Position BasalGanglia nodes
        basalganglia_list = sorted([n for n in G.nodes() if n.startswith("Bg_")])
        for i, node in enumerate(basalganglia_list):
            pos[node] = (basalganglia_x, i / max(1, len(basalganglia_list)))

        # Position Thalamus nodes
        thalamus_list = sorted([n for n in G.nodes() if n.startswith("Th_")])
        for i, node in enumerate(thalamus_list):
            pos[node] = (thalamus_x, i / max(1, len(thalamus_list)))

        # Draw with different colors
        node_colors = [G.nodes[node].get("color", "gray") for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6, arrowsize=10)

        plt.title(f"BasalGanglia→Thalamus Connections: {pop_name}\n"
                  f"({len(basalganglia_list)} basalganglia neurons → {len(thalamus_list)} thalamus neurons)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


    def view_graph(self):
        """
        Create a network graph showing BasalGanglia-Thalamus connections.
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for basalganglia and thalamus populations
        basalganglia_node = "BasalGanglia"
        G.add_node(basalganglia_node, type="input", color="red")

        # Add thalamus population nodes
        for pop_name in self.conn_i.keys():
            G.add_node(pop_name, type="thalamus", color="blue")

        # Add edges based on connection data
        for pop_name in self.conn_i.keys():
            num_connections = len(self.conn_i[pop_name])
            G.add_edge(basalganglia_node, pop_name, weight=num_connections)

        # Create layout and plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]["weight"]/1000 for u,v in edges]  # scale for visibility
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=weights)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("BasalGanglia to Thalamus Connections")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Print connection statistics
        print("Connection Statistics:")
        for pop_name in self.conn_i.keys():
            num_conn = len(self.conn_i[pop_name])
            print(f"  {pop_name}: {num_conn} connections")
