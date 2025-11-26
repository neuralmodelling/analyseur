import pickle
import blosc # allows to compress the lists

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.parameters import SimulationParams

simparam = SimulationParams()

class CtxToBG(object):
    """
    from analyseur.cbgt.visual.connections.cortex_to_basalganglia import CtxToBG
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = CtxToBG.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)
    ConnThalCtx.view(conn_i)
    """
    def __init__(self, rootfolder=None):
        self.conn_i, self.conn_j = self.fetch_connection_lists_active_cortex_inputs(rootfolder=rootfolder,
                                                                                    verbose=True)
        self.populations_names = list(self.conn_i.keys())


    @staticmethod
    def fetch_connection_lists_active_cortex_inputs(rootfolder=None, verbose=False):
        """
        Allows to fetch the synapses connection lists between the cortex and basal ganglia
        """
        CONNECTION_LISTS_i_active_cortex_inputs = {}
        CONNECTION_LISTS_j_active_cortex_inputs = {}

        folder_name = rootfolder + 'BG/connection_lists/active_cortex_inputs_scale=' + str(
            int(simparam.size_info["bg"]['scale'])) + '_nbchannels=' + str(
            simparam.size_info["bg"]['TOTAL_NUMBER_OF_CHANNELS']) + '/model_' + str(simparam.modelParamsID) + '/'

        with open(folder_name + "connection_lists_i.dat", "rb") as f:
            compressed_pickle_CONNECTION_LISTS_i = f.read()
        CONNECTION_LISTS_i_pickle = pickle.loads(blosc.decompress(compressed_pickle_CONNECTION_LISTS_i))

        for name in CONNECTION_LISTS_i_pickle:
            if verbose:
                print(name)
            CONNECTION_LISTS_i_active_cortex_inputs[name] = CONNECTION_LISTS_i_pickle[name]

        with open(folder_name + "connection_lists_j.dat", "rb") as f:
            compressed_pickle_CONNECTION_LISTS_j = f.read()
        CONNECTION_LISTS_j_pickle = pickle.loads(blosc.decompress(compressed_pickle_CONNECTION_LISTS_j))

        for name in CONNECTION_LISTS_j_pickle:
            if verbose:
                print(name)
            CONNECTION_LISTS_j_active_cortex_inputs[name] = CONNECTION_LISTS_j_pickle[name]

        if verbose:
            print("\n=====> Connection lists fetched in folder: " + folder_name + " \n")

        return CONNECTION_LISTS_i_active_cortex_inputs, CONNECTION_LISTS_j_active_cortex_inputs


    def view_summary(self):
        """
        Show summary of connections at population level
        """
        plt.figure(figsize=(12, 6))

        populations = list(self.conn_i.keys())
        connection_counts = [len(self.conn_i[pop]) for pop in populations]

        plt.bar(populations, connection_counts, color="skyblue")
        plt.title("Title Cortex→BasalGanglia Connections per Population")
        plt.xlabel("BasalGanglia Populations")
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
        unique_cortex = [len(set(self.conn_i[pop])) for pop in populations]
        unique_basalganglia = [len(set(self.conn_j[pop])) for pop in populations]

        x = np.arange(len(populations))
        width = 0.35
        axes[0, 1].bar(x - width / 2, unique_cortex, width, label="Cortex", alpha=0.7)
        axes[0, 1].bar(x + width / 2, unique_basalganglia, width, label="BasalGanglia", alpha=0.7)
        axes[0, 1].set_title("Unique Neurons")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(populations, rotation=45)
        axes[0, 1].legend()

        # Plot 3: Convergence ratio
        convergence = [totals[i] / unique_basalganglia[i] if unique_basalganglia[i] > 0 else 0
                       for i in range(len(populations))]
        axes[1, 0].bar(populations, convergence, color="orange")
        axes[1, 0].set_title("Average Convergence (conns/basalganglia neurons)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Divergence ratio
        divergence = [totals[i] / unique_cortex[i] if unique_cortex[i] > 0 else 0
                      for i in range(len(populations))]
        axes[1, 1].bar(populations, divergence, color="green")
        axes[1, 1].set_title("Average Divergence (conns/cortical neurons)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def view_actual_network(self, pop_name=None, max_neurons=100):
        """
        Visualize actual connections between Cortex and BasalGanglia neurons
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

        # Add Cortex neurons
        cortex_nodes = set(i_neurons)
        for node in cortex_nodes:
            G.add_node(f"Cx_{node}", type="cortex", color="red")

        # Add BasalGanglia neurons
        basalganglia_nodes = set(j_neurons)
        for node in basalganglia_nodes:
            G.add_node(f"Bg_{node}", type="basalganglia", color="blue")
        # Add Edges
        for i, j in zip(i_neurons, j_neurons):
            G.add_edge(f"Cx_{i}", f"Bg_{j}")

        # Create layout
        plt.figure(figsize=(15, 10))

        # Separate Cortex and BasalGanglia nodes spatially
        pos = {}
        cortex_x = 0
        basalganglia_x = 1

        # Position Cortex nodes
        cortex_list = sorted([n for n in G.nodes() if n.startswith("Cx_")])
        for i, node in enumerate(cortex_list):
            pos[node] = (cortex_x, i / max(1, len(cortex_list)))

        # Position BasalGanglia nodes
        basalganglia_list = sorted([n for n in G.nodes() if n.startswith("Bg_")])
        for i, node in enumerate(basalganglia_list):
            pos[node] = (basalganglia_x, i / max(1, len(basalganglia_list)))

        # Draw with different colors
        node_colors = [G.nodes[node].get("color", "gray") for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6, arrowsize=10)

        plt.title(f"Cortex→BasalGanglia Connections: {pop_name}\n"
                  f"({len(cortex_list)} cortex neurons → {len(basalganglia_list)} basalganglia neurons)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()