import pickle
import blosc # allows to compress the lists

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.parameters import SimulationParams

simparam = SimulationParams()

class CtxToThal(object):
    """
    from analyseur.cbgt.visual.connections.cortex_to_thalamus import CtxToThal
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = CtxToThal.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)
    ConnThalCtx.view(conn_i)
    """
    def __init__(self, rootfolder=None):
        self.conn_i, self.conn_j = self.fetch_connection_lists_active_cortex_inputs(rootfolder=rootfolder,
                                                                                    verbose=True)
        self.populations_names = list(self.conn_i.keys())


    @staticmethod
    def fetch_connection_lists_active_cortex_inputs(rootfolder=None, verbose=False):
        """
        Allows to fetch the synapses connection lists between the cortex and thalamus
        """
        THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs = {}
        THALAMUS_CONNECTION_LISTS_j_active_cortex_inputs = {}

        folder_name = rootfolder + 'THALAMUS/connection_lists/active_cortex_inputs_nbpops=' + str(
            int(simparam.size_info["thalamus"]['TOTAL_NUMBER_OF_POPULATIONS'])) + '/'

        with open(folder_name + "connection_lists_i.dat", "rb") as f:
            compressed_pickle_THALAMUS_CONNECTION_LISTS_i = f.read()
        THALAMUS_CONNECTION_LISTS_i_pickle = pickle.loads(
            blosc.decompress(compressed_pickle_THALAMUS_CONNECTION_LISTS_i))

        for name in THALAMUS_CONNECTION_LISTS_i_pickle:
            if verbose:
                print(name)
            THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs[name] = THALAMUS_CONNECTION_LISTS_i_pickle[name]

        with open(folder_name + "connection_lists_j.dat", "rb") as f:
            compressed_pickle_THALAMUS_CONNECTION_LISTS_j = f.read()
        THALAMUS_CONNECTION_LISTS_j_pickle = pickle.loads(
            blosc.decompress(compressed_pickle_THALAMUS_CONNECTION_LISTS_j))

        for name in THALAMUS_CONNECTION_LISTS_j_pickle:
            if verbose:
                print(name)
            THALAMUS_CONNECTION_LISTS_j_active_cortex_inputs[name] = THALAMUS_CONNECTION_LISTS_j_pickle[name]

        if verbose:
            print("\n=====> Connection lists fetched in folder: " + folder_name + " \n")

        return THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs, THALAMUS_CONNECTION_LISTS_j_active_cortex_inputs


    def view_summary(self):
        """
        Show summary of connections at population level
        """
        plt.figure(figsize=(12, 6))

        populations = list(self.conn_i.keys())
        connection_counts = [len(self.conn_i[pop]) for pop in populations]

        plt.bar(populations, connection_counts, color="skyblue")
        plt.title("Title Cortex→Thalamus Connections per Population")
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
        unique_cortex = [len(set(self.conn_i[pop])) for pop in populations]
        unique_thalamus = [len(set(self.conn_j[pop])) for pop in populations]

        x = np.arange(len(populations))
        width = 0.35
        axes[0, 1].bar(x - width / 2, unique_cortex, width, label="Cortex", alpha=0.7)
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
        divergence = [totals[i] / unique_cortex[i] if unique_cortex[i] > 0 else 0
                      for i in range(len(populations))]
        axes[1, 1].bar(populations, divergence, color="green")
        axes[1, 1].set_title("Average Divergence (conns/cortical neurons)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
