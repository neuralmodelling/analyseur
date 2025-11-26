import pickle
import blosc # allows to compress the lists

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.parameters import SimulationParams

simparam = SimulationParams()

class ConnThalCtx(object):
    """
    from analyseur.cbgt.visual.conn import ConnThalCtx
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = ConnThalCtx.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)
    ConnThalCtx.view(conn_i)
    """

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


    @classmethod
    def view_graph_in_ax(cls, ax, rootfolder=None):
        conn_i, conn_j = cls.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)

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
        axes[0, 1].bar(x - width/2, unique_thalamus, width, label="Thalamus", alpha=0.7)
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

class ConnBGThal(object):
    """
    from analyseur.cbgt.visual.conn import ConnBGThal
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = ConnBGThal.fetch_connection_lists_BG_inputs(rootfolder=rootfolder, verbose=True)
    ConnBGThal.view(conn_i)
    """

    @staticmethod
    def fetch_connection_lists_BG_inputs(rootfolder=None, verbose=False):
        """
        fetch the list stacked on the computer
        """
        THALAMUS_CONNECTION_LISTS_i_BG_inputs = {}
        THALAMUS_CONNECTION_LISTS_j_BG_inputs = {}

        folder_name = rootfolder + 'THALAMUS/connection_lists/BG_inputs_nbpops=' + str(
            int(simparam.size_info["thalamus"]["TOTAL_NUMBER_OF_POPULATIONS"])) + '/'
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

    @staticmethod
    def view(THALAMUS_CONNECTION_LISTS_i_BG_inputs):
        """
        Create a network graph showing basalganglia-thalamus connections.
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for basalganglia and thalamus populations
        bg_node = "BasalGanglia"
        G.add_node(bg_node, type="input", color="red")

        # Add thalamus population nodes
        for pop_name in THALAMUS_CONNECTION_LISTS_i_BG_inputs.keys():
            G.add_node(pop_name, type="thalamus", color="blue")

        # Add edges based on connection data
        for pop_name in THALAMUS_CONNECTION_LISTS_i_BG_inputs.keys():
            num_connections = len(THALAMUS_CONNECTION_LISTS_i_BG_inputs[pop_name])
            G.add_edge(bg_node, pop_name, weight=num_connections)

        # Create layout and plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]["weight"] / 1000 for u, v in edges]  # scale for visibility
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=weights)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("BasalGanglia to Thalamus Connections")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Print connection statistics
        print("Connection Statistics:")
        for pop_name in THALAMUS_CONNECTION_LISTS_i_BG_inputs.keys():
            num_conn = len(THALAMUS_CONNECTION_LISTS_i_BG_inputs[pop_name])
            print(f"  {pop_name}: {num_conn} connections")


class ConnCtxThal(object):
    """
    from analyseur.cbgt.visual.conn import ConnCtxThal
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = ConnCtxThal.fetch_connection_lists_active_cortex_inputs(rootfolder=rootfolder, verbose=True)
    ConnCtxThal.view(conn_i)
    """

    @staticmethod
    def fetch_connection_lists_active_cortex_inputs(rootfolder=None, verbose=False):
        """
        fetch the list stacked on the computer
        """
        THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs = {}
        THALAMUS_CONNECTION_LISTS_j_active_cortex_inputs = {}

        folder_name = rootfolder + 'THALAMUS/connection_lists/active_cortex_inputs_nbpops=' + str(
            int(simparam.size_info["thalamus"]["TOTAL_NUMBER_OF_POPULATIONS"])) + '/'
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

    @staticmethod
    def view(THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs):
        """
        Create a network graph showing cortex-thalamus connections.
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for cortex and thalamus populations
        ctx_node = "Cortex"
        G.add_node(ctx_node, type="input", color="red")

        # Add thalamus population nodes
        for pop_name in THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs.keys():
            G.add_node(pop_name, type="thalamus", color="blue")

        # Add edges based on connection data
        for pop_name in THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs.keys():
            num_connections = len(THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs[pop_name])
            G.add_edge(ctx_node, pop_name, weight=num_connections)

        # Create layout and plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]["weight"]/1000 for u,v in edges]  # scale for visibility
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=weights)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Cortex to Thalamus Connections")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Print connection statistics
        print("Connection Statistics:")
        for pop_name in THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs.keys():
            num_conn = len(THALAMUS_CONNECTION_LISTS_i_active_cortex_inputs[pop_name])
            print(f"  {pop_name}: {num_conn} connections")

class ConnCtxBG(object):
    """
    from analyseur.cbgt.visual.conn import ConnCtxBG
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = ConnCtxBG.fetch_connection_lists_active_cortex_inputs(rootfolder=rootfolder, verbose=True)
    ConnCtxBG.view(conn_i)
    """

    @staticmethod
    def fetch_connection_lists_active_cortex_inputs(rootfolder=None, verbose=False):
        """
        Allows to fetch the synapses connection lists between the active cortex and the basal ganglia
        """
        CONNECTION_LISTS_i_active_cortex_inputs = {}
        CONNECTION_LISTS_j_active_cortex_inputs = {}

        folder_name = rootfolder + 'BASAL_GANGLIA/connection_lists/active_cortex_inputs_scale=' + str(
            int(simparam.size_info["bg"]["scale"])) + '_nbchannels=' + str(
            int(simparam.size_info["bg"]["TOTAL_NUMBER_OF_POPULATIONS"])) + '/model_' + str(simparam.modelParamsID) + '/'

        with open(folder_name + "connection_lists_i.dat", "rb") as f:
            compressed_pickle_CONNECTION_LISTS_i = f.read()
        CONNECTION_LISTS_i_pickle = pickle.loads(blosc.decompress(compressed_pickle_CONNECTION_LISTS_i))

        for name in CONNECTION_LISTS_i_pickle:
            if verbose:
                print(name)
            CONNECTION_LISTS_i_active_cortex_inputs[name] = CONNECTION_LISTS_i_pickle[name]

        with open(folder_name + "connection_lists_i.dat", "rb") as f:
            compressed_pickle_CONNECTION_LISTS_j = f.read()
        CONNECTION_LISTS_j_pickle = pickle.loads(blosc.decompress(compressed_pickle_CONNECTION_LISTS_j))

        for name in CONNECTION_LISTS_j_pickle:
            if verbose:
                print(name)
            CONNECTION_LISTS_j_active_cortex_inputs[name] = CONNECTION_LISTS_j_pickle[name]

        if verbose:
            print("\n=====> Connection lists fetched in folder: " + folder_name + " \n")

        return CONNECTION_LISTS_i_active_cortex_inputs, CONNECTION_LISTS_j_active_cortex_inputs

    @staticmethod
    def view(CONNECTION_LISTS_i_active_cortex_inputs):
        """
        Create a network graph showing cortex-basalganglia connections.
        """
        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes for cortex and BG populations
        ctx_node = "Cortex"
        G.add_node(ctx_node, type="input", color="red")

        # Add BG population nodes
        for pop_name in CONNECTION_LISTS_i_active_cortex_inputs.keys():
            G.add_node(pop_name, type="cortex", color="blue")

        # Add edges based on connection data
        for pop_name in CONNECTION_LISTS_i_active_cortex_inputs.keys():
            num_connections = len(CONNECTION_LISTS_i_active_cortex_inputs[pop_name])
            G.add_edge(ctx_node, pop_name, weight=num_connections)

        # Create layout and plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]["weight"]/1000 for u,v in edges]  # scale for visibility
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=weights)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8)

        plt.title("Cortex to BasalGanglia Connections")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Print connection statistics
        print("Connection Statistics:")
        for pop_name in CONNECTION_LISTS_i_active_cortex_inputs.keys():
            num_conn = len(CONNECTION_LISTS_i_active_cortex_inputs[pop_name])
            print(f"  {pop_name}: {num_conn} connections")