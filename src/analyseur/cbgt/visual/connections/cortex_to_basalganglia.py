import pickle
import blosc # allows to compress the lists

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.parameters import SimulationParams

simparam = SimulationParams()

class CtxToBG(object):
    """
    =========
    Use Cases
    =========

    ------------------
    1. Pre-requisites
    ------------------

    1.1. Import Modules
    ````````````````````
    ::

        from analyseur.cbgt.visual.connections.cortex_to_basalganglia import CtxToBG

    1.2. Assign path to data location
    `````````````````````````````````
    ::

        rootfolder = "/path/to/data_folder/"

    The `rootfolder` is the CBGT data directory whose structure is shown below

    .. code-block:: text

        .
        ├── BG/
        │   ├── connection_list/
        │   │   ├── scale=4_nbchannels=4/
        │   │   │   └── model_9/
        │   │   └── active_cortex_inputs_scale=4_nbchannels=4/
        │   │       └── model_9/
        │   └── ...
        ├── CORTEX/
        │   ├── connection_list/
        │   │   ├── Thalamus_inputs_nbpops=4/
        │   │   └── nbpops=4/
        │   └── ...
        ├── THALAMUS/
        │   ├── connection_list/
        │   │   ├── nbpops=4/
        │   │   ├── BG_inputs_nbpops=4/
        │   │   └── active_cortex_inputs_nbpops=4/
        │   └── ...
        ├── ...
        :

    where

    * terminal folders in `connection_list/` contains files `connection_lists_i.dat` and `connection_lists_j.dat`

    1.3. Instantiate class object
    `````````````````````````````
    ::

        conn = CtxToBG(rootfolder)

    ---------
    2. Cases
    ---------
    For visualizing connection related stuffs invoke `conn.<method_name>` from the available options:

    +-------------------------------------------+
    | Method name                               |
    +===========================================+
    | :py:meth:`.connections_bar_chart`         |
    +-------------------------------------------+
    | :py:meth:`.overall_connections_bar_chart` |
    +-------------------------------------------+
    | :py:meth:`.global_stats`                  |
    +-------------------------------------------+
    | :py:meth:`.plot_connectivity_matrix`      |
    +-------------------------------------------+
    | :py:meth:`.plot_density`                  |
    +-------------------------------------------+
    | :py:meth:`.plot_all_density`              |
    +-------------------------------------------+
    | :py:meth:`.plot_degree_distribution`      |
    +-------------------------------------------+
    | :py:meth:`.plot_global_connectivity`      |
    +-------------------------------------------+
    | :py:meth:`.plot_global_density`           |
    +-------------------------------------------+
    | :py:meth:`.plot_channel_projection`       |
    +-------------------------------------------+
    | :py:meth:`.plot_all_channel_projections`  |
    +-------------------------------------------+
    | :py:meth:`.plot_population_connectome`    |
    +-------------------------------------------+

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    def __init__(self, rootfolder=None):
        self.conn_i, self.conn_j = self.__fetch_connection_lists_active_cortex_inputs(rootfolder=rootfolder,
                                                                                    verbose=True)
        self.source_target_pairs = list(self.conn_i.keys())  # eg PTN->MSN
        self.n_pairs = len(self.source_target_pairs)

        self.n_channels = simparam.size_info["bg"]["TOTAL_NUMBER_OF_CHANNELS"]


    @staticmethod
    def __fetch_connection_lists_active_cortex_inputs(rootfolder=None, verbose=False):
        """
        Allows to fetch the synapses connection lists between the cortex and basal ganglia

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
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

    def __validate_n_channels(self, n_channels):
        if n_channels is None:
            n_channels = self.n_channels
        if not (1 <= n_channels <= self.n_channels):
            raise ValueError(f"n_channels must be between {1} and {self.n_channels}")

        return n_channels


    def connections_bar_chart()(self):
        """
        Show summary of connections at population level

        .. code-block:: text

            Cortex → Basal Ganglia Connectivity

            Populations   Number of Connections
            --------------------------------
            CSN → MSN     ██████████████████████████
            PTN → MSN     ████
            CSN → FSI     ████
            PTN → STN     █
            PTN → FSI     ▏

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        plt.figure(figsize=(12, 6))

        populations = list(self.conn_i.keys())
        connection_counts = [len(self.conn_i[pop]) for pop in self.source_target_pairs]

        plt.bar(self.source_target_pairs, connection_counts, color="skyblue")
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
        for pop in self.source_target_pairs:
            i_conn = len(self.conn_i[pop])
            j_conn = len(self.conn_j[pop])
            print(f" {pop}: {i_conn} connections (should equal {j_conn})")


    def overall_connections_bar_chart(self):
        """
        Compare connection patterns across all populations

        .. code-block:: text

            Cortex → Basal Ganglia Connectivity

            Total Connections
            CSN→MSN █████████████████████████████████████████
            CSN→FSI ██
            PTN→MSN ██
            PTN→STN ▏
            PTN→FSI ▏

            Unique Neurons
            Cortex:        CSN→MSN ████████  CSN→FSI ███████  PTN→MSN ███████
            BasalGanglia:  CSN→MSN ██████████████████████████  PTN→MSN ████████

            Avg Convergence (BG neurons)
            CSN→MSN █████████████████
            CSN→FSI █████████████
            PTN→STN █████

            Avg Divergence (Cortex neurons)
            CSN→MSN █████████████████████████████████████
            CSN→FSI ██
            PTN→MSN ██

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Total connections
        totals = [len(self.conn_i[pop]) for pop in self.source_target_pairs]
        axes[0,0].bar(self.source_target_pairs, totals, color="lightblue")
        axes[0,0].set_title("Total Connections")
        axes[0,0].tick_params(axis="x", rotation=45)

        # Plot 2: Unique neurons
        unique_cortex = [len(set(self.conn_i[pop])) for pop in self.source_target_pairs]
        unique_basalganglia = [len(set(self.conn_j[pop])) for pop in self.source_target_pairs]

        x = np.arange(self.n_pairs)
        width = 0.35
        axes[0, 1].bar(x - width / 2, unique_cortex, width, label="Cortex", alpha=0.7)
        axes[0, 1].bar(x + width / 2, unique_basalganglia, width, label="BasalGanglia", alpha=0.7)
        axes[0, 1].set_title("Unique Neurons")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.source_target_pairs, rotation=45)
        axes[0, 1].legend()

        # Plot 3: Convergence ratio
        convergence = [totals[i] / unique_basalganglia[i] if unique_basalganglia[i] > 0 else 0
                       for i in range(self.n_pairs)]
        axes[1, 0].bar(self.source_target_pairs, convergence, color="orange")
        axes[1, 0].set_title("Average Convergence (conns/basalganglia neurons)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Divergence ratio
        divergence = [totals[i] / unique_cortex[i] if unique_cortex[i] > 0 else 0
                      for i in range(self.n_pairs)]
        axes[1, 1].bar(self.source_target_pairs, divergence, color="green")
        axes[1, 1].set_title("Average Divergence (conns/cortical neurons)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()


    def plot_connectivity_matrix(self, pop_name):
        """
        Plot connection matrix for a `<CTX nucleus>-><BG nucleus>` (e.g `PTN->MSN`)

        .. code-block:: text

            Basal Ganglia neuron index
            ↑
            │ 42000 ┤                             ███████████████
            │       │                             ███████████████
            │ 32000 ┤              ███████████████
            │       │              ███████████████
            │ 21000 ┤      ███████████████
            │       │      ███████████████
            │ 10000 ┤██████████████
            │       │██████████████
            │     0 └──────────────────────────────────────────→ Cortex neuron index
                    0      2000      4000      6000      8000

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        if pop_name not in self.conn_i:
            print("Population not found")
            return

        i = np.array(self.conn_i[pop_name])
        j = np.array(self.conn_j[pop_name])

        plt.figure(figsize=(8,8))

        plt.scatter(i, j, s=1, alpha=0.5)

        plt.xlabel("Cortex neuron index")
        plt.ylabel("Basal Ganglia neuron index")

        plt.title(f"Connectivity matrix: Cortex → {pop_name}")

        plt.grid(alpha=0.2)
        plt.show()


    def plot_all_connectivity_matrices(self):
        """
        Shows all the projection patterns.

        .. code-block:: text

            Cortex → Basal Ganglia (BG) connectivity

            BG
            │
            │        ████
            │      ████
            │    ████
            │  ████
            └──────────────── Cortex

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        cols = 3
        rows = int(np.ceil(self.n_pairs / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))

        axes = axes.flatten()

        for idx, pop in enumerate(self.source_target_pairs):

            ctx_to_bg = pop.split("->")

            i = np.array(self.conn_i[pop])
            j = np.array(self.conn_j[pop])

            axes[idx].scatter(i, j, s=1, alpha=0.4)
            # axes[idx].set_title(pop)
            axes[idx].set_xlabel(f"Cortex ({ctx_to_bg[0]})")
            axes[idx].set_ylabel(f"BG ({ctx_to_bg[1]})")
            axes[idx].grid(alpha=0.2)

        # Hide empty panels
        for k in range(idx+1, len(axes)):
            axes[k].axis("off")

        plt.suptitle("Cortex → Basal Ganglia Connectivity (All Populations)")
        plt.tight_layout()
        plt.show()


    def plot_density(self, pop_name, bins=100):
        """
        Plot density heatmap for a `<CTX nucleus>-><BG nucleus>` (e.g `PTN->MSN`)

        .. code-block:: text

            Basal Ganglia neurons
            ↑

            16000 |                           :##::*:*:#*
            14000 |                           **:#*::*:#:

            12000 |               :*#*::*:#*:
            10000 |               :*#::*:#*#:

             8000 |      *:#*#::*#:
             6000 |      :##::*:*#*

             4000 | *:#*::*:#:
             2000 | :*#::*:#*

                  ----------------------------------------------------→ Cortex neurons
                    0      2000      4000      6000      8000

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        i = np.array(self.conn_i[pop_name])
        j = np.array(self.conn_j[pop_name])

        plt.figure(figsize=(8,8))

        plt.hist2d(i, j, bins=bins, cmap="inferno")

        plt.colorbar(label="Number of connections")

        plt.xlabel("Cortex neuron")
        plt.ylabel("Basal Ganglia neuron")

        plt.title(f"Projection density: Cortex → {pop_name}")

        plt.show()


    def plot_all_density(self, bins=100):
        """
        Shows connection density patterns for all cortical nucleus to basal ganglia nucleus.

        .. code-block:: text

            Projection Density: Cortex → Basal Ganglia

            Basal Ganglia neurons
            ↑
            │   [::::***:::#:]
            │        [:::**:*::]
            │             [::*:#::*]
            │                  [::**:#::]
            │
            └────────────────────────────────→ Cortex neurons

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        cols = 3
        rows = int(np.ceil(self.n_pairs / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten()

        for idx, pop in enumerate(self.source_target_pairs):

            ctx_to_bg = pop.split("->")

            i = np.array(self.conn_i[pop])
            j = np.array(self.conn_j[pop])

            axes[idx].hist2d(i, j, bins=bins, cmap="inferno")
            # axes[idx].set_title(pop)
            axes[idx].set_xlabel(f"Cortex ({ctx_to_bg[0]})")
            axes[idx].set_ylabel(f"BG ({ctx_to_bg[1]})")

        for k in range(idx+1, len(axes)):
            axes[k].axis("off")

        plt.suptitle("Connection Density: Cortex → BG")
        plt.tight_layout()
        plt.show()


    def plot_degree_distribution(self, pop_name):
        """
        Plot convergence and divergence patterns for a `<CTX nucleus>-><BG nucleus>` (e.g `PTN->MSN`)

        .. code-block:: text

            Cortex divergence
            1 ███████████████████████████████████
            2 ███████████████████
            3 █████████
            4 ███
            5 █
            6 ▏
            7 ▏

            BG convergence
            1 █████████████████████████████████████████████████
            2 ▏
            3 ▏
            4 ▏

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        i = np.array(self.conn_i[pop_name])
        j = np.array(self.conn_j[pop_name])

        cortex_deg = np.bincount(i)
        bg_deg = np.bincount(j)

        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        plt.hist(cortex_deg[cortex_deg>0], bins=50)
        plt.title("Cortex divergence")
        plt.xlabel("Connections per neuron")

        plt.subplot(1,2,2)
        plt.hist(bg_deg[bg_deg>0], bins=50)
        plt.title("BG convergence")
        plt.xlabel("Connections per neuron")

        plt.tight_layout()
        plt.show()


    def plot_global_connectivity(self, n_channels=None, band_height=2000, density_contours=False):
        """
        Shows global connectivity scatter plot with channel boundaries.

        .. code-block:: text

            Basal Ganglia neurons
            ↑
            |----|----|----|----|
            | ██ |    |    |    |
            |----|----|----|----|
            |    | ██ |    |    |
            |----|----|----|----|
            |    |    | ██ |    |
                                → Cortex neurons

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        n_channels = self.__validate_n_channels(n_channels)

        plt.figure(figsize=(10,8))

        yticks = []
        ylabels = []

        cortex_max = 0

        colors = plt.cm.tab10(np.linspace(0,1,len(self.conn_i)))

        for idx, pop in enumerate(self.conn_i.keys()):

            i = np.array(self.conn_i[pop])
            j = np.array(self.conn_j[pop])

            cortex_max = max(cortex_max, i.max())

            # normalize BG neurons inside band
            j_norm = (j - j.min()) / (j.max() - j.min() + 1e-9)
            j_scaled = j_norm * band_height + idx * band_height

            plt.scatter(i, j_scaled, s=1, alpha=0.4, color=colors[idx])

            # Density contours highlights where most synapses occur
            if density_contours:
                plt.hexbin(i, j_scaled, gridsize=100, cmap="inferno", alpha=0.6)

            yticks.append(idx * band_height + band_height/2)
            ylabels.append(pop)

        # draw cortex channel boundaries
        cortex_per_channel = cortex_max / n_channels

        for c in range(1, n_channels):
            plt.axvline(c * cortex_per_channel,
                        color="black",
                        linestyle="--",
                        alpha=0.3)

        # draw BG population boundaries
        for p in range(1, len(self.conn_i)):
            plt.axhline(p * band_height,
                        color="black",
                        linewidth=1)

        plt.xlabel("Cortex neuron index")
        plt.ylabel("Basal Ganglia populations")

        plt.yticks(yticks, ylabels)

        plt.title("Global Cortex → Basal Ganglia Connectivity")

        plt.tight_layout()
        plt.show()


    def plot_global_density(self, bins=150):
        """
        Connectome-style plot.

        .. code-block:: text

            Cortex → BG projection density

            BG
            │      ░▓█
            │    ░▓█
            │  ░▓█
            │░▓█
            └──────── Cortex

            Legend:

            █ very high
            ▓ high
            ▒ medium
            ░ low
            zero

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        all_i = []
        all_j = []

        offset = 0
        pop_offsets = {}

        for pop in self.conn_i.keys():

            i = np.array(self.conn_i[pop])
            j = np.array(self.conn_j[pop])

            j_shifted = j + offset

            all_i.append(i)
            all_j.append(j_shifted)

            pop_offsets[pop] = offset

            offset += max(j) + 10

        all_i = np.concatenate(all_i)
        all_j = np.concatenate(all_j)

        plt.figure(figsize=(10,8))

        plt.hist2d(all_i, all_j, bins=bins, cmap="inferno")

        plt.colorbar(label="Number of connections")

        plt.xlabel("Cortex neurons")
        plt.ylabel("Basal Ganglia populations")

        plt.title("Global Cortex → BG Projection Density")

        plt.tight_layout()
        plt.show()


    def global_stats(self):
        """
        Returns connection statistics for each population connection pair:

        * total connections
        * number of cortex neurons
        * number of basal ganglia neurons
        * convergence
        * divergence

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        print("\nGlobal Connectivity Statistics\n")

        for pop in self.conn_i.keys():

            i = np.array(self.conn_i[pop])
            j = np.array(self.conn_j[pop])

            total = len(i)
            unique_cortex = len(np.unique(i))
            unique_bg = len(np.unique(j))

            convergence = total / unique_bg if unique_bg else 0
            divergence = total / unique_cortex if unique_cortex else 0

            print(f"{pop}")
            print(f"  total connections : {total}")
            print(f"  cortex neurons    : {unique_cortex}")
            print(f"  BG neurons        : {unique_bg}")
            print(f"  convergence       : {convergence:.2f}")
            print(f"  divergence        : {divergence:.2f}")
            print()

    def plot_channel_projection(self, pop_name, n_channels=None):
        """
        Show channel-projection map for desired population pair as diagonal channel blocks.

        .. code-block:: text

                     BG channels
                    0   1   2   3
            Cx 0    ██
            Cx 1        ██
            Cx 2            ██
            Cx 3                ██

        **Patterns References:**

        *Focused connectivity*

        .. code-block:: text

            █
              █
                █
                  █

        *Diffuse connectivity*

        .. code-block:: text

            ████
            ████
            ████

        *Surround inhibition*

        .. code-block:: text

             █
            ███
             █

        *Channel crosstalk*

        .. code-block:: text

            █ █
             █ █

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        n_channels = self.__validate_n_channels(n_channels)

        i = np.array(self.conn_i[pop_name])
        j = np.array(self.conn_j[pop_name])

        # estimate neurons per channel
        cortex_per_channel = max(i) // n_channels + 1
        bg_per_channel = max(j) // n_channels + 1

        cx_channels = i // cortex_per_channel
        bg_channels = j // bg_per_channel

        matrix = np.zeros((n_channels, n_channels))

        for cx, bg in zip(cx_channels, bg_channels):
            matrix[cx, bg] += 1

        plt.figure(figsize=(6,6))

        plt.imshow(matrix, cmap="inferno", origin="lower")
        plt.colorbar(label="Number of connections")

        # Set ticks from 1 to n_channels
        plt.xticks(range(n_channels), range(1, n_channels + 1))
        plt.yticks(range(n_channels), range(1, n_channels + 1))

        plt.xlabel("BG channel")
        plt.ylabel("Cortex channel")

        plt.title(f"Channel Projection: Cortex → {pop_name}")

        plt.tight_layout()
        plt.show()


    def plot_all_channel_projections(self, n_channels=None):
        """
        Show channel-projection map for all population pairs using :py:meth:`.plot_channel_projection`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        for pop in self.conn_i.keys():

            print("Population:", pop)
            self.plot_channel_projection(pop, n_channels)


    def plot_population_connectome(self):
        """
        Displays model connectivity as a connectome diagram.

        .. code-block:: text

                Cortex
                //   \\
              CSN     PTN
                \\   //
                Striatum
                //   \\
              FSI     MSN

        .. raw:: html

            <hr
        """
        G = nx.DiGraph()
        edges = []

        for pop in self.conn_i.keys():
            n_conn = len(self.conn_i[pop])
            if "->" in pop:
                src, dst = pop.split("->")
            else:
                src = "Cortex"
                dst = pop
            edges.append((src, dst, n_conn))

        for src, dst, weight in edges:
            G.add_edge(src, dst, weight=weight)

        # Determine node colors based on role
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        color_map = []
        for node in G.nodes():
            if out_deg[node] > 0 and in_deg[node] == 0:
                color_map.append("lightgreen")   # source nodes
            elif in_deg[node] > 0 and out_deg[node] == 0:
                color_map.append("salmon")       # target nodes
            else:
                color_map.append("lightblue")    # intermediate (both)

        pos = nx.circular_layout(G)
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        widths = 1 + 6 * weights / weights.max() if weights.size > 0 else []

        plt.figure(figsize=(7, 7))
        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=color_map)
        nx.draw_networkx_labels(G, pos)
        if widths.size > 0:
            nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowsize=20)

        plt.title("Cortex → Basal Ganglia Population Connectivity")
        plt.axis("off")
        plt.show()
