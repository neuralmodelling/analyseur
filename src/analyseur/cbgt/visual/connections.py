# ~/analyseur/cbgt/visual/connections.py
#
# Documentation by Lungsi 12 March 2026

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from analyseur.cbgt.loader import FetchConnectionList
from analyseur.cbgt.parameters import SimulationParams

simparam = SimulationParams()
simparam.nuclei_thal = simparam.nuclei_thal + ["CMPf"]

class Conn(object):
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

        from analyseur.cbgt.visual.connectiions import Conn

    1.2. Assign path to data location
    `````````````````````````````````
    ::

        root_folder = "/path/to/data_folder/"

    The `root_folder` is the CBGT data directory whose structure is shown below

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
    Following the choice of desired connected regions

    * `"CTX->CTX"`
    * `"CTX->BG"`
    * `"CTX->THAL"`
    * `"BG->THAL"`
    * `"THAL->CTX"`
    * `"BG->BG"`
    * `"THAL->THAL"`

    Note that tests abbreviations are **not** case-sensitive. Instantiate for `"ctx->bg"`

    ::

        conn = Conn(rootfolder=root_folder, region_connections="ctx->bg")

        # or simply
        conn = Conn(root_folder, "ctx->bg")

    ---------
    2. Cases
    ---------
    For visualizing connection related stuffs invoke `conn.<method_name>` from the available options:

    +-------------------------------------------+--------------------------------+
    | Method name                               | Obligatory argument            |
    +===========================================+================================+
    | :py:meth:`.connections_bar_chart`         | no argument is mandatory       |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.overall_connections_bar_chart` | no argument is mandatory       |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.global_stats`                  | no argument is mandatory       |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_connectivity_matrix`      | string: "<nucleus>-><nucleus>" |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_density`                  | string: "<nucleus>-><nucleus>" |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_degree_distribution`      | string: "<nucleus>-><nucleus>" |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_all_density`              | no argument is mandatory       |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_global_connectivity`      | no argument is mandatory       |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_global_density`           | no argument is mandatory       |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_channel_projection`       | string: "<nucleus>-><nucleus>" |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_all_channel_projections`  | no argument is mandatory       |
    +-------------------------------------------+--------------------------------+
    | :py:meth:`.plot_population_connectome`    | no argument is mandatory       |
    +-------------------------------------------+--------------------------------+

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    # Dispatch table as a class attribute
    _FETCH_HANDLERS = {
        ("Cortex", "Cortex"): "within_cortex",
        ("BasalGanglia", "BasalGanglia"): "within_bg",
        ("Thalamus", "Thalamus"): "within_thalamus",
        ("Cortex", "BasalGanglia"): "cortex_to_bg",
        ("Cortex", "Thalamus"): "cortex_to_thalamus",
        ("BasalGanglia", "Thalamus"): "bg_to_thalamus",
        ("Thalamus", "Cortex"): "thalamus_to_cortex",
    }

    def __init__(self, rootfolder=None, region_connections="ctx->bg"):


        (self.source_region, self.target_region), fetch = self.__fetch_function(region_connections)

        self.conn_i, self.conn_j = fetch(rootfolder=rootfolder, verbose=True, nuclei_filter=True)

        self.source_target_pairs = list(self.conn_i.keys())
        self.n_pairs = len(self.source_target_pairs)

        self.unique_sources = {key.split("->")[0] for key in self.conn_i.keys() if "->" in key}
        self.unique_targets = {key.split("->")[1] for key in self.conn_i.keys()}

        self.n_channels = simparam.size_info["bg"]["TOTAL_NUMBER_OF_CHANNELS"]


    def __fetch_function(self, region_connections: str):
        # 1. Split
        if "->" not in region_connections:
            raise ValueError("Format must be 'Source->Target'")
        src_raw, dst_raw = region_connections.split("->")

        # 2. Normalise using your static method (or a helper)
        src_norm = self.__normalize_region(src_raw)   # returns e.g. "cortex"
        dst_norm = self.__normalize_region(dst_raw)   # returns e.g. "basalganglia"

        # 3. Validate against the canonical set
        if (src_norm, dst_norm) not in simparam.connected_regions:
            raise ValueError(f"Connection '{region_connections}' is not allowed")
        else:
            return (src_norm, dst_norm), getattr(FetchConnectionList, self._FETCH_HANDLERS[(src_norm, dst_norm)])


    @staticmethod
    def __normalize_region(name: str) -> str:
        clean = name.strip().lower().replace(" ", "").replace("_", "")
        if clean not in simparam.REGION_ALIASES:
            raise ValueError(f"Unknown region '{name}'")
        return simparam.REGION_ALIASES[clean]


    def __validate_n_channels(self, n_channels):
        """
        Check n_channels.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        if n_channels is None:
            n_channels = self.n_channels
        if not (1 <= n_channels <= self.n_channels):
            raise ValueError(f"n_channels must be between {1} and {self.n_channels}")

        return n_channels


    def __has_connections(self, pair):
        return len(self.conn_i[pair]) > 0 and len(self.conn_j[pair]) > 0


    def connections_bar_chart(self, show=True):
        """
        Show summary of connections at population level

        .. code-block:: text

            Source Region → Target Region Connectivity

            Populations   Number of Connections
            src  target
            --------------------------------
            R1a → R2a     ██████████████████████████
            R1b → R2a     ████
            R1a → R2b     ████
            R1b → R2c     █
            R1b → R2b     ▏

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        fig, ax = plt.figure(figsize=(12, 6))

        connection_counts = [len(self.conn_i[pair]) for pair in self.source_target_pairs]

        ax.bar(self.source_target_pairs, connection_counts, color="skyblue")
        ax.set_title(f"Title {self.source_region}→{self.target_region} Connections per Population")
        ax.set_xlabel(f"{self.target_region} Populations")
        ax.set_ylabel("Number of Connections")
        ax.ticks_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(connection_counts):
            ax.text(i, v, str(v), ha="center", va="bottom")

        fig.tight_layout()

        if show:
            plt.show()

        print("Connection Statistics:")
        for pair in self.source_target_pairs:
            i_conn = len(self.conn_i[pair])
            j_conn = len(self.conn_j[pair])
            print(f" {pair}: {i_conn} connections (should equal {j_conn})")

        return fig, ax


    def overall_connections_bar_chart(self, show=True):
        """
        Compare connection patterns across all populations

        .. code-block:: text

            Source Region → Target Region Connectivity

            Total Connections
            R1a→R2a █████████████████████████████████████████
            R1a→R2b ██
            R1b→R2a ██
            R1b→R2c ▏
            R1b→R2b ▏

            Unique Neurons
            Cortex:        R1a→R2a ████████  R1a→R2b ███████  R1b→R2a ███████
            BasalGanglia:  R1a→R2a ██████████████████████████  R1b→R2a ████████

            Avg Convergence (BG neurons)
            R1a→R2a █████████████████
            R1a→R2b █████████████
            R1b→R2c █████

            Avg Divergence (Cortex neurons)
            R1a→R2a █████████████████████████████████████
            R1a→R2b ██
            R1b→R2a ██

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Total connections
        totals = [len(self.conn_i[pair]) for pair in self.source_target_pairs]
        axes[0,0].bar(self.source_target_pairs, totals, color="lightblue")
        axes[0,0].set_title("Total Connections")
        axes[0,0].tick_params(axis="x", rotation=45)

        # Plot 2: Unique neurons
        unique_source = [len(set(self.conn_i[pair])) for pair in self.source_target_pairs]
        unique_target = [len(set(self.conn_j[pair])) for pair in self.source_target_pairs]

        x = np.arange(self.n_pairs)
        width = 0.35
        axes[0, 1].bar(x - width / 2, unique_source, width, label=f"{self.source_region}", alpha=0.7)
        axes[0, 1].bar(x + width / 2, unique_target, width, label=f"{self.target_region}", alpha=0.7)
        axes[0, 1].set_title("Unique Neurons")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.source_target_pairs, rotation=45)
        axes[0, 1].legend()

        # Plot 3: Convergence ratio
        convergence = [totals[i] / unique_target[i] if unique_target[i] > 0 else 0
                       for i in range(self.n_pairs)]
        axes[1, 0].bar(self.source_target_pairs, convergence, color="orange")
        axes[1, 0].set_title(f"Average Convergence (conns/{self.target_region} neurons)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Divergence ratio
        divergence = [totals[i] / unique_source[i] if unique_source[i] > 0 else 0
                      for i in range(self.n_pairs)]
        axes[1, 1].bar(self.source_target_pairs, divergence, color="green")
        axes[1, 1].set_title(f"Average Divergence (conns/{self.source_region} neurons)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        fig.tight_layout()

        if show:
            plt.show()

        return fig, axes


    def plot_connectivity_matrix(self, pair_name, show=True):
        """
        Plot connection matrix for a `<source nucleus>-><target nucleus>` (e.g `PTN->MSN`)

        .. code-block:: text

            Target neuron index
            ↑
            │ 42000 ┤                             ███████████████
            │       │                             ███████████████
            │ 32000 ┤              ███████████████
            │       │              ███████████████
            │ 21000 ┤      ███████████████
            │       │      ███████████████
            │ 10000 ┤██████████████
            │       │██████████████
            │     0 └──────────────────────────────────────────→ Source neuron index
                    0      2000      4000      6000      8000

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        if pair_name not in self.conn_i:
            print("Population not found")
            return

        i = np.array(self.conn_i[pair_name])
        j = np.array(self.conn_j[pair_name])

        fig, ax = plt.subplots(figsize=(8, 8))

        # ax.scatter(i, j, s=1, alpha=0.5)
        ax.scatter(i, j, s=1, alpha=0.3, rasterized=True)

        ax.set_xlim(0, max(i) + 1)
        ax.set_ylim(0, max(j) + 1)

        src_to_dst = pair_name.split("->")

        ax.set_xlabel(f"{src_to_dst[0]} neuron index")
        ax.set_ylabel(f"{src_to_dst[1]} neuron index")

        ax.set_title(f"Connectivity matrix: {self.source_region} ({src_to_dst[0]}) → {self.target_region} ({src_to_dst[1]})")

        # ax.set_aspect('equal')
        ax.grid(alpha=0.2)

        if show:
            plt.show()

        return fig, ax


    def plot_all_connectivity_matrices(self, show=True):
        """
        Shows all the projection patterns.

        .. code-block:: text

            Source → Target connectivity

            Target
            │
            │        ████
            │      ████
            │    ████
            │  ████
            └──────────────── Source

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        cols = len(self.unique_targets)
        rows = int(np.ceil(self.n_pairs / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))

        axes = axes.flatten()

        for idx, pair in enumerate(self.source_target_pairs):

            src_to_dst = pair.split("->")

            i = np.array(self.conn_i[pair])
            j = np.array(self.conn_j[pair])

            axes[idx].scatter(i, j, s=1, alpha=0.4)
            # axes[idx].set_title(pair)
            axes[idx].set_xlabel(f"{self.source_region} ({src_to_dst[0]})")
            axes[idx].set_ylabel(f"{self.target_region} ({src_to_dst[1]})")
            axes[idx].grid(alpha=0.2)

        # Hide empty panels
        for k in range(idx+1, len(axes)):
            axes[k].axis("off")

        fig.suptitle(f"{self.source_region} → {self.target_region} Connectivity (All Populations)")
        fig.tight_layout()

        if show:
            plt.show()

        return fig, axes


    def plot_density(self, pair_name, bins=100, show=True):
        """
        Plot density heatmap for a `<Source nucleus>-><Target nucleus>` (e.g `PTN->MSN`)

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
        i = np.array(self.conn_i[pair_name])
        j = np.array(self.conn_j[pair_name])

        fig, ax = plt.subplots(figsize=(8, 8))

        # ax.hist2d(i, j, bins=bins, cmap="inferno")
        h = ax.hist2d(i, j, bins=bins, cmap="inferno", rasterized=True)

        fig.colorbar(h[3], ax=ax, label="Number of connections")

        ax.set_xlim(0, max(i) + 1)
        ax.set_ylim(0, max(j) + 1)

        src_to_dst = pair_name.split("->")

        ax.set_xlabel(f"{src_to_dst[0]} neuron")
        ax.set_ylabel(f"{src_to_dst[1]} neuron")

        ax.set_title(f"Projection density: {self.source_region} ({src_to_dst[0]}) → {self.target_region} ({src_to_dst[1]})")

        if show:
            plt.show()

        return fig, ax


    def plot_all_density(self, bins=100, show=True):
        """
        Shows connection density patterns for all source region nucleus to target region nucleus.

        .. code-block:: text

            Projection Density: Source Region → Target

            Target neurons
            ↑
            │   [::::***:::#:]
            │        [:::**:*::]
            │             [::*:#::*]
            │                  [::**:#::]
            │
            └────────────────────────────────→ Source neurons

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        cols = len(self.unique_targets)
        rows = int(np.ceil(self.n_pairs / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, pair in enumerate(self.source_target_pairs):

            src_to_dst = pair.split("->")

            i = np.array(self.conn_i[pair])
            j = np.array(self.conn_j[pair])

            axes[idx].hist2d(i, j, bins=bins, cmap="inferno")
            # axes[idx].set_title(pair)
            axes[idx].set_xlabel(f"{self.source_region} ({src_to_dst[0]})")
            axes[idx].set_ylabel(f"{self.target_region} ({src_to_dst[1]})")

        for k in range(idx+1, len(axes)):
            axes[k].axis("off")

        fig.suptitle(f"Connection Density: {self.source_region} → {self.target_region}")
        fig.tight_layout()

        if show:
            plt.show()

        return fig, axes


    def plot_degree_distribution(self, pair_name, show=True):
        """
        Plot convergence and divergence patterns for a `<Source nucleus>-><Target nucleus>` (e.g `PTN->MSN`)

        .. code-block:: text

            Source divergence
            1 ███████████████████████████████████
            2 ███████████████████
            3 █████████
            4 ███
            5 █
            6 ▏
            7 ▏

            Target convergence
            1 █████████████████████████████████████████████████
            2 ▏
            3 ▏
            4 ▏

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        i = np.array(self.conn_i[pair_name])
        j = np.array(self.conn_j[pair_name])

        source_deg = np.bincount(i)
        target_deg = np.bincount(j)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(source_deg[source_deg>0], bins=50)
        axes[0].set_title(f"{self.source_region} divergence")
        axes[0].set_xlabel("Connections per neuron")

        axes[1].hist(target_deg[target_deg>0], bins=50)
        axes[0].set_title(f"{self.target_region} convergence")
        axes[0].set_xlabel("Connections per neuron")

        fig.tight_layout()

        if show:
            plt.show()

        return fig, axes


    def plot_global_connectivity(self, n_channels=None, band_height=2000, density_contours=False):
        """
        Shows global connectivity scatter plot with channel boundaries.

        .. code-block:: text

            Target neurons
            ↑
            |----|----|----|----|
            | ██ |    |    |    |
            |----|----|----|----|
            |    | ██ |    |    |
            |----|----|----|----|
            |    |    | ██ |    |
                                → Source neurons

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        n_channels = self.__validate_n_channels(n_channels)

        plt.figure(figsize=(10,8))

        yticks = []
        ylabels = []

        source_max = 0

        colors = plt.cm.tab10(np.linspace(0,1,len(self.conn_i)))

        for idx, pair in enumerate(self.conn_i.keys()):

            i = np.array(self.conn_i[pair])
            j = np.array(self.conn_j[pair])

            if len(i) == 0:
                continue

            source_max = max(source_max, i.max())

            # normalize target neurons inside band
            j_norm = (j - j.min()) / (j.max() - j.min() + 1e-9)
            j_scaled = j_norm * band_height + idx * band_height

            plt.scatter(i, j_scaled, s=1, alpha=0.4, color=colors[idx])

            # Density contours highlights where most synapses occur
            if density_contours:
                plt.hexbin(i, j_scaled, gridsize=100, cmap="inferno", alpha=0.6)

            yticks.append(idx * band_height + band_height/2)
            ylabels.append(pair)

        # draw source channel boundaries
        source_per_channel = source_max / n_channels

        for c in range(1, n_channels):
            plt.axvline(c * source_per_channel,
                        color="black",
                        linestyle="--",
                        alpha=0.3)

        # draw target population boundaries
        for p in range(1, len(self.conn_i)):
            plt.axhline(p * band_height,
                        color="black",
                        linewidth=1)

        plt.xlabel(f"{self.source_region} neuron index")
        plt.ylabel(f"{self.target_region} populations")

        plt.yticks(yticks, ylabels)

        plt.title(f"Global {self.source_region} → {self.target_region} Connectivity")

        plt.tight_layout()
        plt.show()


    def plot_global_density(self, bins=150, show=True):
        """
        Connectome-style plot.

        .. code-block:: text

            Source → Target projection density

            Target
            │      ░▓█
            │    ░▓█
            │  ░▓█
            │░▓█
            └──────── Source

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

        for pair in self.conn_i.keys():

            i = np.array(self.conn_i[pair])
            j = np.array(self.conn_j[pair])

            j_shifted = j + offset

            all_i.append(i)
            all_j.append(j_shifted)

            pop_offsets[pair] = offset

            if len(j) == 0:
                continue

            offset += max(j) + 10

        all_i = np.concatenate(all_i)
        all_j = np.concatenate(all_j)

        fig, ax = plt.subplots(figsize=(10, 8))

        # plt.hist2d(all_i, all_j, bins=bins, cmap="inferno")
        h = ax.hist2d(all_i, all_j, bins=bins, cmap="inferno", rasterized=True)

        fig.colorbar(h[3], ax=ax, label="Number of connections")

        ax.set_xlabel(f"{self.source_region} neurons")
        ax.set_ylabel(f"{self.target_region} neurons")

        ax.set_title(f"Global {self.source_region} → {self.target_region} Projection Density")

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax


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

        for pair in self.conn_i.keys():

            i = np.array(self.conn_i[pair])
            j = np.array(self.conn_j[pair])

            total = len(i)
            unique_source = len(np.unique(i))
            unique_target = len(np.unique(j))

            convergence = total / unique_target if unique_target else 0
            divergence = total / unique_source if unique_source else 0

            print(f"{pair}")
            print(f"  total connections            : {total}")
            print(f"  {self.source_region} neurons : {unique_source}")
            print(f"  {self.target_region} neurons : {unique_target}")
            print(f"  convergence                  : {convergence:.2f}")
            print(f"  divergence                   : {divergence:.2f}")
            print()

    def plot_channel_projection(self, pair_name, n_channels=None, show=True):
        """
        Show channel-projection map for desired population pair as diagonal channel blocks.

        .. code-block:: text

                   Target channels
                    0   1   2   3
            Sx 0    ██
            Sx 1        ██
            Sx 2            ██
            Sx 3                ██

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

        i = np.array(self.conn_i[pair_name])
        j = np.array(self.conn_j[pair_name])

        # estimate neurons per channel
        source_per_channel = max(i) // n_channels + 1
        target_per_channel = max(j) // n_channels + 1

        source_channels = i // source_per_channel
        target_channels = j // target_per_channel

        matrix = np.zeros((n_channels, n_channels))

        for cx, bg in zip(source_channels, target_channels):
            matrix[cx, bg] += 1

        fig, ax = plt.subplots(figsize=(6, 6))

        im = ax.imshow(matrix, cmap="inferno", origin="lower")

        fig.colorbar(im, ax=ax, label="Number of connections")

        # Set ticks from 1 to n_channels
        ax.set_xticks(range(n_channels), range(1, n_channels + 1))
        ax.set_yticks(range(n_channels), range(1, n_channels + 1))

        src_to_dst = pair_name.split("->")

        ax.set_xlabel(f"{src_to_dst[0]} channel")
        ax.set_ylabel(f"{src_to_dst[1]} channel")

        ax.set_title(f"Channel Projection: {self.source_region} ({src_to_dst[0]}) → {self.target_region} ({src_to_dst[1]})")

        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax


    def plot_all_channel_projections(self, n_channels=None):
        """
        Show channel-projection map for all population pairs using :py:meth:`.plot_channel_projection`

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        for pair in self.conn_i.keys():

            if not self.__has_connections(pair):
                continue

            print("Population:", pair)
            self.plot_channel_projection(pair, n_channels)


    def plot_population_connectome(self, show=True):
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
        legend_elements = [
            Patch(facecolor='lightgreen', label='Source'),
            Patch(facecolor='salmon', label='Target'),
            # Patch(facecolor='lightblue', label='Intermediate'),
            ]

        G = nx.DiGraph()
        edges = []

        for pair in self.conn_i.keys():
            n_conn = len(self.conn_i[pair])
            if "->" in pair:
                src, dst = pair.split("->")
            else:
                src = f"{self.source_region}"
                dst = pair
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

        # pos = nx.circular_layout(G)
        pos = nx.spring_layout(G, seed=42)

        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])

        # widths = 1 + 6 * weights / weights.max() if weights.size > 0 else []
        if weights.size > 0 and weights.max() > 0:
            widths = 1 + 6 * weights / weights.max()
        else:
            widths = np.ones_like(weights)

        fig, ax = plt.subplots(figsize=(7, 7))

        # nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=color_map)
        # nx.draw_networkx_labels(G, pos)
        # if widths.size > 0:
        #     nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowsize=20)

        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=color_map, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowsize=20, ax=ax)

        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_title(f"{self.source_region} → {self.target_region} Population Connectivity")
        ax.axis("off")

        if show:
            plt.show()

        return fig, ax
