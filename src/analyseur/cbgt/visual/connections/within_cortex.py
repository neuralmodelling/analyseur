import pickle
import blosc # allows to compress the lists

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from analyseur.cbgt.parameters import SimulationParams

simparam = SimulationParams()

class InCtx(object):
    """
    from analyseur.cbgt.visual.connections.within_cortex import InCtx
    rootfolder = "/home/lungsi/DockerShare/data/17Oct2025/"
    conn_i, conn_j = InCtx.fetch_connection_lists_Thalamus_inputs(rootfolder=rootfolder, verbose=True)
    ConnThalCtx.view(conn_i)
    """
    def __init__(self, rootfolder=None):
        self.conn_i, self.conn_j = self.fetch_connection_lists(rootfolder=rootfolder,
                                                               verbose=True,)
        self.populations_names = list(self.conn_i.keys())


    @staticmethod
    def fetch_connection_lists(rootfolder=None, verbose=False, disp_stats=False):
        """
        Allows to fetch the synapses connection lists within cortex
        """
        CORTEX_CONNECTION_LISTS_i = {}
        CORTEX_CONNECTION_LISTS_j = {}

        folder_name = rootfolder + 'CORTEX/connection_lists/nbpops=' + str(
            int(simparam.size_info["cortex"]['TOTAL_NUMBER_OF_POPULATIONS'])) + '/'

        with open(folder_name + "connection_lists_i.dat", "rb") as f:
            compressed_pickle_CORTEX_CONNECTION_LISTS_i = f.read()
        CORTEX_CONNECTION_LISTS_i_pickle = pickle.loads(blosc.decompress(compressed_pickle_CORTEX_CONNECTION_LISTS_i))

        for name in CORTEX_CONNECTION_LISTS_i_pickle:
            if verbose:
                print(name)
            CORTEX_CONNECTION_LISTS_i[name] = CORTEX_CONNECTION_LISTS_i_pickle[name]

        with open(folder_name + "connection_lists_j.dat", "rb") as f:
            compressed_pickle_CORTEX_CONNECTION_LISTS_j = f.read()
        CORTEX_CONNECTION_LISTS_j_pickle = pickle.loads(blosc.decompress(compressed_pickle_CORTEX_CONNECTION_LISTS_j))

        for name in CORTEX_CONNECTION_LISTS_j_pickle:
            if verbose:
                print(name)
            CORTEX_CONNECTION_LISTS_j[name] = CORTEX_CONNECTION_LISTS_j_pickle[name]

        if verbose:
            print("\n=====> Connection lists fetched in folder: " + folder_name + " \n")

        # TODO: make the "disp_stats" part more efficient, as it is very inefficient!
        if disp_stats:
            print("\n ========== STATISTIC DISPLAY OF CONNECTION LISTS ========== ")
            # print(' Cortex population sizes: ', CORTEX_POPULATION_SIZES)
            # for connection_name in CORTEX_CONNECTION_LISTS_i:
            #     output_nuclei = connection_name.split('->')[1]
            #     input_nuclei = connection_name.split('->')[0]
            #     print("\n====> ", connection_name, simparam.projection_types["cortex"][connection_name])
            #     array_i = np.array(CORTEX_CONNECTION_LISTS_i[connection_name])
            #     array_j = np.array(CORTEX_CONNECTION_LISTS_j[connection_name])
            #
            #     TOTAL_output = simparam.size_info["cortex"]['TOTAL_NUMBER_OF_POPULATIONS']
            #     TOTAL_input = simparam.size_info["cortex"]['TOTAL_NUMBER_OF_POPULATIONS']
            #
            #     for target_pop_id in range(TOTAL_output):
            #         list_num_connections_pops = [[] for k in range(TOTAL_input)]
            #         output_pop = output_nuclei + '_' + str(target_pop_id)
            #         target_pop_start_id = CORTEX_POPULATION_START_ID[
            #             output_pop]  # fetch correct neuron id range for this population
            #
            #         for j in range(target_pop_start_id, target_pop_start_id + CORTEX_POPULATION_SIZES[output_pop]):
            #             list_num_connections_pops_j = [0 for k in range(TOTAL_input)]
            #             indices_j = np.where(array_j == j)
            #             # print(list(indices_j[0]))
            #             for index in list(
            #                     indices_j[0]):  # look for connections to this j, and increment number of connections
            #                 # print(index)
            #                 for k in range(TOTAL_input):
            #                     input_pop = input_nuclei + '_' + str(k)
            #                     input_pop_start_id = CORTEX_POPULATION_START_ID[input_pop]
            #                     if array_i[index] >= input_pop_start_id and array_i[index] < input_pop_start_id + \
            #                             CORTEX_POPULATION_SIZES[input_pop]:
            #                         list_num_connections_pops_j[k] += 1
            #
            #             # save data once all connections are found
            #             for k in range(TOTAL_input):
            #                 list_num_connections_pops[k].append(list_num_connections_pops_j[k])
            #
            #         # ===Display data for this target POP
            #         print(' \ntarget pop ', target_pop_id)
            #         sum = 0  # summ of all inputs pops connections, for 1 single target POP!
            #         total = 0
            #         for k in range(TOTAL_input):
            #             print('     input pop id: ', k)
            #             print('         min number of connections: ', np.min(list_num_connections_pops[k]))
            #             print('         max number of connections: ', np.max(list_num_connections_pops[k]))
            #             print('         mean number of connections: ', np.mean(list_num_connections_pops[k]))
            #             # print(list_num_connections_POPs[k])
            #             sum += np.sum(list_num_connections_pops[k])
            #             total += len(list_num_connections_pops[k])  # nb output neurons in target pop
            #             # print(sum, total)
            #         mean_total_among_all_input_pops = TOTAL_input * sum / total  # multiply by total number of input pop for this nuclei as total should be counted only once to be correct!
            #         INDEGREE = CONNECTION_PROBAS[connection_name] * TOTAL_NUMBER_OF_NEURONS_PER_NUCLEI[input_nuclei]
            #         print("\n*EXPECTED in-degree: ", INDEGREE, ', re-computed in degree: ',
            #               mean_total_among_all_input_pops)
            #         # print("*EXPECTED MIN indegree: ", floor((1-heterogeneity_factor)*INDEGREE) ,
            #         #      "*EXPECTED MAX indegree: ",ceil((1+heterogeneity_factor)*INDEGREE))

        return CORTEX_CONNECTION_LISTS_i, CORTEX_CONNECTION_LISTS_j


    def view_summary(self):
        """
        Show summary of connections at population level
        """
        plt.figure(figsize=(12, 6))

        populations = list(self.conn_i.keys())
        connection_counts = [len(self.conn_i[pop]) for pop in populations]

        plt.bar(populations, connection_counts, color="skyblue")
        plt.title("Title ↻Cortex Connections per Population")
        plt.xlabel("Cortex Populations")
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
        unique_icortex = [len(set(self.conn_i[pop])) for pop in populations]
        unique_jcortex = [len(set(self.conn_j[pop])) for pop in populations]

        x = np.arange(len(populations))
        width = 0.35
        axes[0, 1].bar(x - width / 2, unique_icortex, width, label="i-Cortex", alpha=0.7)
        axes[0, 1].bar(x + width / 2, unique_jcortex, width, label="j-Cortex", alpha=0.7)
        axes[0, 1].set_title("Unique Neurons")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(populations, rotation=45)
        axes[0, 1].legend()

        # Plot 3: Convergence ratio
        convergence = [totals[i] / unique_jcortex[i] if unique_jcortex[i] > 0 else 0
                       for i in range(len(populations))]
        axes[1, 0].bar(populations, convergence, color="orange")
        axes[1, 0].set_title("Average Convergence (conns/j-cortex neurons)")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Divergence ratio
        divergence = [totals[i] / unique_icortex[i] if unique_icortex[i] > 0 else 0
                      for i in range(len(populations))]
        axes[1, 1].bar(populations, divergence, color="green")
        axes[1, 1].set_title("Average Divergence (conns/i-cortex neurons)")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    def view_actual_network(self, pop_name=None, max_neurons=100):
        """
        Visualize actual connections among the Cortical neurons
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

        # Add i-Cortex neurons
        icortex_nodes = set(i_neurons)
        for node in icortex_nodes:
            G.add_node(f"iCx_{node}", type="i-cortex", color="red")

        # Add j-Cortex neurons
        jcortex_nodes = set(j_neurons)
        for node in jcortex_nodes:
            G.add_node(f"jCx_{node}", type="j-cortex", color="blue")

        # Add Edges
        for i, j in zip(i_neurons, j_neurons):
            G.add_edge(f"iCx_{i}", f"jCx_{j}")

        # Create layout
        plt.figure(figsize=(15, 10))

        # Separate i-Cortex and j-Cortex nodes spatially
        pos = {}
        icortex_x = 0
        jcortex_x = 1

        # Position i-Cortex nodes
        icortex_list = sorted([n for n in G.nodes() if n.startswith("iCx_")])
        for i, node in enumerate(icortex_list):
            pos[node] = (icortex_x, i / max(1, len(icortex_list)))

        # Position j-Cortex nodes
        jcortex_list = sorted([n for n in G.nodes() if n.startswith("jCx_")])
        for i, node in enumerate(jcortex_list):
            pos[node] = (jcortex_x, i / max(1, len(jcortex_list)))

        # Draw with different colors
        node_colors = [G.nodes[node].get("color", "gray") for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6, arrowsize=10)

        plt.title(f"↻Cortex Connections: {pop_name}\n"
                  f"({len(icortex_list)} i-cortical neurons → {len(jcortex_list)} j-cortical neurons)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
