"""
========================================================
Population activity and membrane dynamics of 150 neurons
========================================================

The figure is invisibly generated and saved under the current working directory and
under the sub-directory `~/raster150_1s/`

Structure
---------

.. code-block:: text

    +---------------------+-----------+-----------+
    |                     |           |           |
    |      subplot 1      | subplot 2 | subplot 3 |
    |                     |           |           |
    +---------------------+-----------+-----------+
    |                     |                       |
    |      subplot 4      |       subplot 5       |
    |                     |                       |
    +---------------------+-----------------------+

Figure contains five subplots such that for each disinhibition experiment it plots:

* subplot 1: raster of the neurons
* subplot 2: CV distribution of the neurons
* subplot 3: mean rate of the neurons
* subplot 4: mean membrane voltage
* subplot 5: spike count distribution

Guide
------

+-------+---------------------------------------+--------------------------------------------------------------------------------+
|Figure | Content                               | Interpretation                                                                 |
+=======+=======================================+================================================================================+
| 1     | raster of all the neurons             | :meth:`analyseur.cbgtc.visual.connections.Conn.plot_all_connectivity_matrices` |
+-------+---------------------------------------+--------------------------------------------------------------------------------+
| 2     | CV distribution of all the neurons    | :meth:`analyseur.cbgtc.visual.connections.Conn.plot_population_connectome`     |
+-------+---------------------------------------+--------------------------------------------------------------------------------+

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">
"""

from pathlib import Path

import matplotlib.pyplot as plt

from analyseur.cbgtc.visual.connections import Conn

rootpath = "/home/lungsi/DockerShare/data/09Feb2026/"

connectregions = {"CTX->CTX": "Connections within Cortex",
                  "CTX->BG": "Cortex to BasalGanglia",
                  "CTX->THAL": "Cortex to Thalamus",
                  "BG->THAL": "BasalGanglia to Thalamus",
                  "THAL->CTX": "Thalamus to Cortex",
                  "BG->BG": "Connections within BasalGanglia",
                  "THAL->THAL": "Connections within Thalamus",
                  }

def main():

    for reg_to_reg in connectregions.keys():
        conn = Conn(rootfolder=rootpath, region_connections=reg_to_reg)

        fig1, fig1_ax = conn.plot_all_connectivity_matrices(show=False)
        fig2, fig2_ax = conn.plot_population_connectome(show=False)

        Path("topography").mkdir(parents=True, exist_ok=True)
        fig1.savefig("topography/Connectivity_matrices_of_" + reg_to_reg + ".png")
        fig2.savefig("topography/Connectome_of_" + reg_to_reg + ".png")
        #
        plt.close()

        # for nuc_to_nuc in conn.source_target_pairs:
        #     pass

# RUN THE SCRIPT
if __name__ == "__main__":
    main()
