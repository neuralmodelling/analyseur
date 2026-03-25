"""
=========================================================
View the connectivity and summary statistics of the model
=========================================================

Three figures are invisibly generated and saved under the current working directory and
under the sub-directory `~/topography/`

* figure 1: view all the projection patterns
* figure 2: summary of connections at population level
* figure 3: connectivity as a connectome diagram

Guide
------

+-------+--------------------------------------------+--------------------------------------------------------------------------------+
|Figure | Content                                    | Interpretation                                                                 |
+=======+============================================+================================================================================+
| 1     | view all the projection patterns           | :meth:`analyseur.cbgtc.visual.connections.Conn.plot_all_connectivity_matrices` |
+-------+--------------------------------------------+--------------------------------------------------------------------------------+
| 2     | summary of connections at population level | :meth:`analyseur.cbgtc.visual.connections.Conn.connections_bar_chart`          |
+-------+--------------------------------------------+--------------------------------------------------------------------------------+
| 3     | connectivity as a connectome diagram       | :meth:`analyseur.cbgtc.visual.connections.Conn.plot_population_connectome`     |
+-------+--------------------------------------------+--------------------------------------------------------------------------------+

View script
-----------
:mod:`analyseur.script.cbgtc.model_topography`

Download script
---------------
:file:`analyseur/script/cbgtc/model_topography.py`

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
        fig2, fig2_ax = conn.connections_bar_chart(show=False)
        fig3, fig3_ax = conn.plot_population_connectome(show=False)

        Path("topography").mkdir(parents=True, exist_ok=True)
        fig1.savefig("topography/Connectivity_matrices_of_" + reg_to_reg + ".png")
        fig2.savefig("topography/Summary_of_" + reg_to_reg + ".png")
        fig3.savefig("topography/Connectome_of_" + reg_to_reg + ".png")
        #
        plt.close()

        # for nuc_to_nuc in conn.source_target_pairs:
        #     pass

# RUN THE SCRIPT
if __name__ == "__main__":
    main()
