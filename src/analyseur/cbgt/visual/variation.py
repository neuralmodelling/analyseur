# ~/analyseur/cbgt/visual/variation.py
#
# Documentation by Lungsi 30 Oct 2025
#
# This contains function for SpikingStats
#

"""
==============================
Plot Coefficient of Variations
==============================

-----------------
1. Pre-requisites
-----------------

1.1. Import Modules
````````````````````
::

    from analyseur.cbgt.loader import LoadSpikeTimes
    from analyseur.cbgt.visual.variation import plotCV

1.2. Load file and get spike times
```````````````````````````````````
::

    loadST = LoadSpikeTimes("spikes_GPi.csv")
    spiketimes_superset = loadST.get_spiketimes_superset()

---------
2. Cases
---------

2.1. Visualize CV with default setting
``````````````````````````````````````
::

    [fig, ax] = plotCV(spiketimes_superset)

2.2. Visualize CV in portrait mode
``````````````````````````````````
::

    [fig, ax] = plotCV(spiketimes_superset, mode="portrait")

2.3. Visualize CV in portrait mode with nucleus name in title
`````````````````````````````````````````````````````````````
::

    [fig, ax] = plotCV(spiketimes_superset, mode="portrait", nucleus="GPi")

2.4. Create the plot for customization
``````````````````````````````````````
This is for power users who for instance want to insert the CV plot in their
collage of subplots.
::

    import matplotlib.pyplot as plt
    from analyseur.cbgt.visual.variation import plotCV_in_ax

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')

    ax1 = plotCV_in_ax(ax1, spiketimes_superset)
    ax2 = plotCV_in_ax(ax2, spiketimes_superset)

    plt.show()

NOTE: This example shows :func:`plotCV_in_ax` in default setting but this function works like
:func:`plotCV` therefore all the cases 2.1, 2.2 and 2.3 are applicable for :func:`plotCV_in_ax`.

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, alpha

import re

from analyseur.cbgt.curate import get_desired_spiketimes_subset
from analyseur.cbgt.stats.isi import InterSpikeInterval
from analyseur.cbgt.stats.variation import Variations
from analyseur.cbgt.parameters import SpikeAnalysisParams, SimulationParams

__spikeanal = SpikeAnalysisParams()
__simparams = SimulationParams()


##########################################################################
#    CV PLOT
##########################################################################

def plotCV_in_ax(ax, spiketimes_superset, nucleus=None, mode=None):
    """
    Draws the Coefficient of Variation on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param neurons: "all" [default] or list: range(a, b) or [1, 4, 5, 9]
    - :param nucleus: string; name of the nucleus
    - :param mode: "portrait" or None/landscape [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """

    n_neurons = len(spiketimes_superset)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    all_isi = InterSpikeInterval.compute(spiketimes_superset)
    CVarr = Variations.computeCV(all_isi)
    vec_CV = CVarr.values()

    window = __spikeanal.window
    binsz = __spikeanal.binsz_100perbin
    n_bins = round((window[1] - window[0]) / binsz)

    if orient=="horizontal":
        ax.barh(range(len(vec_CV)), vec_CV, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
        ax.set_xlabel("CV")
    else:
        ax.bar(range(len(vec_CV)), vec_CV, color="steelblue", edgecolor="black")
        # ax.hist(vec_CV, bins=n_bins, alpha=0.7, color="green", edgecolor="black", )
        ax.set_ylabel("CV")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("CV Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plotCV(spiketimes_superset, nucleus=None, mode=None):
    """
    Visualize Coefficient of Variation of the given neuron population.

    :param spiketimes_superset: Dictionary returned using :meth:`analyseur.cbgt.stats.isi.InterSpikeInterval.compute`

    OPTIONAL parameters

    - :param neurons: "all" or list: range(a, b) or [1, 4, 5, 9]
    - :param nucleus: string; name of the nucleus
    - :param mode: "portrait" or None/landscape [default]
    - :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    if mode=="portrait":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plotCV_in_ax(ax, spiketimes_superset, nucleus=nucleus, mode=mode)

    plt.show()

    return fig, ax