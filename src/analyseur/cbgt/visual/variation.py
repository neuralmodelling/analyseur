# ~/analyseur/cbgt/visual/variation.py
#
# Documentation by Lungsi 30 Oct 2025
#
# This contains function for SpikingStats
#

"""
+------------------------------+-----------------------------------------------------------------------------------------------------+
| Functions                    | Purpose                                                                                             |
+==============================+=====================================================================================================+
| :func:`plotCV`               | plots Coefficient of Variations of all the neurons in a population                                  |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| :func:`plotCV_in_ax`         | draws the Coefficient of Variations of all the neurons into a given `matplotlib.pyplot.axis`        |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| :func:`plotCV2`              | plots Local Coefficient of Variations of all the neurons in a population                            |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| :func:`plotCV2_in_ax`        | draws the Local Coefficient of Variations of all the neurons into a given `matplotlib.pyplot.axis`  |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| :func:`plotLV`               | plots Local Variations of all the neurons in a population                                           |
+------------------------------+-----------------------------------------------------------------------------------------------------+
| :func:`plotLV_in_ax`         | draws the Local Variations of all the neurons into a given `matplotlib.pyplot.axis`                 |
+------------------------------+-----------------------------------------------------------------------------------------------------+

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

====================================
Plot Local Coefficient of Variations
====================================

Same as documented above for plotting CV but using the function :func:`plotCV2` and :func:`plotCV2_in_ax`
imported as
::

    from analyseur.cbgt.visual.variation import plotCV2, plotCV2_in_ax

=====================
Plot Local Variations
=====================

Same as documented above for plotting CV but using the function :func:`plotLV` and :func:`plotLV_in_ax`
imported as
::

    from analyseur.cbgt.visual.variation import plotLV, plotLV_in_ax

.. raw:: html

    <hr style="border: 2px solid red; margin: 20px 0;">

"""

import matplotlib.pyplot as plt

from analyseur.cbgt.stats.isi import InterSpikeInterval
from analyseur.cbgt.stats.variation import Variations


##########################################################################
#    CV PLOT
##########################################################################

def plotCV_in_ax(ax, spiketimes_set, nucleus=None, mode=None):
    """
    Draws the Coefficient of Variation on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    OPTIONAL parameters

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    n_neurons = len(spiketimes_set)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    [all_isi, _] = InterSpikeInterval.compute(spiketimes_set)
    CVarr = Variations.computeCV(all_isi)
    vec_CV = CVarr.values()

    # window = __spikeanal.window
    # binsz = __spikeanal.binsz_100perbin
    # n_bins = round((window[1] - window[0]) / binsz)

    if orient=="horizontal":
        ax.barh(range(len(vec_CV)), vec_CV, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
        ax.set_xlabel("CV")
        ax.margins(y=0)
    else:
        ax.bar(range(len(vec_CV)), vec_CV, color="steelblue", edgecolor="black")
        # ax.hist(vec_CV, bins=n_bins, alpha=0.7, color="green", edgecolor="black", )
        ax.set_ylabel("CV")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("CV Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plotCV(spiketimes_set, nucleus=None, mode=None):
    """
    Visualize Coefficient of Variation of the given neuron population.

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    [OPTIONAL]

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    if mode=="portrait":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plotCV_in_ax(ax, spiketimes_set, nucleus=nucleus, mode=mode)

    plt.show()

    return fig, ax


##########################################################################
#    CV2 PLOT
##########################################################################

def plotCV2_in_ax(ax, spiketimes_set, nucleus=None, mode=None):
    """
    Draws the Local Coefficient of Variation on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    [OPTIONAL]

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    n_neurons = len(spiketimes_set)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    [all_isi, _] = InterSpikeInterval.compute(spiketimes_set)
    CV2arr = Variations.computeCV2(all_isi)
    vec_CV2 = CV2arr.values()

    if orient=="horizontal":
        ax.barh(range(len(vec_CV2)), vec_CV2, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
        ax.set_xlabel(r"CV_2")
    else:
        ax.bar(range(len(vec_CV2)), vec_CV2, color="steelblue", edgecolor="black")
        ax.set_ylabel(r"CV_2")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title(r"CV_2" + " Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plotCV2(spiketimes_set, nucleus=None, mode=None):
    """
    Visualize Local Coefficient of Variation of the given neuron population.

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    [OPTIONAL]

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    if mode=="portrait":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plotCV2_in_ax(ax, spiketimes_set, nucleus=nucleus, mode=mode)

    plt.show()

    return fig, ax


##########################################################################
#    LV PLOT
##########################################################################

def plotLV_in_ax(ax, spiketimes_set, nucleus=None, mode=None):
    """
    Draws the Local Variation on the given
    `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_

    :param ax: object `matplotlib.pyplot.axis``
    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    [OPTIONAL]

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    n_neurons = len(spiketimes_set)

    match mode:
        case "portrait":
            orient = "horizontal"
        case _:
            orient = "landscape"

    get_axis = lambda orient: "x" if orient=="horizontal" else "y"

    [all_isi, _] = InterSpikeInterval.compute(spiketimes_set)
    LVarr = Variations.computeLV(all_isi)
    vec_LV = LVarr.values()

    if orient=="horizontal":
        ax.barh(range(len(vec_LV)), vec_LV, color="steelblue", edgecolor="black")
        ax.set_ylabel("Neurons")
        ax.set_xlabel("LV")
    else:
        ax.bar(range(len(vec_LV)), vec_LV, color="steelblue", edgecolor="black")
        ax.set_ylabel("LV")
        ax.set_xlabel("Neurons")

    ax.grid(True, alpha=0.3, axis=get_axis(orient))

    nucname = "" if nucleus is None else " in " + nucleus
    ax.set_title("LV Distribution of " + str(n_neurons) + " neurons" + nucname)

    return ax

def plotLV(spiketimes_set, nucleus=None, mode=None):
    """
    Visualize Local Variation of the given neuron population.

    :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
    or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

    [OPTIONAL]

    :param neurons: `"all"` or `scalar` or `range(a, b)` or list of neuron ids like `[2, 3, 6, 7]`

            - `"all"` means subset = superset
            - `N` (a scalar) means subset of first N neurons in the superset
            - `range(a, b)` or `[2, 3, 6, 7]` means subset of selected neurons

    :param nucleus: string; name of the nucleus
    :param mode: "portrait" or None/landscape [default]
    :return: object `ax` with Rate Distribution plotting done into it

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """
    if mode=="portrait":
        fig, ax = plt.subplots(figsize=(6, 10))
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax = plotLV_in_ax(ax, spiketimes_set, nucleus=nucleus, mode=mode)

    plt.show()

    return fig, ax