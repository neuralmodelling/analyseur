# ~/analyseur/cbgt/visual/composite/comp_spikes.py
#
# Documentation by Lungsi 23 Oct 2025
#
# This contains function for Peri-Stimulus Time Histogram (PSTH)
#
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
from scipy.special import linestyle

from analyseur.cbgt.loader import LoadSpikeTimes
from analyseur.cbgt.visual.raster import rasterplot
from analyseur.cbgt.visual.distribution import cv_distrib, mean_freq_distrib

# loadST = LoadSpikeTimes("spikes_GPi.csv")
# spiketimes_superset = loadST.get_spiketimes_superset()


def __copy_axes_contents(source_ax, target_ax):
    target_ax.clear()

    # Copy lines
    for line in source_ax.get_lines():
        xdata, ydata = line.get_data()
        target_ax.plot(xdata, ydata, color=line.get_color(),
                       linestyle=line.get_linestyle(),
                       linewidth=line.get_linewidth(),
                       label=line.get_label())

    # Copy collections (scatter, etc.)
    for collection in source_ax.collections:
        offsets = collection.get_offsets()
        if offsets.size > 0:
            target_ax.scatter(offsets[:,0], offsets[:,1],
                              color=collection.get_facecolor())

    # Copy patches (bars, etc.)
    for patch in source_ax.patches:
        target_ax.add_patch(plt.Rectangle((patch.get_x(), patch.get_y()),
                                          patch.get_width(), patch.get_height(),
                                          facecolor=patch.get_facecolor()))

    target_ax.set_xlabel(source_ax.get_xlabel())
    target_ax.set_ylabel(source_ax.get_ylabel())
    target_ax.set_title(source_ax.get_title())
    target_ax.grid(source_ax.get_gridlines() is not None)
    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())


def plot(spiketimes_superset, nucleus=None, show=True, save=False):
    fig1_1, ax1_1 = rasterplot(spiketimes_superset, neurons="all", nucleus=None, show=show)
    fig1_2, ax1_2 = cv_distrib(spiketimes_superset, orient="horizontal", show=show)
    fig1_3, ax1_3 = mean_freq_distrib(spiketimes_superset, orient="horizontal", show=show)

    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, width_ratios=[2, 1, 1]) # first gets 50% while rest is 25% each

    # Create subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    __copy_axes_contents(ax1_1, ax1)
    __copy_axes_contents(ax1_2, ax2)
    __copy_axes_contents(ax1_3, ax3)

    plt.tight_layout()
    plt.show()