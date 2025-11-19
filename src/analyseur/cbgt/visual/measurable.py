# ~/analyseur/cbgt/visual/measurable.py
#
# Documentation by Lungsi 14 Nov 2025
#
import matplotlib.pyplot as plt

import numpy as np

from analyseur.cbgt.loader import LoadMembraneVorI
from analyseur.cbgt.parameters import SimulationParams, SignalAnalysisParams


class VoltageTrace(object):
    __siganal = SignalAnalysisParams()
    __sufix = "_V_syn_v_1msgrouped_mean_preprocessed4Matlab_SHIFT.csv"

    @classmethod
    def load_measurables(cls, fileloc, nucleus):
        filepath = fileloc + nucleus + cls.__sufix
        loadV = LoadMembraneVorI(filepath)

        return loadV.get_measurables()

    @classmethod
    def volt_trace_collections(cls, all_fileloc, nucleus):
        m = len(all_fileloc)
        for i, fileloc in enumerate(all_fileloc):
            volt_trace_array = cls.load_measurables(fileloc, nucleus)

            if i==0:
                n = volt_trace_array.shape[0]
                X = np.zeros((m, n))

            X[0, :] = volt_trace_array.reshape(1, n)

        return X, X.mean(axis=0)

    @classmethod
    def plot_in_ax(cls, ax, fileloc, nucleus, window=None):
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        t_start = window[0] * cls.__siganal._1000ms
        t_end = window[1] * cls.__siganal._1000ms

        i_start = int(t_start)
        i_end = int(t_end)

        Varr = cls.load_measurables(fileloc, nucleus)

        ax.plot(np.arange(t_start, t_end), Varr[i_start:i_end], color='green')
        ax.set_ylabel("Voltage (mV)")
        ax.set_xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Trace of mean voltage across neurons" + nucname)

        return ax

    @classmethod
    def plot(cls, fileloc, nucleus=None, window=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax = cls.plot_in_ax(ax, fileloc, nucleus, window=window)

        plt.show()

        return fig, ax


    @classmethod
    def plot_collection_in_ax(cls, ax, all_fileloc, nucleus, window=None):
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        t_start = window[0] / cls.__siganal._1000ms
        t_end = window[1] / cls.__siganal._1000ms

        i_start = int(t_start)
        i_end = int(t_end)

        Vmat, Vmu = cls.volt_trace_collections(all_fileloc, nucleus)

        ax.plot(np.arange(t_start, t_end), Vmu[i_start:i_end], color='red')
        ax.set_ylabel("Voltage (mV)")
        ax.set_xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Distribution of Spike Count Correlations, mean of neurons (across time)" + nucname)

        return ax

    @classmethod
    def plot_collection(cls, all_fileloc, nucleus=None, window=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax = cls.plot_collection_in_ax(ax, all_fileloc, nucleus, window=window)

        plt.show()

        return fig, ax
