# ~/analyseur/cbgtc/visual/measurable.py
#
# Documentation by Lungsi 14 Nov 2025
#
import matplotlib.pyplot as plt

import numpy as np

from analyseur.cbgtc.loader import LoadMembraneVorI
from analyseur.cbgtc.parameters import SimulationParams, SignalAnalysisParams


class VoltageTrace(object):
    """
    +-----------------------------------------+
    | Methods                                 |
    +=========================================+
    | :py:meth:`.plot`                        |
    +-----------------------------------------+
    | :py:meth:`.plot_in_ax`                  |
    +-----------------------------------------+
    | :py:meth:`.plot_collection`             |
    +-----------------------------------------+
    | :py:meth:`.plot_collection_in_ax`       |
    +-----------------------------------------+
    | :py:meth:`.load_measurables`‡           |
    +-----------------------------------------+
    | :py:meth:`.volt_trace_collections`‡     |
    +-----------------------------------------+

    **NOTE:** ‡Are not plotting methods.

    =========
    Use Cases
    =========

    -----------------
    1. Pre-requisites
    -----------------

    1.1. Import Modules
    ````````````````````
    ::

        from analyseur.cbgtc.visual.measurable import VoltageTrace

    Consider the path
    ::

        locpath = "/data/BG/1/"
        all_paths = ["/data/BG/1/", "/data/BG/2/", "/data/BG/3/", "/data/BG/4/"]

    ------------
    2. Visualize
    ------------

    2.1. View average membrane voltage
    ``````````````````````````````````
    ::

        VoltageTrace.plot(locpath, "MSN")

    This plots the voltage trace for the *whole* simulation run.

    2.2. View average membrane voltage within a desired window
    ```````````````````````````````````````````````````````````
    ::

        VoltageTrace.plot(locpath, "MSN", window=(0,1))

    This plots the voltage trace inside the first second of the simulation.

    2.3. View average membrane voltage across multiple runs
    ```````````````````````````````````````````````````````
    ::

        VoltageTrace.plot_collection(all_paths, "MSN")

    For a desired window do
    ::

        VoltageTrace.plot_collection(all_paths, "MSN", window=(0,1))

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    __siganal = SignalAnalysisParams()
    __sufix = "_V_syn_v_1msgrouped_mean_preprocessed4Matlab_SHIFT.csv"

    @classmethod
    def load_measurables(cls, fileloc, nucleus):
        """
        Loads the membrane voltage preprocessed by averaging across all neurons
        in the respective nuclei and sampled at 1 ms. Note that this preprocessing
        and saving the data in respective csv file is done at simulation.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        filepath = fileloc + nucleus + cls.__sufix
        loadV = LoadMembraneVorI(filepath)

        return loadV.get_measurables()

    @classmethod
    def volt_trace_collections(cls, all_fileloc, nucleus):
        """
        This performs NumPy‑ization of multiple :py:meth:`.load_measurables` and
        return it and the mean of the coerced array.

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        m = len(all_fileloc)
        for i, fileloc in enumerate(all_fileloc):
            volt_trace_array = cls.load_measurables(fileloc, nucleus)

            if i==0:
                n = volt_trace_array.shape[0]
                X = np.zeros((m, n))

            X[i, :] = volt_trace_array.reshape(1, n)

        return X, X.mean(axis=0)

    @classmethod
    def plot_in_ax(cls, ax, fileloc, nucleus, window=None):
        """
        .. code-block:: text

            Voltage trace (mean across neurons)

            Voltage (mV)
               ^
               |   ~~~ ~~~~ ~~ ~~~ ~~ ~~~ ~~ ~~ ~~~ ~~~
               |  ~   ~~   ~~   ~~  ~~  ~~   ~~   ~~  ~
            -39 ─────────────────────────────────────────
               |   v      v        v     v        v
               |  v v    v v      v v   v v      v v
            -40 ─────────────────────────────────────────
               |
               +----------------------------------------> Time (ms)
                0            ...                 10000

        Displays the Membrane Voltage averaged over all the neurons in the desired nuclei as a time-series trace
        on the given `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_
        and returns the axis.

        :param ax: object `matplotlib.pyplot.axis``
        :param fileloc: string; "path/to/file"
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :return: object axis

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        # Convert the 1 ms sampled data to data in seconds
        t_start = window[0] * cls.__siganal._1000ms
        t_end = window[1] * cls.__siganal._1000ms

        # i_start = int(t_start)
        # i_end = int(t_end)

        Varr = cls.load_measurables(fileloc, nucleus)

        i_end = min(int(t_end), len(Varr))
        i_start = min(int(t_start), i_end)

        y = Varr[i_start:i_end].squeeze()
        x = np.arange(i_start, i_start + len(y))

        # ax.plot(np.arange(t_start, t_end), Varr[i_start:i_end], color='green')
        ax.plot(x, y, color='green')
        ax.set_ylabel("Voltage (mV)")
        ax.set_xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Trace of mean voltage across neurons" + nucname)

        return ax

    @classmethod
    def plot(cls, fileloc, nucleus=None, window=None):
        """
        Displays the Membrane Voltage averaged over all the neurons in the desired nuclei using :py:meth:`.plot_in_ax`

        :param fileloc: string; "path/to/file"
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :return: fig object and axis object comprising the plot

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        fig, ax = plt.subplots(figsize=(10, 6))

        ax = cls.plot_in_ax(ax, fileloc, nucleus, window=window)

        plt.show()

        return fig, ax


    @classmethod
    def plot_collection_in_ax(cls, ax, all_fileloc, nucleus, window=None):
        """
        .. code-block:: text

            Mean Membrane Voltage Across Simulations

            Voltage (mV)
               ^
               |   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            -48 ───────────────────────────────────────────
               |  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
               |   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            -49 ───────────────────────────────────────────
               |
               +------------------------------------------> Time (ms)
                0              ...                 10000

        Displays the Average Membrane Voltage over multiple runs in the desired nuclei as a time-series trace
        on the given `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_
        and returns the axis.

        :param ax: object `matplotlib.pyplot.axis``
        :param all_fileloc: ["path/to/file1", "path/to/file2", "path/to/file3", "path/to/file4"]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :return: object axis

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        t_start = window[0] * cls.__siganal._1000ms
        t_end = window[1] * cls.__siganal._1000ms

        Vmat, Vmu = cls.volt_trace_collections(all_fileloc, nucleus)

        i_end = min(int(t_end), len(Vmu))
        i_start = min(int(t_start), i_end)

        y = Vmu[i_start:i_end].squeeze()
        x = np.arange(i_start, i_start + len(y))

        ax.plot(x, y, color='red')
        ax.set_ylabel("Voltage (mV)")
        ax.set_xlabel("Time (ms)")

        nucname = "" if nucleus is None else " in " + nucleus
        ax.set_title("Average Membrane Voltage Trace Across Runs" + nucname)

        return ax

    @classmethod
    def plot_collection(cls, all_fileloc, nucleus=None, window=None):
        """
        Displays the Average Membrane Voltage over multiple runs in the desired nuclei using :py:meth:`.plot_collection_in_ax`

        :param all_fileloc: ["path/to/file1", "path/to/file2", "path/to/file3", "path/to/file4"]
        :param window: defines upper and lower range of the bins but ignore lower and upper outliers [default: (0,10000)]
        :param nucleus: [OPTIONAL] None or name of the nucleus (string)
        :return: fig object and axis object comprising the plot

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if window is None:
            window = cls.__siganal.window

        fig, ax = plt.subplots(figsize=(10, 6))

        ax = cls.plot_collection_in_ax(ax, all_fileloc, nucleus, window=window)

        plt.show()

        return fig, ax
