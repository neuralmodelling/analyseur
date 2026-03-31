# ~/analyseur/rbcbg/visual/powspec.py
#
# Documentation by Lungsi 4 Nov 2025
#
# This contains function for Visualizing Power Spectrum
#
import numbers

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import re

from analyseur.rbcbg.stats.psd import PowerSpectrum
from analyseur.rbcbg.parameters import SignalAnalysisParams


class VizPSD(object):
    """
    View Power Spectral Density (PSD) Class.

    +-----------------------------------+------------------------------------+
    | Plot in axis                      | View                               |
    +===================================+====================================+
    | :py:meth:`.plot_global_in_ax`     | overall frequency content          |
    +-----------------------------------+------------------------------------+
    | :py:meth:`.plot_tv_in_axis`       | temporal evolution of frequencies  |
    +-----------------------------------+------------------------------------+

    **Use Case:**

    1. Setup

    ::

      from analyseur.rbcbg.loader import LoadRates
      from analyseur.rbcbg.visual.powspec import VizPSD

      loadFR = LoadRates("GPiSNr_model_9_percent_0.csv")
      t_sec, rates_Hz = loadFR.get_rates()


    2. Power Spectral Density for the entire signal

    ::

      binsz = 0.001

      fig, ax = plt.subplots(figsize=(6, 10))

      ax = VizPSD.plot_global_in_ax(ax, rates_Hz, binsz, nucleus="GPiSNr")

      plt.show()

    3. Time-varying Power Spectrum

    ::

      fig, ax = plt.subplots(figsize=(6, 10))

      fig, ax = VizPSD.plot_tv_in_axis(fig, ax, rates_Hz, nucleus="GPiSNr")

      plt.show()

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """

    __siganal = SignalAnalysisParams()
    __xlabelsec = "Time (s)"
    __xlabelHz = "Frequency (Hz)"
    __ylabelPSD = "Power Spectral Density"


    @classmethod
    def plot_global_in_ax(cls, ax, mu_rate_arr, binsz, nucleus=None,
                          resolution=None, method=None, withbands=False, offset_title=None):
        """
        .. code-block:: text

            PSD (log scale)
            │
            │   ▲
            │  ▲ ▲
            │ ▲   ▲
            │▲     ▲
            │       ▲
            │        ▲
            │         ▲
            │          ▲
            │           ▲
            │            ▲____
            │                 \\______
            │                        \\________
            │                                 \\______
            └──────────────────────────────────────────→ f (Hz)
            strong low-freq power     →     high-freq attenuation

        Given a `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_ this draws
        the average (or single/global) power spectrum of the entire signal tells us the *overall frequency content*
        using :meth:`~analyseur.rbcbg.stats.psd.PowerSpectrum.compute_for_rate`.

        :param ax: object `matplotlib.pyplot.axis`
        :param mu_rate_array: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
        :param binsz: integer or float

        [OPTIONAL]

        :param nucleus: string; name of the nucleus
        :param method: `"welch"` or `"fft"` or `"fft-mag"`
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :param withbands: True or False [default]
        :param offset_title: x-axis offset; scalar or None [default]
        :return: ax with respective plotting

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============

        # Compute power spectrum using Welch's method
        freqs, power = PowerSpectrum.compute_for_rate(mu_rate_arr, binsz, method=method, resolution=resolution)

        # Plot power spectrum
        ax.semilogy(freqs, power, "b-", linewidth=1, label="Power Spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")

        nucname = "" if nucleus is None else " in " + nucleus
        if offset_title is None:
            ax.set_title("Power Spectrum of neurons" + nucname)
        else:
            ax.set_title("Power Spectrum of neurons" + nucname, x=offset_title)

        ax.grid(True, alpha=0.3)

        if withbands:
            freq_bands = cls.__siganal.freq_bands
            del freq_bands["Low Gamma"]
            del freq_bands["High Gamma"]

            # Add vertical lines and annotations for band boundaries
            band_boundaries = []
            band_labels = []
            for k, v in freq_bands.items():
                band_labels.append(k)
                if k=="Delta":
                    band_boundaries.append(v[0])
                    band_boundaries.append(v[1])
                else:
                    band_boundaries.append(v[1])

            for boundary in band_boundaries:
                ax.axvline(x=boundary, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

            # Add band labels at the top
            for i, (label, start, end) in enumerate(zip(band_labels, band_boundaries[:-1], band_boundaries[1:])):
                mid = (start + end) / 2
                ax.text(mid, ax.get_ylim()[1] * 0.9, label, ha="center", va="top", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            # ax.set_xlim(0, 100)
            ax.set_xlim(0, freqs.max())

        return ax

    @classmethod
    def plot_tv_in_axis(cls, fig, ax, mu_rate_arr, nucleus=None, resolution=None,
                        offset_title=None):
        """
        .. code-block:: text

            Frequency (Hz)
            │100 ─────────────────────────────────────────────
            │ 80 ─────────────────────────────────────────────
            │ 60 ────────────────:::::::::────────────────────
            │ 40 ───────────:::::::::::::::───────────:::::::::
            │ 20 ───────::::::::::::::::::::::::::::::: :::::::
            │ 10 ─────:::::::::::::::::::::::::::::::::::::::::
            │  5 ────::::::::::::::::::::::::::::::::::::::::::
            │  0 ─────────────────────────────────────────────
            │
            └──────────────────────────────────────────────────→ Time (s)
            0        2        4        6        8       10

            Legend:
            ":"  → higher power
            "."  → lower power
            " "  → minimal / background power

        Given a `matplotlib.pyplot.figure <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html>`_
        and its `matplotlib.pyplot.axis <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html>`_
        this draws the time-varying power spectrum telling us *how frequencies evolve*
        using :meth:`~analyseur.rbcbg.stats.psd.PowerSpectrum.compute_spectrogram`.

        :param fig: object `matplotlib.pyplot.figure`
        :param ax: object `matplotlib.pyplot.axis`
        :param mu_rate_array: array returned using :meth:`~analyseur.rbcbg.loader.LoadRates.get_rates`
        :param binsz: integer or float

        [OPTIONAL]

        :param nucleus: string; name of the nucleus
        :param resolution: `~ 9.76 Hz = sampling_rate/1024` [default]
        :param offset_title: x-axis offset; scalar or None [default]
        :return: fig and ax with respective plotting

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        freq_arr, time_arr, power_arr = PowerSpectrum.compute_spectrogram(mu_rate_arr, resolution=resolution)

        # Plot spectrogram (time-varying power)
        # im = ax.pcolormesh( time_arr, freq_arr, power_arr, shading="auto", cmap="viridis" )

        dt = np.diff(time_arr).mean()
        t_edges = np.concatenate([[time_arr[0] - dt/2], time_arr + dt/2])

        df = np.diff(freq_arr).mean()
        f_edges = np.concatenate([[freq_arr[0] - df/2], freq_arr + df/2])

        im = ax.pcolormesh(t_edges, f_edges, power_arr, shading="auto", cmap="viridis")

        # Labels
        ax.set_xlabel(cls.__xlabelsec)
        ax.set_ylabel(cls.__xlabelHz)

        nucname = "" if nucleus is None else " in " + nucleus
        if offset_title is None:
            ax.set_title("Time-varying Power Spectrum" + nucname)
        else:
            ax.set_title("Time-varying Power Spectrum" + nucname, x=offset_title)

        # Limit frequency range (optional but VERY useful)
        ax.set_ylim(0, 100)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)

        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Power (dB)")

        return fig, ax
