# ~/analyseur/cbgt/visual/scalo.py
#
# Documentation by Lungsi 21 Oct 2025
#
# This contains function for SpikingStats
#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analyseur.cbgt.stats.wavelet import ContinuousWaveletTransform as cwt

class Scalogram(object):

    def __init__(self, spiketimes_superset):
        self.spiketimes_superset = spiketimes_superset
        # get_binary_spiketrains(spiketimes_superset, window=(0, 10), sampling_rate=None, neurons="all"):

    def plot_single(self, scales=None, wavelet=None, sampling_rate=None, window=None, sigma=None,
                    neuron_indx=None, show=True, save=False, nucleus=None,):
        [coefficients, frequencies, neuronid, time_axis] = \
            cwt.compute_cwt_single(self.spiketimes_superset, sampling_rate=sampling_rate,
                                   window=window, sigma=sigma,
                                   scales=scales, wavelet=wavelet, neuron_indx=neuron_indx)

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f" ({neuronid})"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, np.abs(coefficients),
                             shading="gouraud", cmap="YlOrRd")
        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax

    @classmethod
    def plot_avg(cls, spiketimes_set, sampling_rate=None, window=None,
                 sigma=None, scales=None, wavelet=None, neurons=None,
                 nucleus=None, show=True, save=False,):
        """
        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        [avg_coefficients, frequencies, yticks, time_axis] = \
            cwt.compute_cwt_avg(spiketimes_set, sampling_rate=sampling_rate,
                                window=window, neurons=neurons, sigma=sigma,
                                scales=scales, wavelet=wavelet, )

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f"( {len(yticks)} average)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, avg_coefficients,
                             shading="gouraud", cmap="YlOrRd")
        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax

    @classmethod
    def plot_sum(cls, spiketimes_set, sampling_rate=None, window=None,
                 sigma=None, scales=None, wavelet=None, neurons=None,
                 nucleus=None, show=True, save=False,):
        """
        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        [coefficients, frequencies, yticks, time_axis] = \
            cwt.compute_cwt_sum(self.spiketimes_superset, sampling_rate=sampling_rate,
                                window=window, neurons=neurons, sigma=sigma,
                                scales=scales, wavelet=wavelet, )

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f"( {len(yticks)} sum)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, coefficients,
                             shading="gouraud", cmap="YlOrRd")
        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax

    @classmethod
    def plot_avg_coi(cls, spiketimes_set, sampling_rate=None, window=None,
                     sigma=None, scales=None, wavelet=None, neurons=None,
                     nucleus=None, show=True, save=False,):
        """
        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        [avg_coefficients, frequencies, yticks, time_axis] = \
            cwt.compute_cwt_avg(spiketimes_set, sampling_rate=sampling_rate,
                                window=window, neurons=neurons, sigma=sigma,
                                scales=scales, wavelet=wavelet, )
        magnitude = np.abs(avg_coefficients)

        # Create Simple parabolic COI
        T = time_axis[-1]
        f_max = frequencies[0]
        f_min = frequencies[-1]

        # Create parabolic COI
        coi_freq = np.logspace(np.log10(f_min), np.log10(f_max), 50)
        coi_width = 0.15 * T * (coi_freq / f_min)**(-0.5)  # parabola shape

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f"( {len(yticks)} average)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        mesh = ax.pcolormesh(time_axis, frequencies, magnitude,
                             shading="gouraud", cmap="YlOrRd")
        # plot COI
        ax.plot(0.5 * T - coi_width, coi_freq, "w--", linewidth=2, alpha=0.8)
        ax.plot(0.5 * T + coi_width, coi_freq, "w--", linewidth=2, alpha=0.8)

        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        fig.colorbar(mesh, ax=ax, label="CWT Coefficient Magnitude")

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax

    @staticmethod
    def matlab_jet_cmap():
        colors = [(0, 0, 0.5), (0, 0, 1), (0, 0.5, 1), (0, 1, 1),
                  (0.5, 1, 0.5), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.5, 0, 0)]
        return LinearSegmentedColormap("matlab_jet", colors, N=256)

    def plot_avg_coi1(self, scales=None, wavelet=None, show=True, save=False, nucleus=None,
                     sampling_rate=None, window=None, neurons=None, sigma=None, ):
        [avg_coefficients, frequencies, yticks, time_axis] = \
            cwt.compute_cwt_avg(self.spiketimes_superset, sampling_rate=sampling_rate,
                                window=window, neurons=neurons, sigma=sigma,
                                scales=scales, wavelet=wavelet, )
        magnitude = np.abs(avg_coefficients)

        # Create Simple parabolic COI
        T = time_axis[-1]
        f_max = frequencies[0]
        f_min = frequencies[-1]

        # Create parabolic COI
        coi_freq = np.logspace(np.log10(f_min), np.log10(f_max), 50)
        coi_width = 0.15 * T * (coi_freq / f_min)**(-0.5)  # parabola shape

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f"( {len(yticks)} average)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        cmap = self.matlab_jet_cmap()

        mesh = ax.pcolormesh(time_axis, frequencies, np.abs(avg_coefficients),
                             shading="gouraud", cmap=cmap,
                             vmin=np.percentile(np.abs(avg_coefficients), 5),
                             vmax=np.percentile(np.abs(avg_coefficients), 95))

        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label("Magnitude", rotation=270, labelpad=15, fontsize=12)

        ax.tick_params(axis="both", which="major", labelsize=10)

        plt.tight_layout()

        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax


    def plot_avg_coi2(self, scales=None, wavelet=None, show=True, save=False, nucleus=None,
                     sampling_rate=None, window=None, neurons=None, sigma=None, ):
        [avg_coefficients, frequencies, yticks, time_axis] = \
            cwt.compute_cwt_avg(self.spiketimes_superset, sampling_rate=sampling_rate,
                                window=window, neurons=neurons, sigma=sigma,
                                scales=scales, wavelet=wavelet, )
        magnitude = np.abs(avg_coefficients)

        max_scale = len(scales)
        coi_freq = frequencies[-1] * np.sqrt(2)

        # Create Simple parabolic COI
        T = time_axis[-1] - time_axis[0]
        coi_times = np.linspace(0, T, len(frequencies))

        coi_curve = coi_freq * np.sqrt((coi_times - T/2)**2 + 1)

        nucname = "" if nucleus is None else " of " + nucleus
        suptitle = "Scalogram" + nucname + f"( {len(yticks)} average)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        cmap = plt.get_cmap("jet")

        mesh = ax.pcolormesh(time_axis, frequencies, magnitude,
                             shading="gouraud", cmap="jet",
                             vmin=np.percentile(magnitude, 10),
                             vmax=np.percentile(magnitude, 90))
        # plot COI
        ax.plot(time_axis[::len(time_axis)//len(frequencies)], coi_curve, "w--",
                linewidth=2, label="Cone of Influence")
        ax.fill_between(time_axis[::len(time_axis)//len(frequencies)], coi_curve,
                        frequencies[-1], alpha=0.2, color="white", hatch="//")

        ax.set_title(suptitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.lengend(loc="upper right", framealpha=0.9)

        cbar = plt.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label("Magnitude", rotation=270, labelpad=15, fontsize=12)


        plt.tight_layout()
        if show:
            plt.show()

        if save:
            plt.savefig(suptitle.replace(" ", "_"))

        return fig, ax

    def accurate_coi_scalogram(time_axis, frequencies, coefficients, scales, wavelet_name='cmor1.5-1.0'):
        """
        More accurate COI calculation based on wavelet properties
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        magnitude = np.abs(coefficients)

        # Create scalogram
        mesh = ax.pcolormesh(time_axis, frequencies, magnitude,
                             shading='gouraud', cmap='jet',
                             vmin=np.percentile(magnitude, 5),
                             vmax=np.percentile(magnitude, 95))

        # ===== ACCURATE COI CALCULATION =====
        # COI depends on the specific wavelet used
        if 'cmor' in wavelet_name:
            # For Morlet wavelet: COI ~ 2√2 × scale
            # Convert scales to equivalent periods
            coi_boundary = []
            for i, scale in enumerate(scales):
                # COI in time units (simplified)
                # For Morlet wavelet, the cone of influence expands with scale
                coi_time = scale * np.sqrt(2)
                # Convert to frequency space for plotting
                if i < len(frequencies):
                    coi_boundary.append(frequencies[i] * coi_time)

            # Create symmetric COI
            time_center = (time_axis[-1] + time_axis[0]) / 2
            time_from_center = np.abs(time_axis - time_center)

            # Plot COI for left and right edges
            half_len = len(time_axis) // 2
            coi_x_left = time_axis[:half_len]
            coi_x_right = time_axis[half_len:]

            # Interpolate COI boundary to match time axis
            from scipy.interpolate import interp1d
            if len(coi_boundary) > 1:
                coi_interp = interp1d(np.linspace(0, 1, len(coi_boundary)),
                                      coi_boundary, bounds_error=False,
                                      fill_value='extrapolate')

                coi_left = coi_interp(np.linspace(0, 1, half_len))
                coi_right = coi_interp(np.linspace(0, 1, half_len))[::-1]

                ax.plot(coi_x_left, coi_left, 'w--', linewidth=2.5, alpha=0.8, label='COI')
                ax.plot(coi_x_right, coi_right, 'w--', linewidth=2.5, alpha=0.8)

        # Final formatting
        ax.set_ylabel('Frequency [Hz]', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time [s]', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.2)
        ax.set_title('MATLAB-Style Scalogram with Cone of Influence',
                     fontsize=14, fontweight='bold')

        plt.colorbar(mesh, ax=ax, label='Magnitude')
        ax.legend()

        plt.tight_layout()
        plt.show()