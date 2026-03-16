# ~/analyseur/cbgt/stats/popact.py
#
# Documentation by Lungsi 8 Dec 2025
#

from collections import Counter
import numpy as np

from analyseur.cbgt.parameters import SignalAnalysisParams

class PopAct(object):
    __siganal = SignalAnalysisParams()

    @classmethod
    def count_allspikes_per_bin(cls, spiketimes_set, binsz):
        """
        Returns spike counts (from any neuron) in each bin

        .. math::

            \\{C_k\\}_{k=0}^{K-1}

        where :math:`C_k` is the spike couns per bin :math:`k` given by

        .. math::

            C_k = \\sum_{i=1}^{n_\\text{nuc}} \\sum_{j=1}^{K_i}1(b_k \\le t_{ij} < b_{k+1})]

        with the indicator function :math:`1(.)`, bin width :math:`\\Delta t`, number of bins :math:`\\lceil(T_\\text{max} - T_\\text{min})/\\Delta t\\rceil` such that :math:`T_\\text{min} = \\text{min}(S_\\text{all}), T_\\text{max} = \\text{max}(S_\\text{all}), S_\\text{all} = \\bigcup_{i=1}^{n_\\text{nuc}} S_i, S_i = \\{t_{i1}, t_{i2}, \\ldots, t_{iK_i}\\}`, and bin edges :math:`b_k = T_\\text{min} + k\\Delta t` for :math:`k=0,1,\\ldots,K`. The bin centers are :math:`c_k = (b_k + b_{k+1})/2`.

        :param spiketimes_set: Dictionary returned using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_superset`
        or using :meth:`~analyseur.cbgt.loader.LoadSpikeTimes.get_spiketimes_subset`

        :param binsz: integer or float; `0.01` [default]

        :return: 2-tuple

        - array of spike counts per unique bins
        - array of time bin centers

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        # ============== DEFAULT Parameters ==============
        if binsz is None:
            binsz = cls.__siganal.binsz_100perbin

        # Find overall time range
        all_spikes = [tspike for spiketimes in spiketimes_set.values() for tspike in spiketimes]
        try:
            min_time = min(all_spikes)
            max_time = max(all_spikes)

            # Create time bins
            nbins = int(np.ceil((max_time - min_time) / binsz))
            bins = np.linspace(min_time, max_time, nbins + 1)

            spike_counts_per_bin = np.zeros(nbins, dtype=int)
            # Count spikes in each bin (from any neuron)
            for spiketimes in spiketimes_set.values():
                # Digitize; which bin each spike belongs to
                bin_indices = np.digitize(spiketimes, bins) - 1
                # Ensure indices are within valid range
                bin_indices = bin_indices[(bin_indices >= 0) & (bin_indices < nbins)]
                # Count unique bins (each neuron only once per bin)
                unique_bins, counts = np.unique(bin_indices, return_counts=True)
                spike_counts_per_bin[unique_bins] += counts
                # To count all spikes regardless of neuron
                # use np.bincount
            bin_centers = (bins[:-1] + bins[1:]) / 2
            # histo_bins = np.arange(spike_counts_per_bin.min() - 0.5,
            #                        spike_counts_per_bin.max() + 1.5, 1)

            return spike_counts_per_bin, bin_centers
        except:
            return np.zeros(10), np.zeros(10)

    @staticmethod
    def compute_complexity_pdf(spike_counts_per_bin):
        """
        .. math::

            P(k) = \\frac{n_k}{B}

        is the complexity probability distribution such that :math:`\\sum_{k=0}^{k_\\text{max}} P(k) = 1`, complexity in the spike count in a bin is :math:`k=0,1,\\ldots,k_\\text{max}`, and the number of bins whose spike count equal :math:`k` is given by

        .. math::

            n_k = \\sum_{b=1}^B 1(C_b = k)

        where :math:`1(.)` is the indicator function, :math:`B` is the number of time bins, :math:`C = \\{C_b\\}` is the spike counts per bin such that :math:`C_b` is the number of spikes in bin :math:`b`, and :math:`k_\\text{max} = \\text{max}(C)`

        Returns spike counts (from any neuron) in each bin

        :param spike_counts_per_bin: array returned using :meth:`.count_allspikes_per_bin`

        :return: 3-tuple

        - complexities ≜ array with the range [0, maximum spike count]
        - complexity_counts ≜ occurrences of each complexity
        - pdf ≜ array of each occurrences ÷ total number of bins

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">
        """
        nbins = len(spike_counts_per_bin)
        if not spike_counts_per_bin.any(): # all zeros
            return np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)
        else:
            # Count occurrences per complexity (spike counts per bin)
            occurrence_counts = Counter(spike_counts_per_bin)

            # Calculate pdf
            max_count = np.max(spike_counts_per_bin)
            successive_counts = np.arange(0, max_count + 1)

            # Create pdf
            pdf = np.zeros(len(successive_counts))
            if len(occurrence_counts)!=0:
                for i, k in enumerate(successive_counts):
                    pdf[i] = occurrence_counts.get(k, 0) / nbins
                # Numerically equal to below
                # pdf = np.array([np.sum(successive_counts == k)
                #                 for k in successive_counts]) / len(spike_counts_per_bin)
            # try:
            #     for i, k in enumerate(successive_counts):
            #         pdf[i] = occurrence_counts.get(k, 0) / nbins
            # except:
            #     pass

            return successive_counts, occurrence_counts, pdf
