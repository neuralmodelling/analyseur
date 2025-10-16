# ~/analyseur/cbgt/visual/tabular.py
#
# Documentation by Lungsi 14 Oct 2025
#
# This contains function for SpikingStats
#

import numpy as np
import matplotlib.pyplot as plt

import re

# from ..stats.isi import InterSpikeInterval
# from ..stats.variation import Variations
from analyseur.cbgt.stats.isi import InterSpikeInterval
from analyseur.cbgt.stats.variation import Variations
from analyseur.cbgt.stats.compute_shared import compute_grand_mean as cgm

class SpikingStats(object):

    def __init__(self, spiketimes_superset):
        self.spiketimes_superset = spiketimes_superset

    def _compute_ISI(self, all_neurons_isi):
        # isi = InterSpikeInterval.compute(self.spiketimes_superset)
        vec_mu = InterSpikeInterval.mean_freqs(all_neurons_isi)
        grand_mu = cgm(vec_mu)

        return vec_mu, grand_mu

    def _compute_CVs(self, all_neurons_isi):
        vec_CV = Variations.computeCV(all_neurons_isi)
        vec_CV2 = Variations.computeCV2(all_neurons_isi)

        grand_CV = cgm(vec_CV)
        grand_CV2 = cgm(vec_CV2)

        return vec_CV, vec_CV2, grand_CV, grand_CV2

    def _compute_LV(self, all_neurons_isi):
        vec_LV = Variations.computeLV(all_neurons_isi)

        grand_LV = cgm(vec_LV)

        return vec_LV, grand_LV

    def compute_stats(self):
        all_isi = InterSpikeInterval.compute(self.spiketimes_superset)
        [vec_mu, grand_mu] = self._compute_ISI(all_isi)
        [vec_CV, vec_CV2, grand_CV, grand_CV2] = self._compute_CVs(all_isi)
        [vec_LV, grand_LV] = self._compute_LV(all_isi)

        return {
            "mean_freqs": vec_mu,
            "CV_array": vec_CV,
            "CV2_array": vec_CV2,
            "LV_array": vec_LV,
            "grand_mean_freqs": grand_mu,
            "grand_CV": grand_CV,
            "grand_CV2": grand_CV2,
            "grand_LV": grand_LV,
        }

    def _extract_neuron_number(self, key):
        numbers = re.findall(r'\d+', key)
        return int(numbers[0])

    def _dict_to_array(self, dict_for_all_neurons):
        sorted_dict = {k: dict_for_all_neurons[k] for k in
                       sorted(dict_for_all_neurons.keys(), key=self._extract_neuron_number)}
        return np.array(list(sorted_dict.values()))

    def _shortened_array(self, dict_for_all_neurons, n=3):
        raw_arr = self._dict_to_array(dict_for_all_neurons)
        arr = np.round(raw_arr, 3) # 3 decimal points
        return f"[{', '.join(map(str, arr[:n]))}, ..., {', '.join(map(str, arr[-n:]))}]"

    def plot(self):
        computed_stats = self.compute_stats()

        # columns = (r"$\vec{\mu}$", r"$\mu_G$", r"$\vec{CV}$", r"$CV_G$", r"$\vec{LV}$", r"$LV_G$",)
        column_headers = ("Statistic", "Array-Values", "Mean")
        row_headers = [x for x in ("Mean Freq.", "Coeff. of Var.", "Local Coeff. of Var.", "Linear Var.")]

        cells = [[self._shortened_array(computed_stats["mean_freqs"]), str(computed_stats["grand_mean_freqs"])],
                 [self._shortened_array(computed_stats["CV_array"]), str(computed_stats["grand_CV"])],
                 [self._shortened_array(computed_stats["CV2_array"]), str(computed_stats["grand_CV2"])],
                 [self._shortened_array(computed_stats["LV_array"]), str(computed_stats["grand_LV"])]]

        fig, ax = plt.subplots(figsize=(10, 4))
        # Create Table
        table = ax.table(cellText=cells, rowLabels=row_headers, colLabels=column_headers[1:],
                         loc='center', cellLoc="center")

        # Style the column and row headers
        for i in range(len(column_headers)-1):
            table[(0,i)].set_facecolor("#2E86AB")
            table[(0,i)].set_text_props(weight="bold", color="white")

        for i in range(len(row_headers)):
            table[(i+1, -1)].set_facecolor("#A23B72")
            table[(i+1, -1)].set_text_props(weight="bold", color="white")

        # Alternate data cell (row) colors
        for i in range(len(row_headers)):
            color = "#F8F9FA" if i % 2 == 0 else "#E9ECEF"
            for j in range(len(column_headers)-1):
                table[(i+1, j)].set_facecolor(color)

        # Adjust table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)

        # Remove axis
        ax.axis("off")

        plt.title("Spiking Statistics Summary", pad=20, fontsize=14, weight="bold")

        plt.tight_layout()
        plt.show()

        return fig
