# ~/analyseur/cbgt/stat/variation.py
#
# Documentation by Lungsi 2 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

from utilities import compute_grand_mean as cgm

class Variations(object):
    """This class
    
    *
    *
    
    """

    @classmethod
    def computeCV(cls, all_neurons_isi=None):
        all_CV = {}

        for n_id, isi in all_neurons_isi.items():
            all_CV[n_id] = np.std(isi) / np.mean(isi)

        return all_CV

    @classmethod
    def computeCV2(cls, all_neurons_isi=None):
        all_CV2 = {}

        for n_id, isi in all_neurons_isi.items():
            abs_diff_over_sum = np.diff(isi) / (isi[1:] + isi[:-1])
            all_CV2[n_id] = np.mean( 2 * abs_diff_over_sum )

        return all_CV2

    @classmethod
    def computeLV(cls, all_neurons_isi=None):
        all_LV = {}

        for n_id, isi in all_neurons_isi.items():
            sq_diff_over_sum = np.square( np.diff(isi) ) / np.square( (isi[1:] + isi[:-1]) )
            all_LV[n_id] = np.mean( 3 * sq_diff_over_sum )

        return all_LV

    @classmethod
    def grandCV(cls, all_neurons_isi=None):
        all_neurons_CV = cls.computeCV(all_neurons_isi)
        return cgm(all_neurons_CV)

    @classmethod
    def grandCV2(cls, all_neurons_isi=None):
        all_neurons_CV2 = cls.computeCV2(all_neurons_isi)
        return cgm(all_neurons_CV2)

    @classmethod
    def grandLV(cls, all_neurons_isi=None):
        all_neurons_LV = cls.computeLV(all_neurons_isi)
        return cgm(all_neurons_LV)
