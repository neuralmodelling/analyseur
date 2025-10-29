# ~/analyseur/cbgt/analytics/ratesparsity.py
#
# Documentation by Lungsi 29 Oct 2025
#

import numpy as np

class Sparsity(object):
    """
    Sparsity or sparsity index is a measure of how concentrated or distributed the firing is across the neurons in the population.
    It quantifies

    - sparse coding; measure of whether a few neurons do most of the firing
    - dense coding; measure of evenly distributed firings across the neurons (in the population)

    Mathematically, sparsity index is the ratio of mean rates and the mean squared rates subtracted from unity.


    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">

    """

    @staticmethod
    def interpret_sparsity_index(sparsity_value):
        """
        Interpret the given sparsity value :math:`\\psi`. Going with the hypothesis that

        - "only a small fraction of neurons are active at any one time"
        - average biological neuronal activity ratio "typically" ranges from near 0 to about 0.5, i.e, :math:`(0, 0.5)`
        - activity ratio above 0.5 cannot be considered sparse because same efficiency can be achieved by inversion

        we interpret :math:`\\psi \\in (0, 0.3)` to low sparsity and :math:`\\psi \\in (0.7, 1.0)` to high sparsity.
        For greater granularity this is further broken down to

        - :math:`[0, 0.2)`: very dense (most neurons are active)
        - :math:`(0.2, 0.4)`: dense (firing is distributed)
        - :math:`(0.4, 0.6)`: moderate (mixed activity)
        - :math:`(0.6, 0.8)`: sparse
        - :math:`(0.8, 1.0)`: very sparse (few neurons are active)

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        if sparsity_value < 0.2:
            interpretation = "Very dense coding; most neurons fire similarly"
        elif sparsity_value < 0.4:
            interpretation = "Dense coding; firing distributed across population"
        elif sparsity_value < 0.6:
            interpretation = "Moderate coding; mixed distribution"
        elif sparsity_value < 0.8:
            interpretation = "Sparse coding; few active neurons"
        else:
            interpretation = "Very sparse coding; very few active neurons"

        return interpretation

    @staticmethod
    def biointerpret_sparsity_index(sparsity_value, brain_region="unknown"):
        """
        Interpret the given sparsity value :math:`\\psi` in a biological context

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        interpretations = []

        if sparsity_value > 0.7:
            interpretations.extend([
                "Possible grandmother cell coding",
                "High stimulus specificity",
                "Vulnerable to cell loss",
                "Common in: Hippocampus, higher visual areas"
            ])
        elif sparsity_value > 0.4:
            interpretations.extend([
                "Distributed representation",
                "Moderate stimulus specificity",
                "Robust to cell loss",
                "Population coding strategy",
                "Common in: Cortex, mid-level sensory areas"
            ])
        else:
            interpretations.extend([
                "Dense population coding",
                "Low stimulus specificity",
                "High redundancy",
                "Very robust representation",
                "Common in: Early sensory areas, cerebellum"
            ])

        return interpretations

    @staticmethod
    def mathinterpret_sparsity_index(sparsity_value, brain_region="unknown"):
        pass

    @classmethod
    def analyze(cls, firing_rates, baseline_rates, response_rates):
        """

        **Formula**

        .. table:: Formula
        ====================================================================================================================== ===============================================================
          Definitions                                                                                                             Interpretation
        ====================================================================================================================== ===============================================================
         total neurons, :math:`n_{Nuc}`                                                                                           total number of neurons in the Nucleus
         neuron index, :math:`i`                                                                                                  i-th neuron in the pool of :math:`n_{Nuc}` neurons
         frequency, :math:`f^{(i)}`                                                                                               firing rate of i-th neuron
         :math:`\\vec{f} = [f^{(i)}]_{\\forall{i} \\in [1, n_{Nuc}],}`                                                                firing rates of all (:math:`n_{Nuc}`) neurons
         :math:`\\psi^{(i)}`                                                                                                      sparsity index of i-th neuron
         :math:`\\vec{\\psi} = [\\psi^{(i)}]_{\\forall{i} \\in [1, n_{Nuc}],}`                                                              sparsity indices of all (:math:`n_{Nuc}`) neurons
        ====================================================================================================================== ===============================================================

        Then the sparsity index vector is defined as

        .. math::

            \\vec{\\psi} = 1 - \\frac{\\mu\\left(\\vec{f}\\right)^2}{\\mu\\left(\\vec{f}^2\\right)}

        Let :math:`\\vec{\\beta}` be the firing rates of all the neurons prior to stimulus onset (baseline firing rate)
        and :math:`\\vec{\\rho}` be the firing rates of all the neurons at and after stimulus onset (response firing rate).
        Then their sparsity is given by

        .. math::

            \\vec{\\psi}_{\\beta} &= 1 - \\frac{\\mu\\left(\\vec{\\beta}\\right)^2}{\\mu\\left(\\vec{\\beta}^2\\right)} \n
            \\vec{\\psi}_{\\rho} &= 1 - \\frac{\\mu\\left(\\vec{\\rho}\\right)^2}{\\mu\\left(\\vec{\\rho}^2\\right)}

        and sparsity change is

        .. math::

            \\Delta = \\vec{\\psi}_{\\rho} - \\vec{\\psi}_{\\beta}

        .. raw:: html

            <hr style="border: 2px solid red; margin: 20px 0;">

        """
        # Comprehensive sparsity analysis
        overall_sparsity = 1 - (np.mean(firing_rates)**2 / np.mean(firing_rates**2))
        baseline_sparsity = 1 - (np.mean(baseline_rates)**2 / np.mean(baseline_rates**2))
        response_sparsity = 1 - (np.mean(response_rates)**2 / np.mean(response_rates**2))

        sparsity_change = response_sparsity - baseline_sparsity

        return {
            "overall_sparsity": overall_sparsity,
            "baseline_sparsity": baseline_sparsity,
            "response_sparsity": response_sparsity,
            "sparsity_change": sparsity_change,
            "interpretation": cls.interpret_sparsity_index(overall_sparsity),
            "biological_context": cls.biointerpret_sparsity_index(overall_sparsity),
        }
