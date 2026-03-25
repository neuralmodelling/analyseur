# ~/analyseur/cbgtc/stat/compute_shared.py
#
# Documentation by Lungsi 3 Oct 2025
#
# This contains function for loading the files
#

import numpy as np

def compute_grand_mean(all_neuron_stat=None):
    """
    Returns the grand/global mean of a given statistics of all the neurons in a nucleus.

    :param all_neuron_stat:
    :return: a number

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    stat_array = np.zeros(len(all_neuron_stat))

    i = 0
    for val in all_neuron_stat.values():
        stat_array[i] = val
        i += 1

    return np.mean(stat_array)

def autocorr(x):
    """
    Performs autocorrelation (biased normalized autocorrelation)

    .. math::

        \\rho(k) = \\frac{\\sum_{t=0}^{N-1-k}\\tilde{x}_t \\tilde{x}_{t+k}}{\\sum_{t=0}^{N-1}\\tilde{x}_t^2}

    where :math:`k=0,1,\\ldots,N-1` and

    .. math::

        \\tilde{x}_t = x_t - \\frac{1}{N}\\sum_{t=0}^{N-1}x_t

    Note that the Fourier transform of the biased normalized autocorrelation is the power spectrum
    (`Wiener-Khinchin Theorem <https://mathworld.wolfram.com/Wiener-KhinchinTheorem.html>`_). Thus,

    .. math::

        P(f) &= \\mathcal{F}\\{\\rho(k)\\} \n
        \\rho(k) &= \\mathcal{F}^{-1}\\{P(f)\\}

    Thus, the pipeline connects

    .. math::

        x(t) \\to \\rho(k) \\Leftrightarrow P(f)

    See :class:`~analyseur.cbgtc.stats.psd.PowerSpectrum`

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:]  # keep non-negative lags
    return corr / corr[0]         # normalize

def correlation_time(rho, binsz, method="zero_crossing"):
    """
    Returns correlation time :math:`\\tau` from :func:`autocorr`

    .. math::

        \\tau &= \\sum_{k=0}^\\infty \\rho(k)\\Delta t \n
        &\\approx \\Delta t \\sum_{k=0}^K \\rho(k)

    where :math:`\\Delta t` is the bin size and :math:`K` is the cutoff where
    correlation :math:`\\rho` becomes negligible.

    **Guide**

    .. table:: Guide
    ============= =========== ====== ==========
     regime        frequency   CV     τ
    ============= =========== ====== ==========
     asynchronous  none        ~1     small
     oscillatory   >0          <1     moderate
     synchronized  strong      low    large
    ============= =========== ====== ==========

    Therefore,

    .. math::

        \\tau = \\Delta t \\sum_{k=0}^K \\rho(k)

    measures how long the system remembers itself.

    .. raw:: html

        <hr style="border: 2px solid red; margin: 20px 0;">
    """
    rho = np.array(rho)

    if method == "zero_crossing":
        idx = np.where(rho <= 0)[0]
        K = idx[0] if len(idx) > 0 else len(rho)

    elif method == "threshold":
        eps = 0.05
        idx = np.where(rho < eps)[0]
        K = idx[0] if len(idx) > 0 else len(rho)

    else:
        K = len(rho)

    tau = binsz * np.sum(rho[:K])
    return tau
