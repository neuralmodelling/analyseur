`cwt_scalogram.m <https://github.com/neuralmodelling/analyseur/blob/main/scripts/cbgtc/cwt_scalogram.m>`_: MATLAB script for generating magnitude scalogram
===========================================================================================================================================================

**MATLAB function**

.. code-block:: matlab
   function cwt_scalogram(desired_dataset)

**Description**
   Perform continuous wavelet transform (CWT) on population rate data from multiple simulations and save scalograms.

**Input**
   :param desired_dataset: Name of the dataset folder (e.g., '23Jan2026').
   :type desired_dataset: string

**Details**
   This function reads CSV files containing smoothed population rates for different nuclei and channels,
   averages across channels, computes the CWT using the `bump` wavelet, and saves two figures per simulation
   and nucleus: a time‑series plot with the CWT scalogram, and a standalone scalogram using MATLAB's built‑in
   `cwt` plot.

   **Data organization** (assumed):
   - The dataset folder must contain a subfolder `data_recorded/` with simulation folders named `model_9_simX_...`
   - Inside each simulation folder, files named `smoothed_pop_rate_<nucleus>_<channel>.csv` with columns `t_ms` and `smoothed_pop_rate`.

**Output**
   - Figures are saved in `<desired_dataset>/figures/` as PNG files.

**Usage**
   .. code-block:: matlab
      cwt_scalogram('23Jan2026');

**Source Code**
   .. literalinclude:: ../../scripts/matlab/cwt_scalogram.m
      :language: matlab
      :linenos:
