=======================================================================================
Using `analyseur <https://github.com/neuralmodelling/analyseur>`_ to analyze rBCBG data
=======================================================================================

The rBCBG Model is `available here <https://gitlab.isir.upmc.fr/cobathaco-catatonia/bcbg-parents/rBCBG-ANNarchy/-/tree/disinhibition-experiments>`_

Regardless of how one sets up the simulation, to use this analyseur tool

Example simulation pipeline
===========================

1. Get (Go to) the model
------------------------

Get the model that has been prepared of graded disinhibition (`disinhibition-experiments` branch)

..  code-block:: shell

    git clone -b disinhibition-experiments ssh://git@gitlab.isir.lan:2222/cobathaco-catatonia/bcbg-parents/rBCBG-ANNarchy.git
    cd rBCBG-ANNarchy

2. Create a shell script for batch run
--------------------------------------

..  code-block:: python

    from core import multirun

    decaylist = [0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    results = multirun(decaylist)


3. Structure of stored results
------------------------------

Running the `multirun` will automatically create a subdirectory `decay` such that its sub-folder structure will look like

.. code-block:: text

    .
    ├── 0/
    │   ├── CTX_I_model_<ID>_percent_0.csv
    │   ├── CTX_E_model_<ID>_percent_0.csv
    │   ├── CSN_model_<ID>_percent_0.csv
    │   ├── FSI_model_<ID>_percent_0.csv
    │   ├── GPe_model_<ID>_percent_0.csv
    │   ├── GPiSNr_model_<ID>_percent_0.csv
    │   ├── MSN_model_<ID>_percent_0.csv
    │   ├── PTN_model_<ID>_percent_0.csv
    │   ├── STN_model_<ID>_percent_0.csv
    │   ├── TH_model_<ID>_percent_0.csv
    │   └── TRN_model_<ID>_percent_0.csv
    ├── 1/
    │   ├── CTX_I_model_<ID>_percent_1.csv
    │   ├── CTX_E_model_<ID>_percent_1.csv
    │   ├── CSN_model_<ID>_percent_1.csv
    │   ├── FSI_model_<ID>_percent_1.csv
    │   ├── GPe_model_<ID>_percent_1.csv
    │   ├── GPiSNr_model_<ID>_percent_1.csv
    │   ├── MSN_model_<ID>_percent_1.csv
    │   ├── PTN_model_<ID>_percent_1.csv
    │   ├── STN_model_<ID>_percent_1.csv
    │   ├── TH_model_<ID>_percent_1.csv
    │   └── TRN_model_<ID>_percent_1.csv
    ├── ...
    ├── ...
    ├── N/
        ├── CTX_I_model_<ID>_percent_N.csv
        ├── CTX_E_model_<ID>_percent_N.csv
        ├── CSN_model_<ID>_percent_N.csv
        ├── FSI_model_<ID>_percent_N.csv
        ├── GPe_model_<ID>_percent_N.csv
        ├── GPiSNr_model_<ID>_percent_N.csv
        ├── MSN_model_<ID>_percent_N.csv
        ├── PTN_model_<ID>_percent_N.csv
        ├── STN_model_<ID>_percent_N.csv
        ├── TH_model_<ID>_percent_N.csv
        └── TRN_model_<ID>_percent_N.csv

where

* respective terminal folder `1/`, `2/`, ..., `N/` contains many csv files that stores firing rates data of respective nucleus.

    * the data is an array with shape `(time_to_simulate, <region>_neurons)`
    * for example `(9999, 1)` for the basal ganglia neurons and `(9999, 24)` for the thalamic neurons

Scripts
=======

.. toctree::
   :maxdepth: 5
   :caption: Scripts:

   ../scripts/rbcbg/spectrograms.rst
