.. analyseur documentation master file, created by
   sphinx-quickstart on Tue Mar 17 16:18:27 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to analyseur's documentation!
=====================================

analyseur |github| is a module for analyzing (potentially) myriad
models developed (or involved) in `Dr. Benoît Girard's <https://www.isir.upmc.fr/personnel/girard/>`_ lab.

|

.. |github| image:: /images/logo-github.svg
   :height: 1.5ex
   :target: https://github.com/neuralmodelling/analyseur

.. image:: /images/analyseur.svg
   :height: 200px
   :alt: Analyseur
   :align: center

.. |pip| image:: /images/logo-pypi.svg
   :height: 1.5ex
   :target: https://pypi.org/project/neurosig-analyseur/

Installation
------------

|pip| install

.. code-block:: bash

   pip install neurosig-analyseur

Or install from |github| (source):

.. code-block:: bash

   git clone https://github.com/neuralmodelling/analyseur.git
   cd analyseur
   pip install -e .

For the `CBGTC model. <https://gitlab.isir.upmc.fr/cobathaco-catatonia/CBGTC>`_
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   simulate_cbgtc.rst
   cbgtc/contents.rst
   analyze_cbgtc.rst

For the `rBCBG model. <https://gitlab.isir.upmc.fr/cobathaco-catatonia/CBGTC>`_
-------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   rbcbg/contents.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
