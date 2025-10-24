# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
# sys.path.append( os.path.dirname( os.getcwd() ) )
sys.path.insert(0, os.path.abspath("../src"))

# -- Verify module structure -------------------------------------------------

# try:
#     from analyseur.cbgt.loader import *
#     print("Import successful!")
#     print([x for x in dir() if not x.startswith("_")])
# except Exception as e:
#     print(f"Import failed: {e}")


# -- Project information -----------------------------------------------------

project = 'analyseur'
copyright = '2025, Lungsi'
author = 'Lungsi'

# The full version, including alpha/beta/rc tags
release = '0.0.1 3-Alpha'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "recommonmark",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# to avoid breaking the building process due to external dependencies not met
autodoc_mock_imports = [
    "numpy", "pandas", "matplotlib", "mpl_toolkits",
    "scipy", "sklearn", "pywt",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# https://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
html_theme = 'agogo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', "guide"]

# -- Options for Notebooks ---------------------------------------------------

# Execute in advance. If notebooks tests code you may run them at build time.
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

#source_suffix = [".rst", ".ipynb"]