from distutils.core import setup

with open( "README.md", "r" ) as fh:
    long_description = fh.read()

setup(
  name = "analyseur",
  packages = [
      "analyseur",
      "analyseur.cbgt",
      #"nmcog.spinnaker.cognitiveplanning",
      #"nmcog.spinnaker.cognitiveplanning.nealplanning",
      ],
  version = "0.0.1",
  license="BSD-3-Clause",
  description = "Module (Library) for analyzing models (CBGTC, rBCBG) outputs.",
  long_description = long_description,
  long_description_content_type = "text/markdown",
  author = "Lungsi Ngwua",
  author_email = "neuralgraphs@gmail.com",
  url = "https://gitlab.isir.upmc.fr/cobathaco-catatonia/analyseur",
  #download_url = "https://github.com/NMCog/archive/v0.2.27.tar.gz",
  keywords = ["NEUROSCIENCE", "COMPUTATIONAL NEUROSCIENCE", "SIGNAL PROCESSING", "SPIKING", "BASAL GANGLIA"],
  #install_requires=[
  #        "numpy",
  #        "quantities",
  #        "neo",
  #        "spynnaker8",
  #    ],
  classifiers=[
    "Development Status :: 3 - Alpha",      # Choose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    "Intended Audience :: Developers",      # Define that your audience are developers
    "Topic :: Software Development :: Build Tools",
    "License :: OSI Approved :: BSD-3-Clause",
    "Programming Language :: Python :: 3.5",
    #"Programming Language :: Python :: 3.6",
  ],
  #python_requires = ">=3.5",
)