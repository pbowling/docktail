# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Make the package importable from the project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

project = 'docktail'
copyright = '2026, Paige E. Bowling, Furyal Ahmed, Charles L. Brooks III'
author = 'Paige E. Bowling, Furyal Ahmed, Charles L. Brooks III'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
	'sphinx.ext.autosummary',
	'sphinx.ext.mathjax',
	'myst_parser',
]

# generate autosummary stub pages
autosummary_generate = True

# Prefer the ReadTheDocs theme; fallback to alabaster if not installed
try:
	import sphinx_rtd_theme  # noqa: F401
	html_theme = 'sphinx_rtd_theme'
except ImportError:
	html_theme = 'alabaster'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
