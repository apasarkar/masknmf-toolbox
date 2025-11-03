# Configuration file for the Sphinx documentation builder.
# mostly copy-pasted from fastplotlib config file in Oct 2025
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# need to force offscreen rendering before importing fpl
# otherwise fpl tries to select glfw canvas
os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "1"

import fastplotlib
import pygfx
import masknmf
from pygfx.utils.gallery_scraper import find_examples_for_gallery
from pathlib import Path
import sys
from sphinx_gallery.sorting import ExplicitOrder
import imageio.v3 as iio


ROOT_DIR = Path(__file__).parents[1].parents[0]  # repo root
EXAMPLES_DIR = Path.joinpath(ROOT_DIR, "examples")

sys.path.insert(0, str(ROOT_DIR))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "masknmf"
copyright = "2025, Amol Pasarkar"
author = "Amol Pasarkar"
release = masknmf.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "myst_parser"
]

sphinx_gallery_conf = {
    "gallery_dirs": "_gallery",
    "notebook_extensions": {},  # remove the download notebook button
    "backreferences_dir": "_gallery/backreferences",
    "doc_module": ("masknmf",),
    "image_scrapers": ("pygfx",),
    "remove_config_comments": True,
    "subsection_order": ExplicitOrder(
        [
            "../../examples/compression",
            # "../../examples/demixing",
            # "../../examples/summary_images",
        ]
    ),
    "ignore_pattern": r'__init__\.py',
    "nested_sections": False,
    "thumbnail_size": (250, 250)
}

extra_conf = find_examples_for_gallery(EXAMPLES_DIR)
sphinx_gallery_conf.update(extra_conf)

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "show_version_warning_banner": True,
    "check_switcher": True,
    "switcher": {
        "json_url": "http://www.masknmf.org/_static/switcher.json",
        "version_match": release
    },
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/apasarkar/masknmf-toolbox",
            "icon": "fa-brands fa-github",
        }
    ]
}

html_static_path = ["_static"]
#html_logo = "_static/logo.png"
html_title = f"v{release}"

autodoc_member_order = "groupwise"
autoclass_content = "both"
add_module_names = False

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "fastplotlib": ("https://www.fastplotlib.org/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None,)
    # "masknmf": ("https://www.masknmf.org/", None),
}
