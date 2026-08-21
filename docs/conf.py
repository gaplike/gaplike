"""Sphinx configuration for the gaplike documentation."""
import os
from importlib.metadata import version as _version

project = "gaplike"
author = "Federico Pozzoli, Ollie Burke"
copyright = "2026, Federico Pozzoli and Ollie Burke"

try:
    release = _version("gaplike")
except Exception:                      # docs built without the package installed
    release = "1.0.0"
version = ".".join(release.split(".")[:2])

# -- extensions -------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",             # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",   # writes .nojekyll for Pages
    "myst_parser",                     # pages written in Markdown
    "sphinx_copybutton",
]

# lisabeta is an optional dependency imported inside a factory; autodoc must
# never need it.  Anything else optional goes here too.
autodoc_mock_imports = ["lisabeta"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "emcee": ("https://emcee.readthedocs.io/en/stable", None),
}
intersphinx_timeout = 15
# Fetching the inventories needs the network; with -W a failed fetch would sink
# the build.  Set GAPLIKE_DOCS_OFFLINE=1 to build without them.
if os.environ.get("GAPLIKE_DOCS_OFFLINE"):
    intersphinx_mapping = {}

# -- general ----------------------------------------------------------------
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nitpicky = False

# -- HTML -------------------------------------------------------------------
html_theme = "furo"
html_title = f"gaplike {version}"
html_static_path = ["_static", "../assets"]   # repo-root media (the GIF)
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "sidebar_hide_name": True,          # the wordmark already says it
    "light_logo": "logo-wordmark.svg",
    "dark_logo": "logo-wordmark-dark.svg",
    "source_repository": "https://github.com/FedericoPozzoli/gaplike/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/FedericoPozzoli/gaplike",
            "class": "",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16">'
                '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38'
                ' 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28'
                '-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28'
                '-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02'
                '.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2'
                '-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65'
                ' 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0'
                ' 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
        },
    ],
}
