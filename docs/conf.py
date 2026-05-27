"""Sphinx configuration for the rustfits documentation."""

# -- Project information -------------------------------------------------

project = "rustfits"
author = "Erin Sheldon"
copyright = "2026, Erin Sheldon"

# Pulled from the installed package so the docs and code can't drift.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("rustfits")
except Exception:
    release = "0.1.0"
version = release

# -- General Sphinx config -----------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # numpy-style docstring parser (no plugin install)
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# Napoleon: prefer numpy-style; we authored the docstrings that way.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# autodoc: show inherited members (so TableHDU's page lists header /
# index / extname / extver / has_data from the HDU base) and keep the
# source order (matches how the API is organized in the .rs files).
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Cross-link external projects so :class:`numpy.dtype` etc. resolve.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML theme ----------------------------------------------------------

html_theme = "furo"
html_title = f"rustfits {release}"
