"""
Tests for the exposed rustfits.__version__.
"""

import re
from importlib.metadata import version

import rustfits


def test_version_is_nonempty_string():
    assert isinstance(rustfits.__version__, str)
    assert rustfits.__version__


def test_version_looks_like_a_release():
    # Baked in from Cargo.toml's [package].version at compile time.
    assert re.match(r"^\d+\.\d+\.\d+", rustfits.__version__)


def test_version_matches_distribution_metadata():
    # The compiled-in value (Cargo.toml) and the installed wheel/sdist
    # metadata (pyproject dynamic version, also Cargo.toml) must agree.
    assert rustfits.__version__ == version("rustfits")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
