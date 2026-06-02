"""
Tests for FITS.__contains__: ``key in fits`` membership tests.

Contract:
  - ``i in fits`` is True iff i (or len+i for negatives) is a valid
    HDU index.
  - ``"name" in fits`` is True iff some HDU has a matching EXTNAME
    (case-insensitive, whitespace-trimmed — same rule as
    ``fits["name"]``).
  - ``b"name" in fits`` works the same as the str form.
  - Bool keys are rejected explicitly (would otherwise satisfy
    isinstance(int)).
  - Non-int / non-str / non-bytes keys raise.
"""

import contextlib
import os
import tempfile

import pytest

import rustfits


@contextlib.contextmanager
def _three_hdus():
    """Primary (no extname) + two named extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "n.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=[2, 3])
            fits.create_image_hdu(dtype="f8", dims=[2, 3], extname="SCI")
            fits.create_image_hdu(dtype="i2", dims=[2, 3], extname="Weights")
        yield fname


# ---------------- int keys ----------------


def test_int_in_range():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert 0 in fits
            assert 1 in fits
            assert 2 in fits


def test_int_out_of_range_returns_false():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert 3 not in fits
            assert 99 not in fits


def test_negative_int_in_range():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert -1 in fits
            assert -2 in fits
            assert -3 in fits


def test_negative_int_out_of_range_returns_false():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert -4 not in fits
            assert -99 not in fits


# ---------------- str keys (EXTNAME) ----------------


def test_extname_present():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert "SCI" in fits
            assert "Weights" in fits


def test_extname_case_insensitive():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert "sci" in fits
            assert "WEIGHTS" in fits
            assert "weights" in fits


def test_extname_whitespace_trimmed():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert "  SCI  " in fits


def test_extname_missing_returns_false():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert "NOSUCH" not in fits
            # Primary HDU has no EXTNAME.
            assert "PRIMARY" not in fits


def test_extname_as_bytes():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert b"SCI" in fits
            assert b"nosuch" not in fits


def test_extname_bytes_non_ascii_rejected():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            with pytest.raises(ValueError, match="ASCII"):
                b"\xff" in fits  # noqa: B015 — exercising __contains__


# ---------------- rejection paths ----------------


def test_bool_key_rejected():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            with pytest.raises(ValueError, match="bool"):
                True in fits  # noqa: B015
            with pytest.raises(ValueError, match="bool"):
                False in fits  # noqa: B015


def test_unsupported_key_type_rejected():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            with pytest.raises(ValueError):
                [1, 2] in fits  # noqa: B015
            with pytest.raises(ValueError):
                1.5 in fits  # noqa: B015


# ---------------- contains + getitem consistency ----------------


def test_contains_then_getitem_never_raises():
    """If `k in fits` returns True, `fits[k]` must succeed."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            for k in [0, 1, 2, -1, -2, -3, "SCI", "weights", b"SCI"]:
                if k in fits:
                    fits[k]  # no exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
