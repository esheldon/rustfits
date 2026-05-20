"""
Tests for FITS.__getitem__ accepting str (EXTNAME) in addition to int.

Contract:
  - fits[i] with int (incl. negatives) still works as positional indexing.
  - fits["name"] returns the HDU whose EXTNAME matches, case-insensitively.
  - The primary HDU has no EXTNAME by default and is not findable by name.
  - Missing name raises ValueError.
  - Bool is rejected explicitly (would otherwise satisfy isinstance(int)).
"""

import os
import tempfile
import contextlib

import numpy as np
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


# ---------------- integer indexing still works ----------------


def test_int_indexing_unchanged():
    # FITS(fname) defaults to mode='r', matching the built-in open()
    # convention.  Most other tests still use the explicit "r" form.
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert fits[0].header["NAXIS1"] == 3
            assert fits[1].header["EXTNAME"] == "SCI"
            assert fits[2].header["EXTNAME"] == "Weights"


def test_negative_int_indexing_unchanged():
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert fits[-1].header["EXTNAME"] == "Weights"
            assert fits[-2].header["EXTNAME"] == "SCI"


def test_int_out_of_range_raises():
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="out of range"):
                fits[99]
            with pytest.raises(ValueError, match="out of range"):
                fits[-99]


# ---------------- string lookup ----------------


def test_string_lookup_exact_case():
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            hdu = fits["SCI"]
            assert hdu.header["EXTNAME"] == "SCI"


def test_string_lookup_case_insensitive():
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            assert fits["sci"].header["EXTNAME"] == "SCI"
            assert fits["Sci"].header["EXTNAME"] == "SCI"
            assert fits["weights"].header["EXTNAME"] == "Weights"
            assert fits["WEIGHTS"].header["EXTNAME"] == "Weights"
            assert fits["WeIgHtS"].header["EXTNAME"] == "Weights"


def test_string_lookup_returns_same_object_as_int():
    """fits[1] and fits[its-extname] should return the same underlying HDU."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            by_int = fits[1]
            by_name = fits["SCI"]
            by_int.header["MARKER"] = "abc"
            assert by_name.header["MARKER"] == "abc"


def test_string_lookup_missing_raises():
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="no HDU named"):
                fits["nope"]


def test_string_lookup_primary_without_extname_not_findable():
    """
    The primary HDU created without extname has no EXTNAME card and is
    not addressable by name."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="no HDU named"):
                fits["PRIMARY"]


def test_string_lookup_supports_extend_on_named_hdu():
    """Round-trip: a string-keyed HDU works for mutations like extend."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            new = np.zeros((1, 3), dtype="f8") + 7.0
            fits["SCI"].extend(new)
            assert fits["SCI"].header["NAXIS2"] == 3   # 2 + 1
        with rustfits.FITS(fname, "r") as fits:
            assert fits["sci"].header["NAXIS2"] == 3


# ---------------- string-like types: bytes and numpy scalars ----------------


def test_bytes_lookup_works():
    """
    Plain bytes objects are accepted; FITS spec restricts EXTNAME to
    printable ASCII so this is well-defined."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            assert fits[b"SCI"].header["EXTNAME"] == "SCI"
            assert fits[b"sci"].header["EXTNAME"] == "SCI"
            assert fits[b"WEIGHTS"].header["EXTNAME"] == "Weights"


def test_numpy_str_lookup_works():
    """np.str_ (U dtype) is a str subclass — accepted directly."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            u_name = np.array(["SCI"])[0]
            assert isinstance(u_name, str)
            assert fits[u_name].header["EXTNAME"] == "SCI"


def test_numpy_bytes_lookup_works():
    """
    np.bytes_ (S dtype) is a bytes subclass — accepted via the bytes
    fallback."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            s_name = np.array([b"SCI"])[0]
            assert isinstance(s_name, bytes)
            assert fits[s_name].header["EXTNAME"] == "SCI"
            # case-insensitive
            assert fits[np.bytes_(b"sci")].header["EXTNAME"] == "SCI"


def test_non_ascii_bytes_rejected():
    """FITS EXTNAME is printable ASCII; non-ASCII bytes can't match."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="ASCII"):
                fits[b"\xff\xfeSCI"]


# ---------------- error paths ----------------


def test_bool_rejected_explicitly():
    """
    Without the explicit bool check, Python's bool-is-int relationship
    would silently make fits[True] mean fits[1]."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises(ValueError, match="bool"):
                fits[True]
            with pytest.raises(ValueError, match="bool"):
                fits[False]


def test_other_key_types_rejected():
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises((ValueError, TypeError)):
                fits[1.5]
            with pytest.raises((ValueError, TypeError)):
                fits[(0, 1)]


def test_list_of_small_ints_not_treated_as_bytes_extname():
    """
    Regression: PyO3's extract::<Vec<u8>>() accepts any iterable of
    small ints, so without explicit PyBytes type-instance checking,
    `fits[[5, 0, 2]]` would silently route to an EXTNAME lookup of
    a 3-byte control-char string.  Must reject as an unknown key
    type."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises((ValueError, TypeError)):
                fits[[5, 0, 2]]


def test_numpy_int_array_not_treated_as_bytes_extname():
    """
    Same regression — numpy arrays of small ints are iterable and
    each element is u8-extractable."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname, "r") as fits:
            with pytest.raises((ValueError, TypeError)):
                fits[np.array([0, 1], dtype=np.uint8)]


# ---------------- edge case: duplicate EXTNAME (first wins) ----------------


def test_duplicate_extname_returns_first():
    """If two HDUs share an EXTNAME, lookup returns the first match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "d.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="i4", dims=[2, 3])
            fits.create_image_hdu(dtype="i4", dims=[2, 3], extname="DUP")
            fits.create_image_hdu(dtype="f8", dims=[4, 4], extname="DUP")

        with rustfits.FITS(fname, "r") as fits:
            hdu = fits["DUP"]
            # First DUP is the i4 image at index 1.
            assert hdu.header["BITPIX"] == 32
            assert hdu.header["NAXIS1"] == 3


# ---------------- iteration ----------------


def test_iter_yields_all_hdus_in_order():
    """
    `for hdu in fits` walks the HDUs in file order, matching
    `for hdu in fits.hdus`."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            via_iter = list(fits)
            via_hdus = list(fits.hdus)
            assert len(via_iter) == 3
            assert via_iter == via_hdus


def test_iter_single_hdu_file():
    """File with only a primary HDU iterates once."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "p.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype="f4", dims=[2, 2])
        with rustfits.FITS(fname) as fits:
            n = 0
            for hdu in fits:
                n += 1
                assert hdu is fits[0]
            assert n == 1


def test_iter_len_matches():
    """len(fits) == number of HDUs yielded by iteration."""
    with _three_hdus() as fname:
        with rustfits.FITS(fname) as fits:
            assert len(fits) == sum(1 for _ in fits)


if __name__ == "__main__":
    test_int_indexing_unchanged()
    test_negative_int_indexing_unchanged()
    test_int_out_of_range_raises()
    test_string_lookup_exact_case()
    test_string_lookup_case_insensitive()
    test_string_lookup_returns_same_object_as_int()
    test_string_lookup_missing_raises()
    test_string_lookup_primary_without_extname_not_findable()
    test_string_lookup_supports_extend_on_named_hdu()
    test_bytes_lookup_works()
    test_numpy_str_lookup_works()
    test_numpy_bytes_lookup_works()
    test_non_ascii_bytes_rejected()
    test_bool_rejected_explicitly()
    test_other_key_types_rejected()
    test_list_of_small_ints_not_treated_as_bytes_extname()
    test_numpy_int_array_not_treated_as_bytes_extname()
    test_duplicate_extname_returns_first()
    test_iter_yields_all_hdus_in_order()
    test_iter_single_hdu_file()
    test_iter_len_matches()
    print("all tests passed")
