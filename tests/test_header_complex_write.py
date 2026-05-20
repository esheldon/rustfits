"""
Phase 2c, step 4: complex value support in header writes.

FITS complex literal: `(real, imag)`.  Components are serialized via the
same float formatter as plain floats, so D/E exponents and special values
go through one shared code path.  No CONTINUE for complex (the entire
value lives on one card).

Each test verifies through BOTH the same FITS handle that did the mutation
AND a fresh reopen.
"""

import os
import tempfile
import contextlib

import pytest

import rustfits


@contextlib.contextmanager
def _new_file(shape=(4, 6), dtype="i4"):
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with rustfits.FITS(fname, "w+") as fits:
            fits.create_image_hdu(dtype=dtype, dims=list(shape))
        yield fname


def _check_both(fname, fits, predicate):
    predicate(fits[0].header)
    with rustfits.FITS(fname, "r") as fits2:
        predicate(fits2[0].header)


# -------- round-trip --------


def test_complex_value_round_trips():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["IMPED"] = complex(50.0, -25.0)

            def check(hd):
                v = hd["IMPED"]
                assert isinstance(v, complex)
                assert v == complex(50.0, -25.0)

            _check_both(fname, fits, check)


def test_complex_with_comment():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["IMPED"] = (complex(50.0, -25.0), "ohm")

            def check(hd):
                assert hd["IMPED"] == complex(50.0, -25.0)
                assert hd.comment_of("IMPED") == "ohm"

            _check_both(fname, fits, check)


def test_complex_integer_components():
    """
    Components are stored as floats per the FITS spec, so an integer
    complex round-trips as complex(3.0, 4.0)."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["CINT"] = complex(3, 4)

            def check(hd):
                v = hd["CINT"]
                assert isinstance(v, complex)
                assert v == complex(3.0, 4.0)

            _check_both(fname, fits, check)


def test_complex_with_negative_and_zero_parts():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["A"] = complex(-1.5, 0.0)
            fits[0].header["B"] = complex(0.0, -2.25)
            fits[0].header["C"] = complex(0.0, 0.0)

            def check(hd):
                assert hd["A"] == complex(-1.5, 0.0)
                assert hd["B"] == complex(0.0, -2.25)
                assert hd["C"] == complex(0.0, 0.0)

            _check_both(fname, fits, check)


def test_complex_with_scientific_notation():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["BIG"] = complex(1.5e10, -2.5e-3)

            def check(hd):
                v = hd["BIG"]
                assert v.real == pytest.approx(1.5e10)
                assert v.imag == pytest.approx(-2.5e-3)

            _check_both(fname, fits, check)


# -------- update / delete --------


def test_complex_update_preserves_position():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["IMPED"] = complex(50.0, -25.0)
            keys_before = list(fits[0].header)
            fits[0].header["IMPED"] = complex(75.0, 10.0)
            assert fits[0].header["IMPED"] == complex(75.0, 10.0)
            assert list(fits[0].header) == keys_before


def test_complex_bare_update_preserves_comment():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["IMPED"] = (complex(50.0, -25.0), "ohm")
            fits[0].header["IMPED"] = complex(75.0, 10.0)  # bare value

            def check(hd):
                assert hd["IMPED"] == complex(75.0, 10.0)
                assert hd.comment_of("IMPED") == "ohm"

            _check_both(fname, fits, check)


def test_complex_delete():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["IMPED"] = complex(50.0, -25.0)
            del fits[0].header["IMPED"]

            def check(hd):
                assert "IMPED" not in hd

            _check_both(fname, fits, check)


# -------- replacing across types --------


def test_complex_replaces_other_type_in_place():
    """Update an existing scalar key to a complex value — position kept."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["VAL"] = 1.5  # float first
            fits[0].header["VAL"] = complex(2.0, 3.0)

            def check(hd):
                assert hd["VAL"] == complex(2.0, 3.0)

            _check_both(fname, fits, check)


def test_other_type_replaces_complex_in_place():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["VAL"] = complex(2.0, 3.0)
            fits[0].header["VAL"] = 42  # back to int

            def check(hd):
                assert hd["VAL"] == 42
                assert isinstance(hd["VAL"], int)
                assert not isinstance(hd["VAL"], complex)

            _check_both(fname, fits, check)


# -------- HIERARCH + complex --------


def test_complex_with_hierarch_key():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO IMPED"] = complex(50.0, -25.0)

            def check(hd):
                v = hd["ESO IMPED"]
                assert isinstance(v, complex)
                assert v == complex(50.0, -25.0)
                assert any(
                    c.startswith("HIERARCH ESO IMPED") for c in hd.cards
                )

            _check_both(fname, fits, check)


def test_hierarch_complex_with_comment():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["ESO IMPED"] = (complex(50.0, -25.0), "ohm")

            def check(hd):
                assert hd["ESO IMPED"] == complex(50.0, -25.0)
                assert hd.comment_of("ESO IMPED") == "ohm"

            _check_both(fname, fits, check)


# -------- FITSHeaderEdit batching --------


def test_complex_in_edit_batch():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with fits[0].header.edit() as h:
                h["IMPED"] = complex(50.0, -25.0)
                h["GAIN"] = complex(2.0, 0.5)
                assert h["IMPED"] == complex(50.0, -25.0)
                assert h["GAIN"] == complex(2.0, 0.5)

            def check(hd):
                assert hd["IMPED"] == complex(50.0, -25.0)
                assert hd["GAIN"] == complex(2.0, 0.5)

            _check_both(fname, fits, check)


def test_complex_edit_rollback():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(RuntimeError):
                with fits[0].header.edit() as h:
                    h["IMPED"] = complex(50.0, -25.0)
                    raise RuntimeError("boom")

            def check(hd):
                assert "IMPED" not in hd

            _check_both(fname, fits, check)


# -------- update() with complex --------


def test_update_dict_with_complex_values():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.update({
                "A": complex(1.0, 2.0),
                "B": (complex(3.0, 4.0), "imp"),
                "C": 42,
            })

            def check(hd):
                assert hd["A"] == complex(1.0, 2.0)
                assert hd["B"] == complex(3.0, 4.0)
                assert hd.comment_of("B") == "imp"
                assert hd["C"] == 42

            _check_both(fname, fits, check)
