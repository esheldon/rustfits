"""Tests for column-units (TUNITn) on TableHDU and image-units
(BUNIT) on ImageHDU.

Units are purely informational — they don't affect read/write.  Two
surfaces: a dict / string accessor, plus inclusion in the __repr__.
"""

import os
import struct
import sys
import tempfile

import pytest

import rustfits


CARDS_PER_BLOCK = 36
BLOCK = 2880


def _show(r):
    """Print the repr the test is asserting on, with a separator
    and leading newline so pytest -s -v output is readable."""
    print()
    print("-" * 70)
    print(r, end="")
    sys.stdout.flush()


def _pad_cards(cards):
    blocks = [c.ljust(80) for c in cards]
    while len(blocks) % CARDS_PER_BLOCK != 0:
        blocks.append(" " * 80)
    return "".join(blocks).encode("ascii")


def _pad_to_block(b):
    return b + b"\x00" * ((BLOCK - len(b) % BLOCK) % BLOCK)


def _write_file(path, *parts):
    with open(path, "wb") as f:
        for cards, data in parts:
            f.write(_pad_cards(cards))
            if data:
                f.write(_pad_to_block(data))


def _primary_no_data(extras=()):
    return [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        *extras,
        "END",
    ]


def _image_primary(bitpix, dims, extras=()):
    cards = [
        "SIMPLE  =                    T",
        f"BITPIX  = {bitpix:>20d}",
        f"NAXIS   = {len(dims):>20d}",
    ]
    for i, d in enumerate(dims, start=1):
        cards.append(f"NAXIS{i:<3d}= {d:>20d}")
    cards.append("EXTEND  =                    T")
    cards.extend(extras)
    cards.append("END")
    return cards


def _bintable_ext(naxis1, naxis2, fields, extras=()):
    cards = [
        "XTENSION= 'BINTABLE'",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        f"NAXIS1  = {naxis1:>20d}",
        f"NAXIS2  = {naxis2:>20d}",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        f"TFIELDS = {len(fields):>20d}",
    ]
    for i, (ttype, tform, *opt) in enumerate(fields, start=1):
        cards.append(f"TTYPE{i:<3d}= '{ttype:<8s}'")
        cards.append(f"TFORM{i:<3d}= '{tform:<8s}'")
        if opt:
            cards.append(f"TDIM{i:<4d}= '{opt[0]:<8s}'")
    cards.extend(extras)
    cards.append("END")
    return cards


def _tunit_card(col_index, value):
    return f"TUNIT{col_index:<3d}= '{value:<8s}'"


# ---------------------------------------------------------------------------
# TableHDU.units accessor
# ---------------------------------------------------------------------------


def test_units_accessor_with_tunit():
    """TUNIT1 set → units dict carries the value; the unaffected
    column maps to None."""
    fields = [("flux", "1E"), ("name", "8A")]
    extras = [_tunit_card(1, "Jy")]
    cards = _bintable_ext(4 + 8, 1, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, bytes(12)))
        with rustfits.FITS(fname) as fits:
            u = fits[1].units
            assert u == {"flux": "Jy", "name": None}


def test_units_accessor_no_tunit_at_all():
    """No TUNITn anywhere → every entry is None."""
    fields = [("a", "1J"), ("b", "1D")]
    cards = _bintable_ext(12, 1, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, bytes(12)))
        with rustfits.FITS(fname) as fits:
            assert fits[1].units == {"a": None, "b": None}


def test_units_accessor_preserves_column_order():
    """The dict reflects the on-disk column order, not alphabetical."""
    fields = [("z", "1J"), ("a", "1J"), ("m", "1J")]
    extras = [
        _tunit_card(1, "s"),
        _tunit_card(2, "m"),
        _tunit_card(3, "kg"),
    ]
    cards = _bintable_ext(12, 1, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, bytes(12)))
        with rustfits.FITS(fname) as fits:
            u = fits[1].units
            assert list(u.keys()) == ["z", "a", "m"]


def test_units_accessor_preserves_case():
    """Unit strings keep their case verbatim — `kJy` stays `kJy`."""
    fields = [("flux", "1E")]
    extras = [_tunit_card(1, "kJy")]
    cards = _bintable_ext(4, 1, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, bytes(4)))
        with rustfits.FITS(fname) as fits:
            assert fits[1].units["flux"] == "kJy"


# ---------------------------------------------------------------------------
# TableHDU repr — unit shows up after dtype/shape
# ---------------------------------------------------------------------------


def test_repr_shows_unit_on_scalar_column():
    """`(Jy)` appears after the dtype on a scalar column with TUNIT."""
    fields = [("flux", "1E")]
    extras = [_tunit_card(1, "Jy")]
    cards = _bintable_ext(4, 1, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, bytes(4)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            flux_line = [
                ln for ln in r.splitlines() if "flux" in ln
            ][0]
            assert "f4" in flux_line
            assert "(Jy)" in flux_line


def test_repr_shows_unit_after_shape():
    """For an array column the order is `dtype  array[...]  (unit)`."""
    fields = [("pos", "3D")]
    extras = [_tunit_card(1, "deg")]
    cards = _bintable_ext(24, 1, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, bytes(24)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            pos_line = [
                ln for ln in r.splitlines() if "pos" in ln
            ][0]
            # Both pieces appear, in the right order.
            assert "array[3]" in pos_line
            assert "(deg)" in pos_line
            assert pos_line.index("array[3]") < pos_line.index("(deg)")


def test_repr_no_unit_no_parens():
    """A column without TUNIT shows no `(...)` parens."""
    fields = [("count", "1J")]
    cards = _bintable_ext(4, 1, fields)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, bytes(4)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            count_line = [
                ln for ln in r.splitlines() if "count" in ln
            ][0]
            assert "(" not in count_line
            assert ")" not in count_line


def test_repr_mixed_units_and_no_units():
    """Some columns have units, others don't — only the ones with
    TUNIT show the parens block."""
    fields = [("flux", "1E"), ("name", "8A"), ("count", "1J")]
    extras = [
        _tunit_card(1, "Jy"),
        # column 2 (name) intentionally has no TUNIT
        _tunit_card(3, "ct"),
    ]
    cards = _bintable_ext(4 + 8 + 4, 1, fields, extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        _write_file(fname, (_primary_no_data(), b""), (cards, bytes(16)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[1])
            _show(r)
            lines = r.splitlines()
            flux_line = [ln for ln in lines if "flux" in ln][0]
            name_line = [ln for ln in lines if "name" in ln][0]
            count_line = [ln for ln in lines if "count" in ln][0]
            assert "(Jy)" in flux_line
            assert "(" not in name_line
            assert "(ct)" in count_line


# ---------------------------------------------------------------------------
# ImageHDU.unit accessor + repr
# ---------------------------------------------------------------------------


def test_image_unit_accessor_with_bunit():
    """BUNIT set → image_hdu.unit returns the string."""
    extras = ["BUNIT   = 'counts/s'"]
    primary = _image_primary(-64, [3, 2], extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "i.fits")
        _write_file(fname, (primary, bytes(3 * 2 * 8)))
        with rustfits.FITS(fname) as fits:
            assert fits[0].unit == "counts/s"


def test_image_unit_accessor_no_bunit():
    """No BUNIT → image_hdu.unit returns None."""
    primary = _image_primary(-64, [3, 2])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "i.fits")
        _write_file(fname, (primary, bytes(3 * 2 * 8)))
        with rustfits.FITS(fname) as fits:
            assert fits[0].unit is None


def test_image_repr_shows_unit():
    """ImageHDU repr includes the BUNIT value when set."""
    extras = ["BUNIT   = 'Jy      '"]
    primary = _image_primary(-32, [10], extras=extras)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "i.fits")
        _write_file(fname, (primary, bytes(10 * 4)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[0])
            _show(r)
            assert "unit: Jy" in r


def test_image_repr_no_unit_line_when_unset():
    """When BUNIT is absent the repr omits the `unit:` line."""
    primary = _image_primary(-32, [10])
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "i.fits")
        _write_file(fname, (primary, bytes(10 * 4)))
        with rustfits.FITS(fname) as fits:
            r = repr(fits[0])
            _show(r)
            assert "unit:" not in r


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-v"]))
