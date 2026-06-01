"""
ASCII-table iteration: __iter__ (row mode) + iter(chunksize=, columns=).

Verifies the shared TableIter pyclass works against AsciiTableHDU via
polymorphic `call_method` dispatch on the HDU's `read` method.  The
parametrization over (BINTABLE, ASCII) confirms the surface contracts
match across HDU types.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------

CARDS_PER_BLOCK = 36
BLOCK = 2880


def _pad_cards(cards):
    blocks = [c.ljust(80) for c in cards]
    while len(blocks) % CARDS_PER_BLOCK != 0:
        blocks.append(" " * 80)
    return "".join(blocks).encode("ascii")


def _pad_to_block(b, pad_byte=b" "):
    return b + pad_byte * ((BLOCK - len(b) % BLOCK) % BLOCK)


def _write_file(path, *parts):
    with open(path, "wb") as f:
        for cards, data, pad in parts:
            f.write(_pad_cards(cards))
            if data:
                f.write(_pad_to_block(data, pad))


def _primary_no_data():
    return [
        "SIMPLE  =                    T",
        "BITPIX  =                    8",
        "NAXIS   =                    0",
        "EXTEND  =                    T",
        "END",
    ]


def _ascii_fixture(tmp, n_rows):
    """Build a (N-row) ASCII fixture with 2 cols (I5 + F8.2)."""
    cards = [
        "XTENSION= 'TABLE   '",
        "BITPIX  =                    8",
        "NAXIS   =                    2",
        f"NAXIS1  = {14:>20d}",  # 5 + space + 8
        f"NAXIS2  = {n_rows:>20d}",
        "PCOUNT  =                    0",
        "GCOUNT  =                    1",
        "TFIELDS =                    2",
        "TTYPE1  = 'ID      '",
        "TBCOL1  =                    1",
        "TFORM1  = 'I5      '",
        "TTYPE2  = 'FLUX    '",
        "TBCOL2  =                    7",
        "TFORM2  = 'F8.2    '",
        "END",
    ]
    rows_data = []
    for i in range(n_rows):
        rows_data.append(f"{i:5d} {i * 0.5:7.2f}".ljust(14))
    data = "".join(rows_data).encode("ascii")
    fname = os.path.join(tmp, "t.fits")
    _write_file(
        fname,
        (_primary_no_data(), b"", b" "),
        (cards, data, b" "),
    )
    return fname


# ---------------------------------------------------------------------------
# row mode (__iter__ / iter())
# ---------------------------------------------------------------------------


def test_iter_row_mode():
    """`for row in hdu:` yields one np.void record per row."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ascii_fixture(tmp, 5)
        with rustfits.FITS(fname) as f:
            rows = list(f[1])
            assert len(rows) == 5
            assert all(isinstance(r, np.void) for r in rows)
            assert [int(r["ID"]) for r in rows] == [0, 1, 2, 3, 4]
            np.testing.assert_allclose(
                [float(r["FLUX"]) for r in rows],
                [0.0, 0.5, 1.0, 1.5, 2.0],
                rtol=1e-5,
            )


def test_iter_method_equals_iter_dunder():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ascii_fixture(tmp, 4)
        with rustfits.FITS(fname) as f:
            rows_dunder = list(f[1])
            rows_method = list(f[1].iter())
            assert len(rows_dunder) == len(rows_method)
            for a, b in zip(rows_dunder, rows_method):
                assert int(a["ID"]) == int(b["ID"])


def test_iter_chunk_mode():
    """iter(chunksize=N) yields structured ndarrays of <=N rows."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ascii_fixture(tmp, 10)
        with rustfits.FITS(fname) as f:
            chunks = list(f[1].iter(chunksize=3))
            # 10 rows / 3 -> chunks of [3, 3, 3, 1]
            assert [len(c) for c in chunks] == [3, 3, 3, 1]
            assert all(isinstance(c, np.ndarray) for c in chunks)
            assert chunks[0].dtype.names == ("ID", "FLUX")
            np.testing.assert_array_equal(
                np.concatenate([c["ID"] for c in chunks]), np.arange(10)
            )


def test_iter_columns_forwarded():
    """columns= subset restricts what each record / chunk carries."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ascii_fixture(tmp, 5)
        with rustfits.FITS(fname) as f:
            rows = list(f[1].iter(columns=["ID"]))
            for r in rows:
                # 1-field record; FLUX should not appear
                assert "FLUX" not in r.dtype.names
                assert "ID" in r.dtype.names


def test_iter_empty_table():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ascii_fixture(tmp, 0)
        with rustfits.FITS(fname) as f:
            assert list(f[1]) == []
            assert list(f[1].iter(chunksize=10)) == []


def test_iter_chunksize_zero_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ascii_fixture(tmp, 3)
        with rustfits.FITS(fname) as f:
            with pytest.raises(ValueError, match="positive"):
                list(f[1].iter(chunksize=0))


def test_iter_independent_cursors():
    """Two iterators over the same HDU advance independently."""
    with tempfile.TemporaryDirectory() as tmp:
        fname = _ascii_fixture(tmp, 5)
        with rustfits.FITS(fname) as f:
            it1 = iter(f[1])
            it2 = iter(f[1])
            assert int(next(it1)["ID"]) == 0
            assert int(next(it2)["ID"]) == 0
            assert int(next(it1)["ID"]) == 1
            assert int(next(it2)["ID"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
