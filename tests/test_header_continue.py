import os
import tempfile

import rustfits


def _card(text):
    """Pad text to exactly 80 chars (one FITS card)."""
    assert len(text) <= 80, f"card too long ({len(text)} chars): {text!r}"
    return text.ljust(80)


def _write_fits_with_extended_header(fname):
    """Hand-craft a minimal FITS file exercising COMMENT, HISTORY, and CONTINUE.

    The primary HDU has no data (NAXIS=0), so the file is just a header block
    padded with spaces to 2880 bytes.
    """
    cards = [
        _card("SIMPLE  =                    T / conforms to FITS standard"),
        _card(
            "BITPIX  =                    8 / number of bits per data pixel"
        ),
        _card("NAXIS   =                    0 / number of data axes"),
        _card("LONGKEY = 'first part of the long string&' / start of comment"),
        _card("CONTINUE  'and the second part&' / middle"),
        _card("CONTINUE  'and the end.' / end of comment"),
        _card(
            "COMMENT FITS (Flexible Image Transport System) format is defined in"
        ),
        _card("COMMENT 'Astronomy and Astrophysics', volume 376, page 359."),
        _card("HISTORY Created 2026-05-16 for testing"),
        _card("HISTORY Calibrated with rustfits test suite"),
        _card("END"),
    ]
    header = "".join(cards).encode("ascii")
    header += b" " * ((-len(header)) % 2880)  # pad to a 2880-byte block

    with open(fname, "wb") as f:
        f.write(header)


def test_continue_long_string():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "ext.fits")
        _write_fits_with_extended_header(fname)

        with rustfits.FITS(fname, "r") as fits:
            hd = fits.hdus[0].header_dict

        assert hd["LONGKEY"]["value"] == (
            "first part of the long stringand the second partand the end."
        )
        assert (
            hd["LONGKEY"]["comment"]
            == "start of comment middle end of comment"
        )


def test_comment_cards_accumulated():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "ext.fits")
        _write_fits_with_extended_header(fname)

        with rustfits.FITS(fname, "r") as fits:
            hd = fits.hdus[0].header_dict

        assert isinstance(hd["COMMENT"], list)
        assert hd["COMMENT"] == [
            "FITS (Flexible Image Transport System) format is defined in",
            "'Astronomy and Astrophysics', volume 376, page 359.",
        ]


def test_history_cards_accumulated():
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "ext.fits")
        _write_fits_with_extended_header(fname)

        with rustfits.FITS(fname, "r") as fits:
            hd = fits.hdus[0].header_dict

        assert isinstance(hd["HISTORY"], list)
        assert hd["HISTORY"] == [
            "Created 2026-05-16 for testing",
            "Calibrated with rustfits test suite",
        ]


def test_continue_does_not_leak_keys():
    """The CONTINUE cards themselves must not appear as separate header keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "ext.fits")
        _write_fits_with_extended_header(fname)

        with rustfits.FITS(fname, "r") as fits:
            hd = fits.hdus[0].header_dict

        assert "CONTINUE" not in hd


if __name__ == "__main__":
    test_continue_long_string()
    test_comment_cards_accumulated()
    test_history_cards_accumulated()
    test_continue_does_not_leak_keys()
    print("all tests passed")
