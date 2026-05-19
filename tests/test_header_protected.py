"""Tests for protected-keyword handling.

Protected keywords are those rustfits manages on the user's behalf — file
structure (NAXIS family, BITPIX, SIMPLE, ...), integrity (CHECKSUM,
DATASUM), and tiled-compression layout (Z* family).  Direct mutation via
header[k] = v or del header[k] is rejected.  The full predicate is
exposed as rustfits.is_protected_key; to_dict(skip_protected=True)
returns the filtered subset for copying.
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


# ============================================================================
# is_protected_key predicate
# ============================================================================


@pytest.mark.parametrize("key", [
    # Image HDU structural
    "SIMPLE", "XTENSION", "EXTEND", "BITPIX", "NAXIS",
    "NAXIS1", "NAXIS2", "NAXIS999", "PCOUNT", "GCOUNT", "END",
    # Binary table structural
    "TFIELDS", "THEAP", "TFORM1", "TDIM3", "TTYPE1", "TSCAL2",
    "TZERO5", "TNULL1",
    # ASCII table structural
    "TBCOL1",
    # Random groups
    "GROUPS", "PTYPE1", "PSCAL1", "PZERO1",
    # Tiled image compression
    "ZIMAGE", "ZCMPTYPE", "ZBITPIX", "ZNAXIS", "ZNAXIS1", "ZTILE1",
    "ZNAME1", "ZVAL1", "ZSIMPLE", "ZEXTEND", "ZBLOCKED", "ZPCOUNT",
    "ZGCOUNT", "ZHECKSUM", "ZDATASUM", "ZTENSION", "ZQUANTIZ",
    "ZDITHER0", "ZMASKCMP", "ZBLANK",
    # Integrity
    "CHECKSUM", "DATASUM",
])
def test_is_protected_key_recognizes_protected(key):
    assert rustfits.is_protected_key(key) is True


@pytest.mark.parametrize("key", [
    # User metadata that's safe to write
    "OBJECT", "EXPTIME", "OBSERVER", "DATE-OBS", "TELESCOP",
    "EXTNAME", "EXTVER", "EXTLEVEL",
    "BUNIT", "BSCALE", "BZERO",   # image-only, not Tier-1 protected
    "CTYPE1", "CRVAL1", "CDELT1",
    # Indexed-family lookalikes that don't match (suffix isn't all digits)
    "NAXISA", "TFORM1A", "ZNAXISX",
    # Empty suffix doesn't count as indexed family
    "TFORM", "TDIM", "TTYPE", "TBCOL",
])
def test_is_protected_key_lets_user_keys_through(key):
    assert rustfits.is_protected_key(key) is False


def test_is_protected_key_is_case_insensitive():
    assert rustfits.is_protected_key("naxis1") is True
    assert rustfits.is_protected_key("Bitpix") is True
    assert rustfits.is_protected_key("checksum") is True


def test_is_protected_key_strips_whitespace():
    assert rustfits.is_protected_key("  NAXIS1  ") is True


# ============================================================================
# __setitem__ rejection
# ============================================================================


@pytest.mark.parametrize("key,value", [
    ("BITPIX", 32),
    ("NAXIS", 3),
    ("NAXIS1", 999),
    ("SIMPLE", True),
    ("CHECKSUM", "abcdefgh"),
    ("DATASUM", "12345"),
    ("ZIMAGE", True),
    ("TFIELDS", 5),
    ("TFORM3", "1J"),
])
def test_setitem_rejects_protected_key(key, value):
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="protected"):
                fits[0].header[key] = value


def test_setitem_rejection_is_case_insensitive():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="protected"):
                fits[0].header["naxis1"] = 7


def test_setitem_inside_edit_batch_rejects_protected_key():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="protected"):
                with fits[0].header.edit() as h:
                    h["BITPIX"] = 32


# ============================================================================
# __delitem__ rejection
# ============================================================================


@pytest.mark.parametrize("key", ["BITPIX", "NAXIS", "NAXIS1", "SIMPLE"])
def test_delitem_rejects_protected_key(key):
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="protected"):
                del fits[0].header[key]


def test_delitem_inside_edit_batch_rejects_protected_key():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="protected"):
                with fits[0].header.edit() as h:
                    del h["BITPIX"]


# ============================================================================
# Mutation rejection leaves state untouched
# ============================================================================


def test_failed_protected_setitem_leaves_state_unchanged():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            bitpix_before = fits[0].header["BITPIX"]
            cards_before = list(fits[0].header.cards)
            with pytest.raises(ValueError):
                fits[0].header["BITPIX"] = 999
            assert fits[0].header["BITPIX"] == bitpix_before
            assert fits[0].header.cards == cards_before
        # Disk also untouched.
        with rustfits.FITS(fname, "r") as fits:
            assert fits[0].header["BITPIX"] == bitpix_before


# ============================================================================
# to_dict(skip_protected=True)
# ============================================================================


def test_to_dict_default_includes_protected_keys():
    """Existing to_dict() behavior is unchanged when called without the flag."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            d = fits[0].header.to_dict()
            assert "BITPIX" in d
            assert "NAXIS" in d
            assert "NAXIS1" in d


def test_to_dict_skip_protected_drops_structural_keys():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["OBJECT"] = "M31"
            fits[0].header["EXPTIME"] = (5.0, "seconds")

            d = fits[0].header.to_dict(skip_protected=True)
            # Protected keys are gone.
            for k in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                      "EXTEND"):
                assert k not in d
            # User keys are preserved with their value+comment shape.
            assert d["OBJECT"]["value"] == "M31"
            assert d["EXPTIME"]["value"] == 5.0
            assert d["EXPTIME"]["comment"] == "seconds"


def test_to_dict_skip_protected_drops_continue_chain():
    """A protected key with a CONTINUE-chained value gets its whole chain
    removed (the orphan CONTINUE cards must NOT leak through)."""
    # We can't easily create a protected key with a real CONTINUE chain
    # from the public API (the writer would reject it).  Instead, write a
    # non-protected long-string key, then check that filtering an
    # unrelated protected key doesn't corrupt the chain.
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header["LONG"] = "X" * 200   # produces CONTINUE chain

            d = fits[0].header.to_dict(skip_protected=True)
            # LONG survives intact.
            assert d["LONG"]["value"] == "X" * 200
            # Protected keys are still removed.
            assert "BITPIX" not in d


def test_to_dict_skip_protected_keeps_commentary():
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            fits[0].header.add_comment("hello")
            fits[0].header.add_history("did a thing")

            d = fits[0].header.to_dict(skip_protected=True)
            assert d["COMMENT"] == ["hello"]
            assert d["HISTORY"] == ["did a thing"]
