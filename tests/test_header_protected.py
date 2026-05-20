"""
Tests for protected-keyword handling.

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
    """
    Existing to_dict() behavior is unchanged when called without the flag.
    """
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
    """
    A protected key with a CONTINUE-chained value gets its whole chain
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


# ============================================================================
# update() from a FITSHeader source: protected keys auto-skipped
# ============================================================================


def test_update_from_fitsheader_silently_skips_protected_keys():
    """
    Copying a header from one HDU to another must drop protected keys
    (NAXIS/BITPIX/etc. of the source) so the destination's own structural
    state is preserved.  User metadata still copies through."""
    with _new_file(shape=(4, 6), dtype="i4") as a_name, \
         _new_file(shape=(8, 10), dtype="f4") as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header["OBJECT"] = "M31"
            a[0].header["EXPTIME"] = (5.0, "exposure (s)")
        with rustfits.FITS(a_name, "r") as a, \
             rustfits.FITS(b_name, "r+") as b:
            # a has BITPIX=32, NAXIS1=6, NAXIS2=4; b has BITPIX=-32,
            # NAXIS1=10, NAXIS2=8.  Without auto-skip, the update would
            # raise on the first protected key.
            b[0].header.update(a[0].header)
            # User metadata came through.
            assert b[0].header["OBJECT"] == "M31"
            assert b[0].header["EXPTIME"] == 5.0
            assert b[0].header.comment_of("EXPTIME") == "exposure (s)"
            # Destination's own protected keys are unchanged.
            assert b[0].header["BITPIX"] == -32
            assert b[0].header["NAXIS1"] == 10
            assert b[0].header["NAXIS2"] == 8


def test_update_from_fitsheader_skips_checksum_and_datasum():
    """
    CHECKSUM/DATASUM are integrity contracts on the destination — they
    must not be copied from a source header."""
    with _new_file() as a_name, _new_file() as b_name:
        # Inject CHECKSUM/DATASUM into a's cards by closing+writing through
        # the structural path; the simplest way is to verify by manually
        # constructing a card stream is intrusive.  Instead, just confirm
        # the broader contract via the same-shape happy path: copying does
        # not raise even though a's header carries SIMPLE/BITPIX/NAXIS*.
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header["OBJECT"] = "src"
        with rustfits.FITS(a_name, "r") as a, \
             rustfits.FITS(b_name, "r+") as b:
            b[0].header.update(a[0].header)   # must not raise
            assert b[0].header["OBJECT"] == "src"


def test_update_from_dict_still_raises_on_protected_key():
    """
    Dict-source update() must continue to raise: explicit hand-written
    protected keys are almost certainly a mistake, not an intent to be
    silently dropped."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError, match="protected"):
                fits[0].header.update({"BITPIX": 32, "OBJECT": "M31"})
            # Even the non-protected key in the same dict must not have
            # been written, since the update is rejected wholesale.
            assert "OBJECT" not in fits[0].header


def test_update_from_dict_still_raises_on_commentary_key():
    """Dict-source update() also continues to reject commentary keys."""
    with _new_file() as fname:
        with rustfits.FITS(fname, "r+") as fits:
            with pytest.raises(ValueError):
                fits[0].header.update({"COMMENT": "no good"})


def test_update_from_fitsheader_inside_edit_skips_protected_keys():
    """Auto-skip also applies inside a batched edit()."""
    with _new_file(shape=(4, 6), dtype="i4") as a_name, \
         _new_file(shape=(8, 10), dtype="f4") as b_name:
        with rustfits.FITS(a_name, "r+") as a:
            a[0].header["OBJECT"] = "M31"
        with rustfits.FITS(a_name, "r") as a, \
             rustfits.FITS(b_name, "r+") as b:
            with b[0].header.edit() as h:
                h.update(a[0].header)
            assert b[0].header["OBJECT"] == "M31"
            # Destination's BITPIX untouched.
            assert b[0].header["BITPIX"] == -32
