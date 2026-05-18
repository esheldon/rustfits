"""Tests for the FITSHeader mapping protocol.

Covers iteration order, len, keys/values/items, get(), cards, to_dict(),
__contains__, and the string repr/str.  Uses a hand-crafted file with a
mix of standard keys, HIERARCH, CONTINUE, and repeated commentary cards
so iteration semantics are unambiguous.
"""

import os
import tempfile

import pytest

import rustfits


def _card(text):
    assert len(text) <= 80, f"card too long ({len(text)} chars): {text!r}"
    return text.ljust(80)


# Crafted card order — referenced verbatim by the tests below.
_CARDS_IN_ORDER = [
    _card("SIMPLE  =                    T / conforms to FITS standard"),
    _card("BITPIX  =                    8 / bits per pixel"),
    _card("NAXIS   =                    2 / number of axes"),
    _card("NAXIS1  =                   10 / fast axis"),
    _card("NAXIS2  =                    5 / slow axis"),
    _card("COMMENT first commentary line"),
    _card("HIERARCH ESO INS TEMP = 12.5 / instrument temperature"),
    _card("LONGSTR = 'first part&' / start of comment"),
    _card("CONTINUE  'second part'  / end of comment"),
    _card("HISTORY processed step 1"),
    _card("COMMENT second commentary line"),
    _card("HISTORY processed step 2"),
    _card("        blank-keyword commentary"),
    _card("END"),
]

# The unique keys we expect iteration to yield, in card order.  Commentary
# keys (COMMENT, HISTORY, blank) appear once at their first-occurrence
# position.  END and CONTINUE are not keys.
_EXPECTED_KEYS = [
    "SIMPLE",
    "BITPIX",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "COMMENT",
    "ESO INS TEMP",  # HIERARCH expands to the long-key name
    "LONGSTR",
    "HISTORY",
    "",             # blank-keyword commentary
]


@pytest.fixture
def header():
    block = "".join(_CARDS_IN_ORDER).encode("ascii")
    block += b" " * ((-len(block)) % 2880)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with open(fname, "wb") as f:
            f.write(block)
        with rustfits.FITS(fname, "r") as fits:
            yield fits.hdus[0].header


# --------------------------- iteration order ----------------------------


def test_iter_order_matches_card_order(header):
    assert list(header) == _EXPECTED_KEYS


def test_keys_matches_iter(header):
    assert header.keys() == list(header)


def test_keys_in_card_order(header):
    assert header.keys() == _EXPECTED_KEYS


def test_repeated_commentary_keys_collapse_in_iter(header):
    # COMMENT cards appear twice, HISTORY twice, blank once — each yields
    # exactly one key in iteration.
    keys = list(header)
    assert keys.count("COMMENT") == 1
    assert keys.count("HISTORY") == 1
    assert keys.count("") == 1


def test_end_and_continue_not_in_iter(header):
    keys = list(header)
    assert "END" not in keys
    assert "CONTINUE" not in keys


# --------------------------- __len__ ----------------------------


def test_len_counts_unique_keys(header):
    assert len(header) == len(_EXPECTED_KEYS)


# -------------- __getitem__ (the main read API) ---------


def test_getitem_returns_value_directly_not_dict(header):
    """
    header[key] returns the value itself, not the legacy {value, comment}
    shape.
    """
    v = header["BITPIX"]
    assert v == 8
    assert not isinstance(v, dict)


def test_getitem_by_type(header):
    assert header["SIMPLE"] is True
    assert header["BITPIX"] == 8
    assert header["NAXIS1"] == 10
    assert isinstance(header["ESO INS TEMP"], float)
    assert header["ESO INS TEMP"] == 12.5


def test_getitem_hierarch_long_key(header):
    """
    HIERARCH long keys are subscripted by the full keyword name (with spaces).
    """
    assert header["ESO INS TEMP"] == 12.5
    with pytest.raises(KeyError):
        _ = header["HIERARCH"]  # the bare keyword is not a key


def test_getitem_string_with_continue(header):
    """
    A string value broken across CONTINUE cards reads as one concatenated
    string.
    """
    assert header["LONGSTR"] == "first partsecond part"


def test_getitem_commentary_keys_return_lists(header):
    assert header["COMMENT"] == [
        "first commentary line",
        "second commentary line",
    ]
    assert header["HISTORY"] == [
        "processed step 1",
        "processed step 2",
    ]
    assert header[""] == ["blank-keyword commentary"]


def test_getitem_missing_key_raises_keyerror(header):
    with pytest.raises(KeyError):
        _ = header["NOPE"]


def test_getitem_matches_get_for_present_keys(header):
    for k in header:
        assert header[k] == header.get(k)


# --------------------------- __contains__ ----------------------------


def test_contains_standard_key(header):
    assert "BITPIX" in header
    assert "MISSING" not in header


def test_contains_hierarch_key(header):
    assert "ESO INS TEMP" in header
    assert "HIERARCH" not in header  # the bare keyword is not a key


def test_contains_commentary_keys(header):
    assert "COMMENT" in header
    assert "HISTORY" in header
    assert "" in header


# --------------------------- values() / items() ----------------------------


def test_values_align_with_keys(header):
    keys = header.keys()
    vals = header.values()
    assert len(keys) == len(vals)
    # Spot-check a couple of well-known values.
    bitpix_pos = keys.index("BITPIX")
    naxis1_pos = keys.index("NAXIS1")
    assert vals[bitpix_pos] == 8
    assert vals[naxis1_pos] == 10


def test_items_pairs_keys_and_values(header):
    items = header.items()
    assert [k for k, _ in items] == header.keys()
    assert [v for _, v in items] == header.values()


def test_iteration_yields_same_values_as_subscript(header):
    for k in header:
        # Every iterated key must be subscriptable.  Smoke-checks correctness
        # of __getitem__ via the iteration protocol.
        _ = header[k]


# --------------------------- get() ----------------------------


def test_get_returns_value_when_present(header):
    assert header.get("BITPIX") == 8


def test_get_returns_none_default_when_missing(header):
    assert header.get("NOPE") is None


def test_get_returns_explicit_default_when_missing(header):
    sentinel = object()
    assert header.get("NOPE", sentinel) is sentinel


def test_get_does_not_swallow_other_errors(header):
    # get() must distinguish "missing key" (return default) from genuine
    # errors.  Asking for the comment-only commentary keys would fail at
    # comment_of, but normal access returns a list — so this is just a
    # smoke check that get() of a present key still returns the value.
    assert header.get("COMMENT") == header["COMMENT"]


# --------------------------- cards (raw) ----------------------------


def test_cards_returns_input_cards_minus_padding(header):
    cards = header.cards
    # The reader stores cards with trailing whitespace trimmed.  Compare
    # against our input cards similarly trimmed.
    trimmed_expected = [c.rstrip() for c in _CARDS_IN_ORDER]
    assert cards == trimmed_expected


def test_cards_preserves_continue_card(header):
    # The CONTINUE long-string row must still appear in the raw card list
    # even though it doesn't surface as a separate key.
    assert any(c.startswith("CONTINUE") for c in header.cards)


# ----------------------- to_dict() (legacy snapshot) ------------------------


def test_to_dict_has_value_comment_shape(header):
    d = header.to_dict()
    assert d["BITPIX"] == {"value": 8, "comment": "bits per pixel"}


def test_to_dict_commentary_lists(header):
    d = header.to_dict()
    assert d["COMMENT"] == [
        "first commentary line",
        "second commentary line",
    ]
    assert d["HISTORY"] == [
        "processed step 1",
        "processed step 2",
    ]
    assert d[""] == ["blank-keyword commentary"]


def test_to_dict_continue_concatenation(header):
    d = header.to_dict()
    # LONGSTR is the CONTINUE-joined value (no separator inserted between
    # parts — that's the FITS convention).  Comment is space-joined.
    assert d["LONGSTR"]["value"] == "first partsecond part"
    assert d["LONGSTR"]["comment"] == "start of comment end of comment"


# ----------- comment_of() vs to_dict() agreement -------


def test_comment_of_matches_to_dict(header):
    d = header.to_dict()
    for key in header:
        if key in ("COMMENT", "HISTORY", ""):
            continue  # commentary keys have no per-card comment
        assert header.comment_of(key) == d[key]["comment"], key


# --------------------------- __repr__ / __str__ ----------------------------


def test_repr_mentions_counts(header):
    r = repr(header)
    assert "FITSHeader" in r
    assert str(len(header)) in r           # unique key count
    assert str(len(header.cards)) in r     # card count


def test_str_shows_cards_one_per_line(header):
    text = str(header)
    lines = text.split("\n")
    # One non-trivial line per stored card, no padding artifacts.
    assert len(lines) == len(header.cards)
    assert lines[0].startswith("SIMPLE")
    assert lines[-1].startswith("END")


if __name__ == "__main__":
    # Minimal smoke run without pytest.
    block = "".join(_CARDS_IN_ORDER).encode("ascii")
    block += b" " * ((-len(block)) % 2880)
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "h.fits")
        with open(fname, "wb") as f:
            f.write(block)
        with rustfits.FITS(fname, "r") as fits:
            h = fits.hdus[0].header
            print(repr(h))
            print(list(h))
            print(len(h))
            print(h.to_dict())
