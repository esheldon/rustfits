"""
Image write/read with unaligned numpy input.

A numpy ndarray constructed via ``np.ndarray(..., buffer=..., offset=k,
strides=...)`` with `k` not a multiple of the dtype's itemsize is
"unaligned" — `arr.flags["ALIGNED"]` is False, and any C-side
multi-byte load via a typed pointer would be undefined behavior on
strict-alignment architectures.

rustfits's write fast path goes through ``RawBuffer::acquire``
(Python's buffer protocol), which must read the bytes via memcpy /
byte-at-a-time rather than typed pointer loads — otherwise unaligned
input would silently corrupt or segfault.

Regression pin against fitsio/tests/test_image.py
::test_image_write_read_unaligned.  Parametrized across the
integer + unsigned-int-trick + float dtype
matrix with explicit non-native byte orders (`<u4`, `>f4`); a NaN
variant on the float types exercises the same path with non-finite
values that would otherwise be silently truncated.

All 18 cases pass today — this file pins the behavior so a future
write-fast-path refactor doesn't accidentally introduce a typed
load on the source buffer.
"""

import os
import tempfile

import numpy as np
import pytest

import rustfits


_DTYPES = ["u1", "i1", "u2", "i2", "<u4", "i4", "i8", ">f4", "f8"]


def _unaligned_view(dtype):
    """
    Build a 19-element ndarray that views into a 20-element source
    buffer starting at byte offset 1.  For any dtype with itemsize
    > 1 this guarantees ``flags["ALIGNED"] == False`` on every
    architecture (byte-1 is not a multiple of 2, 4, 8, or 16).
    """
    source = np.arange(20, dtype=dtype)
    return source, np.ndarray(
        shape=(19,),
        dtype=source.dtype,
        buffer=source.data,
        offset=1,
        strides=source.strides,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("with_nan", [False, True])
def test_unaligned_image_round_trip(dtype, with_nan):
    if with_nan and "f" not in dtype:
        pytest.skip("NaN variant only meaningful on float dtypes")
    source, unaligned = _unaligned_view(dtype)
    if not dtype.endswith("1"):
        assert not unaligned.flags["ALIGNED"], (
            "test fixture is broken — view should be unaligned"
        )
    if with_nan:
        unaligned[3] = np.nan

    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.join(tmp, "t.fits")
        with rustfits.FITS(fname, "w+") as f:
            f.write_image(unaligned)
            # Same-handle read — sees what we just wrote.
            same_handle = f[-1].read()
        np.testing.assert_array_equal(same_handle, unaligned)

        # Post-reopen read — sees the on-disk bytes.
        with rustfits.FITS(fname, "r") as f:
            reread = f[-1].read()
        np.testing.assert_array_equal(reread, unaligned)
