# flake8: noqa
from ._rust import (
    FITS,
    HDU,
    ImageHDU,
    CompressedImageHDU,
    TableHDU,
    AsciiTableHDU,
    FITSHeader,
    FITSHeaderEdit,
    Gzip1,
    Gzip2,
    Hcompress1,
    Rice1,
    is_protected_key,
)

# User-facing convenience wrappers (read, future read_header, write, ...).
# Definitions live in convenience.py; we re-export at the top level so
# users can write `rustfits.read(...)` directly.
from .convenience import read
