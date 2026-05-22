#!/usr/bin/env bash
# Wrap `cargo test` so the test binary can find libpython at runtime.
#
# The rustfits crate uses PyO3, so the test binary built by
# `cargo test` links against libpython.X.so and needs to resolve it
# at load time.  Conda env activation doesn't put $CONDA_PREFIX/lib
# on LD_LIBRARY_PATH, so a bare `cargo test` from an activated env
# fails with "libpython3.X.so.1.0: cannot open shared object file".
#
# This script asks the active Python interpreter where its lib
# directory is (sysconfig.LIBDIR) and prepends it to
# LD_LIBRARY_PATH for the cargo test invocation.  Works in any
# environment where `python` is on PATH and points at the
# interpreter you'd build the extension against — conda envs,
# venvs, system python, CI runners.
#
# All arguments are forwarded to cargo test (e.g.
#   tools/cargo-test.sh --lib --no-default-features).

set -euo pipefail

LIBDIR=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
if [ -z "$LIBDIR" ] || [ ! -d "$LIBDIR" ]; then
    echo "tools/cargo-test.sh: could not find Python lib dir via sysconfig" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"
exec cargo test "$@"
