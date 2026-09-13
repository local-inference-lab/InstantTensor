#!/usr/bin/env bash
# Build a source-addressed InstantTensor wheel for the declared foundation.
set -euo pipefail

mkdir -p /wheelhouse
env -u PYTHONPATH uv build \
  --wheel \
  --no-build-isolation \
  --python /build-venv/bin/python \
  --out-dir /wheelhouse \
  /src/instanttensor

wheel=$(find /wheelhouse -maxdepth 1 -name 'instanttensor-*.whl' -print -quit)
test -n "${wheel}"
/build-venv/bin/python ci/lil_wheels/normalize_wheel.py \
  --wheel "${wheel}" \
  --version "0.1.9+lil.cu134.g${SOURCE_COMMIT:0:12}" \
  --torch-version 2.14.0a0+4fdf77b940.nv26.8.63802676 \
  --source-date-epoch "${SOURCE_DATE_EPOCH:?}"
test "$(find /wheelhouse -maxdepth 1 -name 'instanttensor-*.whl' | wc -l)" -eq 1
