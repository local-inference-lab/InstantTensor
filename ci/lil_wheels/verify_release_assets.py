#!/usr/bin/env python3
"""Verify immutable InstantTensor wheel release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Verify source identity, tag identity, and the wheel digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--beta-tag", required=True)
    parser.add_argument("--promotion", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    if manifest["schema"] != "local-inference-instanttensor-wheel-release/v1":
        raise ValueError("release schema mismatch")
    if manifest["source"]["commit"] != args.source_commit:
        raise ValueError("source commit mismatch")
    if manifest["release_tag"] != args.beta_tag:
        raise ValueError("beta tag mismatch")
    for package in manifest["packages"]:
        wheel = args.directory / package["file"]
        if not wheel.is_file():
            wheel = args.directory / "wheels" / package["file"]
        if not wheel.is_file() or sha256(wheel) != package["sha256"]:
            raise ValueError(f"wheel digest mismatch: {package['file']}")
    if args.promotion and not (args.directory / "stable-promotion.json").is_file():
        raise ValueError("stable promotion record missing")
    print("InstantTensor release assets: PASS")


if __name__ == "__main__":
    main()
