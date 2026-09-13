#!/usr/bin/env python3
"""Give an InstantTensor wheel a source-addressed version and Torch ABI pin."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def rewrite_metadata(metadata: bytes, *, version: str, torch_version: str) -> bytes:
    """Return metadata with a source-addressed version and exact Torch pin."""
    message = BytesParser(policy=compat32).parsebytes(metadata)
    requirements = message.get_all("Requires-Dist", [])
    rewritten: list[str] = []
    found_torch = False
    for value in requirements:
        requirement = Requirement(value)
        if canonicalize_name(requirement.name) == "torch":
            rewritten.append(f"torch=={torch_version}")
            found_torch = True
        else:
            rewritten.append(value)
    if not found_torch:
        raise ValueError("InstantTensor dependency contract changed: torch is missing")
    del message["Version"]
    message["Version"] = version
    del message["Requires-Dist"]
    for requirement in rewritten:
        message["Requires-Dist"] = requirement
    message["X-Local-Inference-Runtime"] = "jovian-cu134-torch214-cxx11"
    return message.as_bytes(policy=compat32.clone(max_line_length=0))


def normalize_wheel(
    wheel: Path,
    *,
    version: str,
    torch_version: str,
    source_date_epoch: int,
) -> Path:
    """Normalize one wheel and return its renamed output path."""
    with tempfile.TemporaryDirectory(prefix="instanttensor-wheel-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        dist_info_dirs = list(root.glob("instanttensor-*.dist-info"))
        if len(dist_info_dirs) != 1:
            raise ValueError("wheel must contain exactly one InstantTensor dist-info")
        dist_info = dist_info_dirs[0]
        metadata = dist_info / "METADATA"
        metadata.write_bytes(
            rewrite_metadata(
                metadata.read_bytes(), version=version, torch_version=torch_version
            )
        )
        renamed_dist_info = root / f"instanttensor-{version}.dist-info"
        dist_info.rename(renamed_dist_info)
        for path in sorted(root.rglob("*"), reverse=True):
            if not path.is_symlink():
                os.utime(path, (source_date_epoch, source_date_epoch))
        os.utime(root, (source_date_epoch, source_date_epoch))
        output_dir = wheel.parent / "normalized"
        output_dir.mkdir()
        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
        subprocess.run(
            [
                "/build-venv/bin/python",
                "-m",
                "wheel",
                "pack",
                "--dest-dir",
                str(output_dir),
                str(root),
            ],
            check=True,
            env=environment,
        )
        produced = list(output_dir.glob("instanttensor-*.whl"))
        if len(produced) != 1:
            raise ValueError("wheel pack did not produce exactly one wheel")
        destination = wheel.parent / produced[0].name
        shutil.move(produced[0], destination)
    wheel.unlink()
    return destination


def main() -> None:
    """Parse the runtime contract and normalize one built wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    output = normalize_wheel(
        args.wheel,
        version=args.version,
        torch_version=args.torch_version,
        source_date_epoch=args.source_date_epoch,
    )
    print(f"instanttensor_wheel={output}")
