"""Tests for InstantTensor wheel metadata normalization."""

from email.parser import BytesParser
from email.policy import compat32

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ci.lil_wheels.normalize_wheel import rewrite_metadata


def test_rewrite_metadata_addresses_source_and_pins_torch() -> None:
    metadata = (
        b"Metadata-Version: 2.4\n"
        b"Name: instanttensor\n"
        b"Version: 0.1.9\n"
        b"Requires-Dist: torch>=2.8.0\n\n"
    )
    rewritten = rewrite_metadata(
        metadata,
        version="0.1.9+lil.cu134.gabc",
        torch_version="2.14.0a0+nv",
    )
    message = BytesParser(policy=compat32).parsebytes(rewritten)
    requirements = {
        canonicalize_name(req.name): str(req)
        for req in map(Requirement, message.get_all("Requires-Dist", []))
    }
    assert message["Version"] == "0.1.9+lil.cu134.gabc"
    assert requirements["torch"] == "torch==2.14.0a0+nv"
    assert message["X-Local-Inference-Runtime"] == "jovian-cu134-torch214-cxx11"


def test_rewrite_metadata_rejects_missing_torch() -> None:
    metadata = b"Metadata-Version: 2.4\nName: instanttensor\nVersion: 0.1.9\n\n"
    with pytest.raises(ValueError, match="torch is missing"):
        rewrite_metadata(
            metadata,
            version="0.1.9+lil.cu134.gabc",
            torch_version="2.14.0a0+nv",
        )
