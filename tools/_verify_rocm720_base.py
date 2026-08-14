#!/usr/bin/env python3
"""Verify the rocm720 base image via tools.verify.verify_image."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from verify import verify_image  # type: ignore

IMAGE = "docker.io/blivioniag/rocm-rdna:7.2.0"
ARCHS = (
    "gfx1030",
    "gfx1100",
    "gfx1101",
    "gfx1150",
    "gfx1151",
    "gfx1200",
    "gfx1201",
)
EXPECTED = {
    "org.opencontainers.image.title": "vllm-rdna-base",
    "org.opencontainers.image.rocm": "7.2.0",
    "org.opencontainers.image.python": "3.12",
    "org.opencontainers.image.pytorch": "2.12.0",
    "org.opencontainers.image.triton": "3.5.1",
    "org.opencontainers.image.pytorch-index-url": "https://download.pytorch.org/whl/rocm7.2",
    "org.opencontainers.image.architectures": ";".join(ARCHS),
    "org.opencontainers.image.primary-arch": "gfx1030",
    "org.opencontainers.image.flash-attention.install": "base",
    "org.opencontainers.image.flash-attention.version": "2.8.4",
    "org.opencontainers.image.flash-attention.repo": "https://github.com/ROCm/flash-attention.git",
    "org.opencontainers.image.flash-attention.ref": "tridao",
}

result = verify_image(
    IMAGE,
    expected=EXPECTED,
    engine="podman",
    architectures=ARCHS,
)
print(f"image      : {IMAGE}")
print(f"label_ok   : {result.label_ok}")
print(f"arch_ok    : {result.arch_ok}")
print(f"sbom_ok    : {result.sbom_ok}")
print(f"digest     : {result.digest or '(none)'}")
print(f"sbom_path  : {result.sbom_path or '(none)'}")
print(f"label_count: {len(result.labels)}")
if result.errors:
    print("errors     :")
    for e in result.errors:
        print(f"  - {e}")
else:
    print("errors     : (none)")
sys.exit(0 if (result.label_ok and result.arch_ok) else 1)
