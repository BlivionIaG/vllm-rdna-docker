#!/usr/bin/env python3
"""docker buildx bake file generation for vllm-rdna-docker.

Bake is the modern, declarative build mechanism: ``docker buildx bake``
reads a JSON file describing the target graph and runs the build. The
graph here is generated entirely from the validated config — adding a
new ``[bases.*]`` or ``[[images]]`` entry in ``config/rocm-7.2.0.toml``
automatically extends the bake graph with no source change.

CI runs on GitHub-hosted runners with Docker + buildx. Local
development uses Podman directly via ``tools/engine.py:render_build_argv``
(``--engine podman``); this Bake path is for the Docker CI flow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:  # pytest / package mode (vllm-rdna-docker/ on sys.path)
    from tools.engine import FLASH_ATTENTION_DEFAULT_REPO
    from tools.records import BaseRecord, ImageRecord, ResolvedBuild
except ModuleNotFoundError:  # script mode
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.engine import FLASH_ATTENTION_DEFAULT_REPO
    from tools.records import BaseRecord, ImageRecord, ResolvedBuild


# ---------------------------------------------------------------------------
# Target name helpers
# ---------------------------------------------------------------------------


def base_target_name(base_id: str) -> str:
    """Canonical Bake target for a ``[bases.*]`` entry."""
    return f"base-{base_id}"


def vllm_target_name(image_id: str) -> str:
    """Canonical Bake target for a ``[[images]]`` entry."""
    return f"vllm-{image_id}"


# ---------------------------------------------------------------------------
# Bake file rendering
# ---------------------------------------------------------------------------


def render_bake_file(
    resolved: ResolvedBuild,
    registry: Mapping[str, str],
    project_dir: Path,
) -> Path:
    """Write ``docker-bake.json`` at ``<project_dir>/docker-bake.json`` and
    return the path. The file contains one target per ``[bases.*]`` entry
    (``base-<id>``) and one target per ``[[images]]`` entry (``vllm-<id>``),
    plus three convenience groups (``all-bases``, ``all-vllm``, ``all``).
    """
    arch_str = ";".join(resolved.architecture_list)
    targets: dict[str, Any] = {}

    for base in resolved.base_records:
        targets[base_target_name(base.id)] = _base_target(base, arch_str, registry)

    for image in resolved.image_records:
        targets[vllm_target_name(image.id)] = _vllm_target(
            image, arch_str, resolved.config_hash, registry
        )

    bake = {
        "group": {
            "all-bases": {
                "targets": [base_target_name(b.id) for b in resolved.base_records],
            },
            "all-vllm": {
                "targets": [vllm_target_name(i.id) for i in resolved.image_records],
            },
            "all": {
                "targets": (
                    [base_target_name(b.id) for b in resolved.base_records]
                    + [vllm_target_name(i.id) for i in resolved.image_records]
                ),
            },
        },
        "target": targets,
    }

    bake_file = project_dir / "docker-bake.json"
    bake_file.write_text(json.dumps(bake, indent=2, sort_keys=True) + "\n")
    return bake_file


def _base_target(
    base: BaseRecord, arch_str: str, registry: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "context": ".",
        "dockerfile": "Dockerfile.base",
        "args": {
            "BASE_IMAGE": base.base_image,
            "BASE_DIGEST": base.base_digest,
            "ROCM_VERSION": base.rocm_version,
            "PYTHON_VERSION": base.python_version,
            "PYTORCH_VERSION": base.pytorch_version,
            "TRITON_VERSION": base.triton_version,
            "PYTORCH_INDEX_URL": base.pytorch_index_url,
            "PYTORCH_ROCM_ARCH": arch_str,
            "BASE_TAG": base.tag,
            **_flash_attention_args(base.flash_attention),
        },
        "tags": [_base_image_ref(registry, base)],
    }


def _vllm_target(
    image: ImageRecord,
    arch_str: str,
    config_hash: str,
    registry: Mapping[str, str],
) -> dict[str, Any]:
    source = image.source_record
    base = image.base_record
    rocm_mm = ".".join(base.rocm_version.split(".")[:2])
    base_image_ref = _base_image_ref(registry, base)
    return {
        "context": ".",
        "dockerfile": "Dockerfile.vllm",
        "args": {
            "BASE_IMAGE": base_image_ref,
            "VLLM_REPOSITORY": source.repository,
            "VLLM_REF": source.ref,
            "VLLM_COMMIT": source.resolved_commit,
            "VLLM_VARIANT": source.variant,
            "PYTORCH_ROCM_ARCH": arch_str,
            "TORCH_BACKEND": f"rocm{rocm_mm}",
            "IMAGE_TAG": image.tag,
            "CONFIG_HASH": config_hash,
            **_flash_attention_args(base.flash_attention),
        },
        "tags": [_vllm_image_ref(registry, image)],
    }


def _base_image_ref(registry: Mapping[str, str], base: BaseRecord) -> str:
    return f"{registry['host']}/{registry['base_repository']}:{base.tag}"


def _vllm_image_ref(registry: Mapping[str, str], image: ImageRecord) -> str:
    return f"{registry['host']}/{registry['vllm_repository']}:{image.tag}"


def _flash_attention_args(fa: Mapping[str, str]) -> dict[str, str]:
    """Render the four FLASH_ATTENTION_* args for a base's flash_attention
    table. Bake handles the per-layer activation (base vs vllm) via a
    Dockerfile-level switch, so we always emit the install value verbatim
    — the consuming Dockerfile branches on it.
    """
    return {
        "FLASH_ATTENTION_INSTALL": fa.get("install", "none"),
        "FLASH_ATTENTION_VERSION": fa.get("version", ""),
        "FLASH_ATTENTION_REPO": fa.get("repo", FLASH_ATTENTION_DEFAULT_REPO),
        "FLASH_ATTENTION_REF": fa.get("ref", ""),
    }


# ---------------------------------------------------------------------------
# Build argv rendering
# ---------------------------------------------------------------------------


def bake_argv(targets: str | list[str], bake_file: Path) -> list[str]:
    """Render the ``docker buildx bake`` argv for one or more targets.

    Accepts a single target name or a list of target names. Multiple
    targets are built in parallel by buildx — this is the multi-target
    capability that makes Bake a real win over one-invocation-per-target
    loops. ``docker buildx bake`` resolves the target graph from the JSON
    file. The --load / --push flags are not set here — the caller
    decides whether the result is loaded into the local daemon (default)
    or pushed to a registry.
    """
    target_list = [targets] if isinstance(targets, str) else list(targets)
    return ["docker", "buildx", "bake", *target_list, "--file", str(bake_file)]