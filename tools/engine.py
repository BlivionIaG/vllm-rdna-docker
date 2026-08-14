#!/usr/bin/env python3
"""Container engine abstraction for vllm-rdna-docker.

One home for everything that knows about the engine executable name
(``podman`` / ``docker``) and how to render build / push / pull / tag
argv lists deterministically.

Podman is the canonical engine; Docker is an adapter. The rendering
differs only in argv[0] and Docker's ``--pull`` flag (dropped for
Podman per the project contract); the ``--file`` / ``--tag`` /
``--build-arg`` lists are identical.

The execution helper ``emit_and_run`` is also here because it is
engine-coupled: it prints each rendered command (stdout, shell-quoted)
and optionally invokes it, raising ``CommandFailed`` on the first
non-zero exit.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

try:  # pytest / package mode (vllm-rdna-docker/ on sys.path)
    from tools.errors import CommandFailed, EngineNotFound
    from tools.records import BaseRecord, ImageRecord
except ModuleNotFoundError:  # script mode
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.errors import CommandFailed, EngineNotFound
    from tools.records import BaseRecord, ImageRecord


# ---------------------------------------------------------------------------
# Build-arg orderings (consumed by the build CLI for deterministic argv)
# ---------------------------------------------------------------------------

#: Build-arg order for ``Dockerfile.base``. New args must be appended.
BASE_ARG_ORDER: tuple[str, ...] = (
    "BASE_IMAGE",
    "BASE_DIGEST",
    "ROCM_VERSION",
    "PYTHON_VERSION",
    "PYTORCH_VERSION",
    "TRITON_VERSION",
    "PYTORCH_INDEX_URL",
    "PYTORCH_ROCM_ARCH",
    "BASE_TAG",
    "FLASH_ATTENTION_INSTALL",
    "FLASH_ATTENTION_VERSION",
    "FLASH_ATTENTION_REPO",
    "FLASH_ATTENTION_REF",
)

#: Build-arg order for ``Dockerfile.vllm``. New args must be appended.
VLLM_ARG_ORDER: tuple[str, ...] = (
    "BASE_IMAGE",
    "VLLM_REPOSITORY",
    "VLLM_REF",
    "VLLM_COMMIT",
    "VLLM_VARIANT",
    "PYTORCH_ROCM_ARCH",
    "TORCH_BACKEND",
    "IMAGE_TAG",
    "CONFIG_HASH",
    "FLASH_ATTENTION_INSTALL",
    "FLASH_ATTENTION_VERSION",
    "FLASH_ATTENTION_REPO",
    "FLASH_ATTENTION_REF",
)

FLASH_ATTENTION_DEFAULT_REPO = "https://github.com/Dao-AILab/flash-attention"


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------


def select_engine(
    requested: str, *, which: Callable[[str], str | None] | None = None
) -> str:
    """Resolve the engine executable name ("podman" or "docker").

    ``requested`` is the raw ``--engine`` value (default from
    ``CONTAINER_ENGINE`` or "auto"). Raises ``EngineNotFound`` for a forced
    engine missing from PATH, for "auto" when neither candidate exists, and
    for any unrecognized engine name. ``which`` is resolved at call time so
    tests can monkeypatch ``shutil.which``.
    """
    if which is None:
        which = shutil.which
    if requested == "auto":
        for candidate in ("podman", "docker"):
            if which(candidate):
                return candidate
        raise EngineNotFound(engine="auto")
    if requested not in ("podman", "docker"):
        raise EngineNotFound(engine=requested)
    if not which(requested):
        raise EngineNotFound(engine=requested)
    return requested


# ---------------------------------------------------------------------------
# Build-arg renderers
# ---------------------------------------------------------------------------


def _flash_attention_args(fa: Mapping[str, str], *, install: str) -> dict[str, str]:
    """Render the four FLASH_ATTENTION_* build args for one install target.

    ``install`` is the layer being built ("base" or "vllm"); when the config's
    install location does not match, the args render as a no-op
    (INSTALL=none) so the other Dockerfile's FA step stays inert.
    """
    fa_install = fa.get("install", "none")
    active = fa_install == install
    return {
        "FLASH_ATTENTION_INSTALL": fa_install if active else "none",
        "FLASH_ATTENTION_VERSION": fa.get("version", "") if active else "",
        "FLASH_ATTENTION_REPO": fa.get("repo", FLASH_ATTENTION_DEFAULT_REPO)
        if active
        else FLASH_ATTENTION_DEFAULT_REPO,
        "FLASH_ATTENTION_REF": fa.get("ref", "") if active else "",
    }


def base_build_args(
    record: BaseRecord, architecture_list: Sequence[str]
) -> list[tuple[str, str]]:
    """Ordered (name, value) build args for one resolved base record."""
    values = {
        "BASE_IMAGE": record.base_image,
        "BASE_DIGEST": record.base_digest,
        "ROCM_VERSION": record.rocm_version,
        "PYTHON_VERSION": record.python_version,
        "PYTORCH_VERSION": record.pytorch_version,
        "TRITON_VERSION": record.triton_version,
        "PYTORCH_INDEX_URL": record.pytorch_index_url,
        "PYTORCH_ROCM_ARCH": ";".join(architecture_list),
        "BASE_TAG": record.tag,
        **_flash_attention_args(record.flash_attention, install="base"),
    }
    return [(name, values[name]) for name in BASE_ARG_ORDER]


def vllm_build_args(
    record: ImageRecord,
    base_image_ref: str,
    config_hash: str,
    architecture_list: Sequence[str],
) -> list[tuple[str, str]]:
    """Ordered (name, value) build args for one resolved image record."""
    source = record.source_record
    rocm_mm = ".".join(record.base_record.rocm_version.split(".")[:2])
    values = {
        "BASE_IMAGE": base_image_ref,
        "VLLM_REPOSITORY": source.repository,
        "VLLM_REF": source.ref,
        "VLLM_COMMIT": source.resolved_commit,
        "VLLM_VARIANT": source.variant,
        "PYTORCH_ROCM_ARCH": ";".join(architecture_list),
        "TORCH_BACKEND": f"rocm{rocm_mm}",
        "IMAGE_TAG": record.tag,
        "CONFIG_HASH": config_hash,
        **_flash_attention_args(record.base_record.flash_attention, install="vllm"),
    }
    return [(name, values[name]) for name in VLLM_ARG_ORDER]


def render_build_argv(
    engine: str,
    dockerfile: str,
    tag: str,
    args: Sequence[tuple[str, str]],
    context: str,
) -> list[str]:
    """Render one deterministic build argv.

    Podman and Docker outputs differ ONLY in argv[0] and Docker's ``--pull``
    (dropped for Podman per the project contract); the ``--file`` / ``--tag``
    / ``--build-arg`` lists are identical.
    """
    argv = [engine, "build", "--no-cache"]
    if engine == "docker":
        argv.append("--pull")
    argv += ["--file", dockerfile]
    for name, value in args:
        argv += ["--build-arg", f"{name}={value}"]
    argv += ["--tag", tag, context]
    return argv


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def emit_and_run(
    commands: Sequence[tuple[str, list[str]]],
    *,
    dry_run: bool,
) -> int:
    """Print each rendered command (stdout, shell-quoted) and optionally run it.

    Returns 0 on success; raises ``CommandFailed`` on the first non-zero
    engine exit (non-dry-run only).
    """
    for label, argv in commands:
        rendered = shlex.join(argv)
        print(rendered)  # stdout: capturable by CI, always emitted
        print(f"[{label}] {rendered}", file=sys.stderr)
        if dry_run:
            print(f"[{label}] dry-run: not executing", file=sys.stderr)
            continue
        proc = subprocess.run(argv, check=False)
        if proc.returncode != 0:
            raise CommandFailed(command=rendered, exit_code=proc.returncode)
    return 0


def display_path(path: Path) -> str:
    """Render a path relative to the cwd when possible (no host absolutes)."""
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:  # different drive (Windows) — fall back to as-is
        return str(path)