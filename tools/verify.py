#!/usr/bin/env python3
"""Image verification, OCI provenance, and SBOM evidence for vllm-rdna-docker.

Verifies that a built image carries the expected OCI labels and architecture
metadata, captures its manifest digest, and produces a non-empty SPDX SBOM
under the evidence directory (``.omo/evidence/``).

Properties:
  * stdlib only (subprocess, shutil, json, shlex, dataclasses, pathlib);
  * syft is OPTIONAL — when it is not on PATH the verifier records
    ``sbom_ok=False`` plus the literal error ``"syft not on PATH"`` and still
    reports the label/architecture verdicts. It never requires syft;
  * engine-neutral — ``<engine> inspect <image>`` produces the same JSON
    shape on Podman (canonical) and Docker (adapter);
  * graceful degradation — a failed ``inspect`` never raises; it cascades
    into ``MissingLabel`` / ``MissingArchitecture`` errors on the result so
    the caller gets structured errors instead of an exception;
  * error strings follow the project convention ``Name(field=value, ...)``
    (e.g. ``MissingLabel(key='vllm.variant')``).

Usage (library):
    from tools.verify import verify_image
    result = verify_image(
        "docker.io/blivioniag/vllm-rdna:v0.26.0",
        expected={"vllm.variant": "upstream", ...},
        engine="podman",
        architectures=("gfx1030", "gfx1100", ...),
    )
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

#: Default evidence root: ``<repo>/.omo/evidence`` (repo = parents[2] of
#: tools/verify.py). Referenced at call time (never as a default argument)
#: so tests can monkeypatch it.
DEFAULT_EVIDENCE_DIR = Path(__file__).resolve().parents[2] / ".omo" / "evidence"

#: Label keys accepted for the architecture list. Either is acceptable.
ARCHITECTURE_LABEL_KEYS: tuple[str, ...] = (
    "org.opencontainers.image.architectures",
    "vllm.architectures",
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifyResult:
    """Structured outcome of one image verification."""

    image: str
    labels: Mapping[str, str] = field(default_factory=dict)
    digest: str = ""
    sbom_path: Path | None = None
    arch_ok: bool = False
    label_ok: bool = False
    sbom_ok: bool = False
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.arch_ok and self.label_ok and self.sbom_ok


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------


def _inspect_image(
    engine: str, image_ref: str, errors: list[str]
) -> tuple[dict[str, str], str]:
    """Run ``<engine> inspect <image>``; return (labels, digest).

    On any failure (non-zero exit, unparseable JSON) records a named error
    and returns empty values — the caller's checks then cascade into
    MissingLabel / MissingArchitecture errors instead of raising.
    """
    argv = shlex.split(f"{engine} inspect {image_ref}")
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        errors.append(
            f"InspectFailed(command={shlex.join(argv)!r}, "
            f"exit_code={proc.returncode})"
        )
        return {}, ""
    try:
        payload = json.loads(proc.stdout)
        obj = payload[0] if isinstance(payload, list) else payload
        if not isinstance(obj, dict):
            raise TypeError(type(obj).__name__)
    except (json.JSONDecodeError, IndexError, TypeError):
        errors.append(f"InspectParseError(command={shlex.join(argv)!r})")
        return {}, ""
    config = obj.get("Config") or {}
    labels = config.get("Labels") or obj.get("Labels") or {}
    digest = obj.get("Digest") or obj.get("Id") or ""
    return dict(labels), str(digest)


# ---------------------------------------------------------------------------
# Label + architecture checks
# ---------------------------------------------------------------------------


def _check_labels(
    labels: Mapping[str, str], expected: Mapping[str, str], errors: list[str]
) -> bool:
    """Every key in ``expected`` must equal ``labels[key]`` (case-sensitive)."""
    ok = True
    for key, want in expected.items():
        if key not in labels:
            errors.append(f"MissingLabel(key={key!r})")
            ok = False
        elif labels[key] != want:
            errors.append(
                f"LabelMismatch(key={key!r}, expected={want!r}, "
                f"actual={labels[key]!r})"
            )
            ok = False
    return ok


def _split_arch_label(raw: str) -> list[str]:
    """Split an architecture label into tokens.

    Comma is the contract separator; semicolon is tolerated because the
    Dockerfiles forward ``ARCH_LIST`` in the semicolon-joined form that
    ``PYTORCH_ROCM_ARCH`` expects.
    """
    return [tok.strip() for tok in raw.replace(";", ",").split(",") if tok.strip()]


def _check_architectures(
    labels: Mapping[str, str], architectures: Sequence[str], errors: list[str]
) -> bool:
    """The resolved architecture list must appear as a comma-separated label.

    Either ``org.opencontainers.image.architectures`` or
    ``vllm.architectures`` is acceptable.
    """
    want = list(architectures)
    if not want:
        return True
    present = [(key, labels[key]) for key in ARCHITECTURE_LABEL_KEYS if key in labels]
    if not present:
        errors.append(
            f"MissingArchitecture(expected={','.join(want)!r}, "
            f"labels={ARCHITECTURE_LABEL_KEYS!r})"
        )
        return False
    for _key, raw in present:
        if _split_arch_label(raw) == want:
            return True
    errors.append(
        f"ArchitectureMismatch(expected={','.join(want)!r}, "
        f"actual={present[0][1]!r})"
    )
    return False


# ---------------------------------------------------------------------------
# SBOM
# ---------------------------------------------------------------------------


def _sbom_filename(image_ref: str) -> str:
    """``docker.io/blivioniag/vllm-rdna:v0.26.0`` -> ``vllm-rdna-v0.26.0.spdx.json``."""
    basename = image_ref.rsplit("/", 1)[-1].replace(":", "-")
    return f"{basename}.spdx.json"


def _generate_sbom(
    engine: str, image_ref: str, evidence_dir: Path, errors: list[str]
) -> tuple[Path | None, bool]:
    """Run ``syft <engine>://<image> -o spdx-json -q`` into the evidence dir.

    syft is optional: when missing from PATH, records the literal error
    ``"syft not on PATH"`` and returns ``(None, False)``. A generated but
    empty file is a failure (``EmptySBOM``).
    """
    if shutil.which("syft") is None:
        errors.append("syft not on PATH")
        return None, False
    argv = ["syft", f"{engine}://{image_ref}", "-o", "spdx-json", "--quiet"]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        errors.append(
            f"SyftFailed(command={shlex.join(argv)!r}, exit_code={proc.returncode})"
        )
        return None, False
    evidence_dir.mkdir(parents=True, exist_ok=True)
    sbom_path = evidence_dir / _sbom_filename(image_ref)
    sbom_path.write_text(proc.stdout)
    if sbom_path.stat().st_size == 0:
        errors.append(f"EmptySBOM(path={str(sbom_path)!r})")
        return sbom_path, False
    return sbom_path, True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def verify_image(
    image_ref: str,
    *,
    expected: Mapping[str, str],
    engine: str = "podman",
    architectures: Sequence[str] = (),
    evidence_dir: Path | str | None = None,
) -> VerifyResult:
    """Verify one built image: OCI labels, architecture metadata, SBOM.

    ``expected`` maps OCI label keys to their required values
    (case-sensitive equality). ``architectures`` is the resolved
    ``summary.architectures`` list; it must appear as a comma-separated
    value under ``org.opencontainers.image.architectures`` or
    ``vllm.architectures`` (semicolons tolerated). ``evidence_dir``
    defaults to ``DEFAULT_EVIDENCE_DIR`` (``.omo/evidence/``).
    """
    evidence = Path(evidence_dir) if evidence_dir is not None else DEFAULT_EVIDENCE_DIR
    errors: list[str] = []
    labels, digest = _inspect_image(engine, image_ref, errors)
    label_ok = _check_labels(labels, expected, errors)
    arch_ok = _check_architectures(labels, architectures, errors)
    sbom_path, sbom_ok = _generate_sbom(engine, image_ref, evidence, errors)
    return VerifyResult(
        image=image_ref,
        labels=labels,
        digest=digest,
        sbom_path=sbom_path,
        arch_ok=arch_ok,
        label_ok=label_ok,
        sbom_ok=sbom_ok,
        errors=tuple(errors),
    )
