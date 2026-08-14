"""Tests for vllm-rdna-docker image verification (tools/verify.py + build.py verify).

No real Podman/Docker/syft is required: ``subprocess.run`` and
``shutil.which`` are monkeypatched throughout. Covers:
  * happy path — inspect labels + digest verify, SBOM generated non-empty;
  * missing label — tampered inspect output fails ``label_ok`` with the
    missing key named in the errors;
  * syft absent — ``sbom_ok=False`` with "syft not on PATH" while
    ``arch_ok``/``label_ok`` stay True;
  * CLI — ``build.py verify --config ... --image <id>`` exits 0 on a clean
    stubbed verify and 1 on tampered labels.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import tools.verify
from tools._build_cli import main, verify_expected_labels
from tools.resolve import load_config, resolve_config
from tools.verify import VerifyResult, verify_image

SUBPROJECT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = SUBPROJECT / "config" / "rocm-7.2.0.toml"

IMAGE_REF = "docker.io/blivioniag/vllm-rdna:v0.26.0-extras-rocm7.14.0"
DIGEST = "sha256:" + "ab" * 32
ARCHES = ("gfx1030", "gfx1100", "gfx1101")

EXPECTED_LABELS = {
    "vllm.variant": "extras-fork",
    "vllm.repository": "https://github.com/blivioniag/vllm.git",
    "vllm.ref": "v0.26.0-extras",
    "vllm.commit": "9f3b6d1a8c5e0274b6d8a0c2e4f6a8c0d2e4f6a8",
    "org.opencontainers.image.config-hash": "f" * 64,
}

SPDX_JSON = '{"spdxVersion": "SPDX-2.3", "name": "vllm-rdna"}'


def _inspect_payload(labels: dict[str, str]) -> str:
    return json.dumps([{"Labels": labels, "Digest": DIGEST}])


def _stub_run(
    calls: list[list[str]],
    labels: dict[str, str],
    syft_stdout: str = SPDX_JSON,
):
    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1:2] == ["inspect"]:
            return subprocess.CompletedProcess(argv, 0, stdout=_inspect_payload(labels), stderr="")
        if argv[0] == "syft":
            return subprocess.CompletedProcess(argv, 0, stdout=syft_stdout, stderr="")
        raise AssertionError(f"unexpected subprocess call: {argv}")

    return _fake_run


def _stub_which(name: str) -> str | None:
    return f"/usr/bin/{name}" if name in ("podman", "docker", "syft") else None


@pytest.fixture()
def happy_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[list[list[str]], Path, dict[str, str]]:
    """Stubbed engine + syft; evidence redirected to tmp_path."""
    labels = dict(EXPECTED_LABELS)
    labels["org.opencontainers.image.architectures"] = ";".join(ARCHES)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, labels))
    monkeypatch.setattr(shutil, "which", _stub_which)
    monkeypatch.setattr(tools.verify, "DEFAULT_EVIDENCE_DIR", tmp_path)
    return calls, tmp_path, labels


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_verify_image_happy(
    happy_env: tuple[list[list[str]], Path, dict[str, str]]
) -> None:
    calls, tmp_path, _ = happy_env
    result = verify_image(
        IMAGE_REF,
        expected=EXPECTED_LABELS,
        engine="podman",
        architectures=ARCHES,
    )
    assert isinstance(result, VerifyResult)
    assert result.arch_ok and result.label_ok and result.sbom_ok
    assert result.errors == ()
    assert result.image == IMAGE_REF
    assert result.digest == DIGEST
    assert result.labels["vllm.variant"] == "extras-fork"
    # Inspect ran with shlex.split'd argv; syft ran with the engine scheme.
    assert calls[0] == ["podman", "inspect", IMAGE_REF]
    assert calls[1] == ["syft", f"podman://{IMAGE_REF}", "-o", "spdx-json", "--quiet"]
    # SBOM evidence exists and is non-empty.
    assert result.sbom_path is not None
    assert result.sbom_path.parent == tmp_path
    assert result.sbom_path.name == "vllm-rdna-v0.26.0-extras-rocm7.14.0.spdx.json"
    assert result.sbom_path.stat().st_size > 0
    assert json.loads(result.sbom_path.read_text())["spdxVersion"] == "SPDX-2.3"


def test_verify_image_arch_alias_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``vllm.architectures`` is accepted as an alias for the OCI key."""
    labels = dict(EXPECTED_LABELS)
    labels["vllm.architectures"] = ",".join(ARCHES)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, labels))
    monkeypatch.setattr(shutil, "which", _stub_which)
    result = verify_image(
        IMAGE_REF, expected=EXPECTED_LABELS, architectures=ARCHES, evidence_dir=tmp_path
    )
    assert result.arch_ok


# ---------------------------------------------------------------------------
# Missing / tampered label
# ---------------------------------------------------------------------------


def test_verify_image_missing_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = dict(EXPECTED_LABELS)
    del labels["vllm.variant"]  # tampered: expected label absent
    labels["org.opencontainers.image.architectures"] = ";".join(ARCHES)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, labels))
    monkeypatch.setattr(shutil, "which", _stub_which)
    result = verify_image(
        IMAGE_REF, expected=EXPECTED_LABELS, architectures=ARCHES, evidence_dir=tmp_path
    )
    assert not result.label_ok
    assert result.arch_ok and result.sbom_ok  # independent verdicts
    assert any("vllm.variant" in error for error in result.errors)


def test_verify_image_tampered_label_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = dict(EXPECTED_LABELS)
    labels["vllm.commit"] = "0" * 40  # tampered value
    labels["org.opencontainers.image.architectures"] = ";".join(ARCHES)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, labels))
    monkeypatch.setattr(shutil, "which", _stub_which)
    result = verify_image(
        IMAGE_REF, expected=EXPECTED_LABELS, architectures=ARCHES, evidence_dir=tmp_path
    )
    assert not result.label_ok
    assert any("vllm.commit" in error for error in result.errors)


def test_verify_image_missing_architecture_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = dict(EXPECTED_LABELS)  # no architectures label at all
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, labels))
    monkeypatch.setattr(shutil, "which", _stub_which)
    result = verify_image(
        IMAGE_REF, expected=EXPECTED_LABELS, architectures=ARCHES, evidence_dir=tmp_path
    )
    assert not result.arch_ok
    assert result.label_ok
    assert any("Architecture" in error for error in result.errors)


# ---------------------------------------------------------------------------
# syft unavailable / SBOM failures
# ---------------------------------------------------------------------------


def test_verify_image_syft_not_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = dict(EXPECTED_LABELS)
    labels["org.opencontainers.image.architectures"] = ";".join(ARCHES)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, labels))
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None
    )
    result = verify_image(
        IMAGE_REF, expected=EXPECTED_LABELS, architectures=ARCHES, evidence_dir=tmp_path
    )
    assert not result.sbom_ok
    assert result.sbom_path is None
    assert "syft not on PATH" in result.errors
    # Label and architecture checks still ran and passed.
    assert result.arch_ok and result.label_ok
    # syft was never invoked.
    assert all(argv[0] != "syft" for argv in calls)


def test_verify_image_empty_sbom_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    labels = dict(EXPECTED_LABELS)
    labels["org.opencontainers.image.architectures"] = ";".join(ARCHES)
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, labels, syft_stdout=""))
    monkeypatch.setattr(shutil, "which", _stub_which)
    result = verify_image(
        IMAGE_REF, expected=EXPECTED_LABELS, architectures=ARCHES, evidence_dir=tmp_path
    )
    assert not result.sbom_ok
    assert any("EmptySBOM" in error for error in result.errors)


# ---------------------------------------------------------------------------
# CLI: build.py verify
# ---------------------------------------------------------------------------


def _cli_labels() -> dict[str, str]:
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    record = next(i for i in resolved.image_records if i.id == "extras026-rocm714")
    labels = verify_expected_labels(record, resolved)
    labels["org.opencontainers.image.architectures"] = ";".join(
        resolved.architecture_list
    )
    return labels


def test_cli_verify_exits_0_on_clean_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, _cli_labels()))
    monkeypatch.setattr(shutil, "which", _stub_which)
    monkeypatch.setattr(tools.verify, "DEFAULT_EVIDENCE_DIR", tmp_path)
    exit_code = main(
        [
            "verify",
            "--config",
            str(EXAMPLE_CONFIG),
            "--image",
            "extras026-rocm714",
            "--engine",
            "podman",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "OK" in captured.err
    assert any(argv[1:2] == ["inspect"] for argv in calls)
    assert any(argv[0] == "syft" for argv in calls)


def test_cli_verify_exits_1_on_tampered_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = _cli_labels()
    labels["vllm.commit"] = "0" * 40  # tampered
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, labels))
    monkeypatch.setattr(shutil, "which", _stub_which)
    monkeypatch.setattr(tools.verify, "DEFAULT_EVIDENCE_DIR", tmp_path)
    exit_code = main(
        [
            "verify",
            "--config",
            str(EXAMPLE_CONFIG),
            "--image",
            "extras026-rocm714",
            "--engine",
            "podman",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.err
    assert "vllm.commit" in captured.err


def test_cli_verify_exits_1_when_syft_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _stub_run(calls, _cli_labels()))
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None
    )
    monkeypatch.setattr(tools.verify, "DEFAULT_EVIDENCE_DIR", tmp_path)
    exit_code = main(
        [
            "verify",
            "--config",
            str(EXAMPLE_CONFIG),
            "--image",
            "extras026-rocm714",
            "--engine",
            "podman",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "syft not on PATH" in captured.err


def test_cli_verify_dry_run_never_invokes_engine(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("verify --dry-run attempted a subprocess call")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(shutil, "which", _stub_which)
    exit_code = main(
        [
            "verify",
            "--config",
            str(EXAMPLE_CONFIG),
            "--image",
            "extras026-rocm714",
            "--engine",
            "podman",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "dry-run" in captured.err
