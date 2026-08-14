"""Tests for vllm-rdna-docker publication and alias promotion.

Covers ``tools/publish.py`` (publish_image / promote_alias) and the
``publish`` / ``promote`` subcommands of ``tools/build.py``:

  * dry-run rendering — exact commands, no engine invocation, no credential
    requirement;
  * live push with a stubbed ``subprocess.run`` — the pushed digest is
    extracted from stdout in both Docker (``digest: sha256:...``) and Podman
    (``Digest: sha256:...``) formats;
  * credential safety — ``MissingCredentials`` fires BEFORE any engine
    invocation when no credential env vars are set; credential VALUES never
    appear in rendered commands or results;
  * failure mapping — a non-zero engine exit raises ``PushFailed`` with the
    captured stderr;
  * legacy tag protection — ``publish --image v0.22.1`` and
    ``promote --alias v0.22.1_base`` are refused with ``ReservedTag``;
  * unknown alias — ``promote --alias latest`` exits 1 with ``UnknownAlias``.

No test requires a real engine, a real registry, or a network connection:
``subprocess.run`` and ``shutil.which`` are monkeypatched throughout.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tools.build import main
from tools.publish import (
    CREDENTIAL_ENV_PAIRS,
    MissingCredentials,
    PushFailed,
    UnknownAlias,
    extract_digest,
    promote_alias,
    publish_image,
)
from tools.validate import ReservedTag

SUBPROJECT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = SUBPROJECT / "config" / "rocm-7.2.0.toml"

VLLM_REPO = "docker.io/blivioniag/vllm-rdna"
FAKE_DIGEST = "sha256:" + "ab" * 32
FAKE_TOKEN = "ci-token-value-that-must-never-leak"

_CRED_VARS = [var for pair in CREDENTIAL_ENV_PAIRS for var in pair]


@pytest.fixture(autouse=True)
def _clean_credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambient registry credentials must never leak into a test."""
    for var in _CRED_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def _engine_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name: f"/usr/bin/{name}" if name == "podman" else None
    )


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGISTRY_USER", "ci-bot")
    monkeypatch.setenv("REGISTRY_TOKEN", FAKE_TOKEN)


def _completed(argv: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")


# ---------------------------------------------------------------------------
# publish_image — dry run
# ---------------------------------------------------------------------------


def test_publish_image_dry_run_renders_command_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("dry-run attempted a subprocess call")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    result = publish_image(
        "docker.io/blivioniag/vllm-rdna:v0.26.0", engine="podman", dry_run=True
    )
    assert result.dry_run is True
    assert result.pushed_digest is None
    assert result.command == "podman push docker.io/blivioniag/vllm-rdna:v0.26.0"
    assert result.image == "docker.io/blivioniag/vllm-rdna:v0.26.0"


def test_publish_image_dry_run_needs_no_credentials() -> None:
    # The autouse fixture has deleted every credential var; dry-run must pass.
    result = publish_image(f"{VLLM_REPO}:v0.26.0", engine="docker", dry_run=True)
    assert result.command == f"docker push {VLLM_REPO}:v0.26.0"


# ---------------------------------------------------------------------------
# publish_image — live (stubbed subprocess)
# ---------------------------------------------------------------------------


def test_publish_image_live_extracts_docker_digest(
    monkeypatch: pytest.MonkeyPatch, _credentials: None
) -> None:
    captured: list[list[str]] = []

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(list(argv))
        return _completed(argv, stdout=f"latest: digest: {FAKE_DIGEST} size: 1234\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = publish_image(f"{VLLM_REPO}:v0.26.0", engine="docker")
    assert captured == [["docker", "push", f"{VLLM_REPO}:v0.26.0"]]
    assert result.dry_run is False
    assert result.pushed_digest == FAKE_DIGEST


def test_publish_image_live_extracts_podman_digest(
    monkeypatch: pytest.MonkeyPatch, _credentials: None
) -> None:
    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, stdout=f"Writing manifest...\nDigest: {FAKE_DIGEST}\n")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = publish_image(f"{VLLM_REPO}:v0.26.0", engine="podman")
    assert result.pushed_digest == FAKE_DIGEST


def test_extract_digest_accepts_both_formats() -> None:
    assert extract_digest(f"digest: {FAKE_DIGEST} size: 1") == FAKE_DIGEST
    assert extract_digest(f"Digest: {FAKE_DIGEST}") == FAKE_DIGEST
    assert extract_digest("no digest here") is None


def test_publish_image_live_missing_credentials_raises_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("push attempted without credentials")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    with pytest.raises(MissingCredentials) as excinfo:
        publish_image(f"{VLLM_REPO}:v0.26.0", engine="podman")
    # Only variable NAMES are referenced — never values.
    assert "REGISTRY_USER/REGISTRY_TOKEN" in str(excinfo.value)
    assert FAKE_TOKEN not in str(excinfo.value)


def test_publish_image_live_engine_failure_raises_push_failed(
    monkeypatch: pytest.MonkeyPatch, _credentials: None
) -> None:
    def _failing_run(argv: list[str], **kwargs: object) -> object:
        raise subprocess.CalledProcessError(125, argv, stderr="denied: requested access")

    monkeypatch.setattr(subprocess, "run", _failing_run)
    with pytest.raises(PushFailed) as excinfo:
        publish_image(f"{VLLM_REPO}:v0.26.0", engine="podman")
    assert excinfo.value.context["image"] == f"{VLLM_REPO}:v0.26.0"
    assert "denied" in excinfo.value.context["stderr"]


# ---------------------------------------------------------------------------
# Credential hygiene
# ---------------------------------------------------------------------------


def test_credentials_never_appear_in_commands_or_results(
    monkeypatch: pytest.MonkeyPatch, _credentials: None
) -> None:
    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert FAKE_TOKEN not in argv
        return _completed(argv, stdout=f"digest: {FAKE_DIGEST}")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    published = publish_image(f"{VLLM_REPO}:v0.26.0", engine="podman")
    promoted = promote_alias(
        f"{VLLM_REPO}:v0.26.0-rocm7.2.1", f"{VLLM_REPO}:v0.26.0", engine="podman"
    )
    for artifact in (published, promoted):
        assert FAKE_TOKEN not in artifact.command
        assert FAKE_TOKEN not in repr(artifact)


# ---------------------------------------------------------------------------
# promote_alias
# ---------------------------------------------------------------------------


def test_promote_alias_dry_run_renders_pull_tag_push_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("dry-run attempted a subprocess call")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    source = f"{VLLM_REPO}:v0.26.0-rocm7.2.1"
    target = f"{VLLM_REPO}:v0.26.0"
    result = promote_alias(source, target, engine="podman", dry_run=True)
    assert result.source == source
    assert result.target == target
    assert result.pushed_digest is None
    assert result.command == (
        f"podman pull {source} && podman tag {source} {target} && podman push {target}"
    )


def test_promote_alias_live_runs_pull_tag_push_in_order(
    monkeypatch: pytest.MonkeyPatch, _credentials: None
) -> None:
    captured: list[list[str]] = []

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(list(argv))
        stdout = f"digest: {FAKE_DIGEST}" if argv[1] == "push" else ""
        return _completed(argv, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    source = f"{VLLM_REPO}:v0.26.0-rocm7.2.1"
    target = f"{VLLM_REPO}:v0.26.0"
    result = promote_alias(source, target, engine="podman")
    assert captured == [
        ["podman", "pull", source],
        ["podman", "tag", source, target],
        ["podman", "push", target],
    ]
    assert result.pushed_digest == FAKE_DIGEST


def test_promote_alias_live_missing_credentials_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("promotion attempted without credentials")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    with pytest.raises(MissingCredentials):
        promote_alias(f"{VLLM_REPO}:a", f"{VLLM_REPO}:b", engine="podman")


# ---------------------------------------------------------------------------
# CLI: publish subcommand
# ---------------------------------------------------------------------------


def test_cli_publish_single_image_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    _engine_present: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("dry-run attempted a subprocess call")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    exit_code = main(
        [
            "publish",
            "--config", str(EXAMPLE_CONFIG),
            "--image", "vllm-026-upstream-rocm721",
            "--engine", "podman",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == (
        f"podman push {VLLM_REPO}:v0.26.0-rocm7.2.1"
    )
    assert FAKE_TOKEN not in captured.out + captured.err


def test_cli_publish_reserved_tag_refused(
    monkeypatch: pytest.MonkeyPatch,
    _engine_present: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("reserved tag reached the engine")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    exit_code = main(
        [
            "publish",
            "--config", str(EXAMPLE_CONFIG),
            "--image", "v0.22.1",
            "--engine", "podman",
            "--dry-run",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ReservedTag" in captured.err
    assert "v0.22.1" in captured.err
    assert captured.out == ""


def test_cli_publish_live_without_credentials_exits_1_before_push(
    monkeypatch: pytest.MonkeyPatch,
    _engine_present: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("push attempted without credentials")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    exit_code = main(
        [
            "publish",
            "--config", str(EXAMPLE_CONFIG),
            "--image", "vllm-026-upstream-rocm720",
            "--engine", "podman",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "MissingCredentials" in captured.err
    assert captured.out == ""


def test_cli_publish_all_dry_run_covers_bases_and_images(
    monkeypatch: pytest.MonkeyPatch,
    _engine_present: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("dry-run ran engine")
    )
    exit_code = main(
        ["publish", "--config", str(EXAMPLE_CONFIG), "--engine", "podman", "--dry-run"]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 9  # 3 bases + 6 images
    assert f"podman push docker.io/blivioniag/rocm-rdna:7.2.0" in lines
    assert f"podman push {VLLM_REPO}:v0.26.0" in lines
    assert f"podman push {VLLM_REPO}:v0.26.0-extras-rocm7.14.0" in lines
    # Legacy tags are never rendered.
    assert "v0.22.1" not in captured.out
    assert "v0.22.1_base" not in captured.out


# ---------------------------------------------------------------------------
# CLI: promote subcommand
# ---------------------------------------------------------------------------


def test_cli_promote_alias_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    _engine_present: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("dry-run ran engine")
    )
    exit_code = main(
        [
            "promote",
            "--config", str(EXAMPLE_CONFIG),
            "--alias", "v0.26.0",
            "--engine", "podman",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    ref = f"{VLLM_REPO}:v0.26.0"
    assert captured.out.strip() == (
        f"podman pull {ref} && podman tag {ref} {ref} && podman push {ref}"
    )


def test_cli_promote_unknown_alias_exits_1(
    monkeypatch: pytest.MonkeyPatch,
    _engine_present: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("unknown alias ran engine")
    )
    exit_code = main(
        [
            "promote",
            "--config", str(EXAMPLE_CONFIG),
            "--alias", "latest",
            "--engine", "podman",
            "--dry-run",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "UnknownAlias" in captured.err
    assert "latest" in captured.err
    assert captured.out == ""


def test_cli_promote_reserved_alias_refused(
    monkeypatch: pytest.MonkeyPatch,
    _engine_present: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("reserved alias ran engine")
    )
    exit_code = main(
        [
            "promote",
            "--config", str(EXAMPLE_CONFIG),
            "--alias", "v0.22.1_base",
            "--engine", "podman",
            "--dry-run",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ReservedTag" in captured.err
    assert "v0.22.1_base" in captured.err


def test_cli_promote_all_dry_run_never_touches_reserved_tags(
    monkeypatch: pytest.MonkeyPatch,
    _engine_present: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("dry-run ran engine")
    )
    exit_code = main(
        ["promote", "--config", str(EXAMPLE_CONFIG), "--engine", "podman", "--dry-run"]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 2  # v0.26.0, v0.26.0-extras
    assert all("&& podman tag" in line and "&& podman push" in line for line in lines)
    assert "v0.22.1" not in captured.out
