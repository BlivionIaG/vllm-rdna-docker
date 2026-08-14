"""Tests for the vllm-rdna-docker engine-neutral build CLI (tools/build.py).

Covers:
  * validate / resolve subcommands on the example config;
  * dry-run rendering of the exact podman/docker build commands for
    build-base and build-vllm, with every resolved ARG present;
  * deterministic rendering — Podman and Docker outputs differ only in the
    executable name and Docker's --pull flag; the --build-arg lists are
    byte-identical;
  * engine selection — auto prefers Podman, forced engines fail with the
    named EngineNotFound error when missing (simulated via monkeypatch;
    Podman/Docker presence on the host is not required);
  * non-dry-run execution — a stubbed subprocess.run proves build-base
    invokes the engine with the rendered argv;
  * pipeline gating — verify/publish/promote refuse an invalid config before
    acting (validate+resolve gate every subcommand).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.build import (
    ENGINE_ENV_VAR,
    base_build_args,
    main,
    render_build_argv,
    select_engine,
    vllm_build_args,
)
from tools.cli_errors import CommandFailed, EngineNotFound
from tools.resolve import load_config, resolve_config

SUBPROJECT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUBPROJECT.parent
EXAMPLE_CONFIG = SUBPROJECT / "config" / "rocm-7.2.0.toml"
BUILD_PY = SUBPROJECT / "tools" / "build.py"

GOOD_COMMIT_2 = "9f3b6d1a8c5e0274b6d8a0c2e4f6a8c0d2e4f6a8"


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.pop(ENGINE_ENV_VAR, None)  # ambient engine override must not leak in
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, str(BUILD_PY), *args],
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
        cwd=REPO_ROOT,  # rendered paths are cwd-relative: deterministic
    )


def _build_args_of(argv: list[str]) -> list[str]:
    """Extract the ordered --build-arg values from a rendered argv."""
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == "--build-arg"]


@pytest.fixture(autouse=True)
def _clean_engine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENGINE_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# validate / resolve subcommands
# ---------------------------------------------------------------------------


def test_validate_subcommand_example_config() -> None:
    result = _run_cli("validate", "--config", str(EXAMPLE_CONFIG))
    assert result.returncode == 0
    assert "OK:" in result.stdout
    for base_id in ("rocm720", "rocm714"):
        assert base_id in result.stdout
    for image_id in ("vllm-026-extras-rocm714",):
        assert image_id in result.stdout


def test_validate_subcommand_invalid_config_fails() -> None:
    fixture = SUBPROJECT / "tests" / "fixtures" / "invalid-unknown-source.toml"
    result = _run_cli("validate", "--config", str(fixture))
    assert result.returncode == 1
    assert "UnknownSource" in result.stderr


def test_resolve_subcommand_json() -> None:
    result = _run_cli("resolve", "--config", str(EXAMPLE_CONFIG), "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert len(payload["config_hash"]) == 64
    assert len(payload["image_records"]) == 4
    assert len(payload["base_records"]) == 2


# ---------------------------------------------------------------------------
# build-base dry-run rendering (subprocess, podman present on this host)
# ---------------------------------------------------------------------------


def test_build_base_dry_run_renders_podman_command() -> None:
    result = _run_cli(
        "build-base", "--config", str(EXAMPLE_CONFIG), "--engine=podman", "--dry-run"
    )
    assert result.returncode == 0
    # Rendered commands go to stdout (one per line, shell-quoted).
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 2  # rocm720, rocm714
    assert all(line.startswith("podman build ") for line in lines)
    rocm720_line = lines[0]
    assert "Dockerfile.base" in rocm720_line
    assert "--build-arg BASE_IMAGE=rocm/dev-ubuntu-22.04:7.2-complete" in rocm720_line
    assert "--build-arg ROCM_VERSION=7.2.0" in rocm720_line
    assert "--build-arg PYTHON_VERSION=3.12" in rocm720_line
    assert "--build-arg PYTORCH_VERSION=2.12.0" in rocm720_line
    assert "--build-arg TRITON_VERSION=3.5.1" in rocm720_line
    assert "--build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2" in rocm720_line
    assert "PYTORCH_ROCM_ARCH=gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201" in rocm720_line
    assert "--build-arg FLASH_ATTENTION_INSTALL=base" in rocm720_line
    assert "--build-arg FLASH_ATTENTION_VERSION=2.8.4" in rocm720_line
    assert "--build-arg FLASH_ATTENTION_REPO=https://github.com/ROCm/flash-attention.git" in rocm720_line
    assert "--build-arg FLASH_ATTENTION_REF=tridao" in rocm720_line
    assert "--tag docker.io/blivioniag/rocm-rdna:7.2.0" in rocm720_line
    # Dry-run: status on stderr, no engine invocation.
    assert "dry-run" in result.stderr


def test_build_base_single_base_rocm714_dry_run() -> None:
    result = _run_cli(
        "build-base",
        "--config", str(EXAMPLE_CONFIG),
        "--engine=podman",
        "--dry-run",
        "--base", "rocm714",
    )
    assert result.returncode == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    line = lines[0]
    assert "--build-arg BASE_IMAGE=rocm/dev-ubuntu-22.04:7.14.0-full" in line
    assert "--build-arg BASE_TAG=7.14.0" in line
    assert "--build-arg ROCM_VERSION=7.14.0" in line
    assert "--build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.14" in line
    assert "--build-arg FLASH_ATTENTION_INSTALL=base" in line
    assert "--build-arg FLASH_ATTENTION_VERSION=2.7.4" in line
    assert "--build-arg FLASH_ATTENTION_REPO=https://github.com/Dao-AILab/flash-attention" in line
    assert "--build-arg FLASH_ATTENTION_REF=main" in line
    assert "--tag docker.io/blivioniag/rocm-rdna:7.14.0" in line


def test_build_base_unknown_base_exits_2_named_error() -> None:
    result = _run_cli(
        "build-base",
        "--config", str(EXAMPLE_CONFIG),
        "--engine=podman",
        "--dry-run",
        "--base", "rocm999",
    )
    assert result.returncode == 2
    assert "UnknownBase" in result.stderr
    assert "rocm999" in result.stderr
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# build-vllm dry-run rendering
# ---------------------------------------------------------------------------


def test_build_vllm_extras_rocm714_dry_run() -> None:
    result = _run_cli(
        "build-vllm",
        "--config", str(EXAMPLE_CONFIG),
        "--engine=podman",
        "--dry-run",
        "--image", "vllm-026-extras-rocm714",
    )
    assert result.returncode == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("podman build ")
    assert "Dockerfile.vllm" in line
    assert "--build-arg BASE_IMAGE=docker.io/blivioniag/rocm-rdna:7.14.0" in line
    assert "--build-arg VLLM_REPOSITORY=https://github.com/blivioniag/vllm.git" in line
    assert f"--build-arg VLLM_COMMIT={GOOD_COMMIT_2}" in line
    assert "--build-arg VLLM_REF=v0.26.0-extras" in line
    assert "--build-arg VLLM_VARIANT=extras-fork" in line
    assert "--build-arg TORCH_BACKEND=rocm7.14" in line
    assert "--build-arg FLASH_ATTENTION_INSTALL=none" in line
    assert "--build-arg IMAGE_TAG=v0.26.0-extras-rocm7.14.0" in line
    assert "--tag docker.io/blivioniag/vllm-rdna:v0.26.0-extras-rocm7.14.0" in line


def test_build_vllm_upstream_rocm720_torch_backend_dry_run() -> None:
    result = _run_cli(
        "build-vllm",
        "--config", str(EXAMPLE_CONFIG),
        "--engine=podman",
        "--dry-run",
        "--image", "vllm-026-upstream-rocm720",
    )
    assert result.returncode == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    line = lines[0]
    assert "--build-arg TORCH_BACKEND=rocm7.2" in line
    assert "--build-arg FLASH_ATTENTION_INSTALL=none" in line


def test_build_vllm_unknown_image_exits_2_named_error() -> None:
    result = _run_cli(
        "build-vllm",
        "--config", str(EXAMPLE_CONFIG),
        "--engine=podman",
        "--dry-run",
        "--image", "no-such-image",
    )
    assert result.returncode == 2
    assert "UnknownImage" in result.stderr
    assert result.stdout == ""


# ---------------------------------------------------------------------------
# Deterministic rendering: Podman vs Docker (library-level, no host engines)
# ---------------------------------------------------------------------------


def test_podman_docker_rendering_differs_only_in_executable_and_pull() -> None:
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    base = next(b for b in resolved.base_records if b.id == "rocm720")
    args = base_build_args(base, resolved.architecture_list)
    common = dict(
        dockerfile="vllm-rdna-docker/Dockerfile.base",
        tag="docker.io/blivioniag/rocm-rdna:7.2.0",
        args=args,
        context="vllm-rdna-docker",
    )
    podman_argv = render_build_argv("podman", **common)
    docker_argv = render_build_argv("docker", **common)

    # Executable name.
    assert podman_argv[0] == "podman"
    assert docker_argv[0] == "docker"
    # Docker-only flag.
    assert "--pull" not in podman_argv
    assert "--pull" in docker_argv
    # Everything else is byte-identical.
    podman_normalized = podman_argv[1:]
    docker_normalized = [tok for tok in docker_argv[1:] if tok != "--pull"]
    assert podman_normalized == docker_normalized
    # In particular the full ARG lists are identical.
    assert _build_args_of(podman_argv) == _build_args_of(docker_argv)


def test_vllm_rendering_arg_lists_identical_across_engines() -> None:
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    image = next(i for i in resolved.image_records if i.id == "vllm-026-extras-rocm714")
    args = vllm_build_args(
        image,
        "docker.io/blivioniag/rocm-rdna:7.14.0",
        resolved.config_hash,
        resolved.architecture_list,
    )
    common = dict(
        dockerfile="vllm-rdna-docker/Dockerfile.vllm",
        tag="docker.io/blivioniag/vllm-rdna:v0.26.0-extras-rocm7.14.0",
        args=args,
        context="vllm-rdna-docker",
    )
    podman_argv = render_build_argv("podman", **common)
    docker_argv = render_build_argv("docker", **common)
    assert _build_args_of(podman_argv) == _build_args_of(docker_argv)
    arg_values = _build_args_of(podman_argv)
    assert f"VLLM_COMMIT={GOOD_COMMIT_2}" in arg_values
    assert "VLLM_VARIANT=extras-fork" in arg_values
    assert "IMAGE_TAG=v0.26.0-extras-rocm7.14.0" in arg_values
    assert f"CONFIG_HASH={resolved.config_hash}" in arg_values


def test_rendering_is_deterministic_across_calls() -> None:
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    base = resolved.base_records[0]
    args = base_build_args(base, resolved.architecture_list)
    first = render_build_argv("podman", "df", "tag:1", args, ".")
    second = render_build_argv("podman", "df", "tag:1", args, ".")
    assert first == second


# ---------------------------------------------------------------------------
# Engine selection
# ---------------------------------------------------------------------------


def test_auto_prefers_podman(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("podman", "docker") else None
    )
    assert select_engine("auto") == "podman"


def test_auto_falls_back_to_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None
    )
    assert select_engine("auto") == "docker"


def test_auto_raises_when_neither_engine_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(EngineNotFound) as excinfo:
        select_engine("auto")
    assert excinfo.value.context == {"engine": "auto"}
    assert str(excinfo.value) == "EngineNotFound(engine='auto')"


def test_forced_podman_missing_raises_engine_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(EngineNotFound) as excinfo:
        select_engine("podman")
    assert excinfo.value.context == {"engine": "podman"}


def test_forced_docker_missing_raises_engine_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(EngineNotFound):
        select_engine("docker")


def test_unknown_engine_name_raises() -> None:
    with pytest.raises(EngineNotFound) as excinfo:
        select_engine("containerd")
    assert excinfo.value.context == {"engine": "containerd"}


def test_cli_engine_not_found_exits_1_named_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--engine=podman with podman missing (simulated) fails before any build."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    exit_code = main(
        ["build-base", "--config", str(EXAMPLE_CONFIG), "--engine=podman", "--dry-run"]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "EngineNotFound" in captured.err
    assert captured.out == ""  # no command rendered


def test_cli_container_engine_env_missing_binary() -> None:
    """CONTAINER_ENGINE=missing is an actionable non-zero failure, even dry-run."""
    result = _run_cli(
        "build-base", "--config", str(EXAMPLE_CONFIG), "--dry-run",
        env={ENGINE_ENV_VAR: "missing"},
    )
    assert result.returncode == 1
    assert "EngineNotFound" in result.stderr
    assert result.stdout == ""


def test_cli_container_engine_env_selects_docker() -> None:
    result = _run_cli(
        "build-base", "--config", str(EXAMPLE_CONFIG), "--dry-run",
        env={ENGINE_ENV_VAR: "docker"},
    )
    assert result.returncode == 0
    assert result.stdout.splitlines()[0].startswith("docker build ")
    assert "--pull" in result.stdout.splitlines()[0]


# ---------------------------------------------------------------------------
# Non-dry-run execution with a stubbed engine
# ---------------------------------------------------------------------------


def test_build_base_executes_rendered_argv_without_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured.append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None)
    monkeypatch.setattr(subprocess, "run", _fake_run)

    exit_code = main(
        ["build-base", "--config", str(EXAMPLE_CONFIG), "--base", "rocm720"]
    )
    assert exit_code == 0
    assert len(captured) == 1
    argv = captured[0]
    assert argv[0] == "podman"
    assert argv[1] == "build"
    arg_values = _build_args_of(argv)
    assert "BASE_IMAGE=rocm/dev-ubuntu-22.04:7.2-complete" in arg_values
    assert "ROCM_VERSION=7.2.0" in arg_values
    assert "BASE_TAG=7.2.0" in arg_values


def test_build_base_engine_failure_raises_command_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _failing_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 125)

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None)
    monkeypatch.setattr(subprocess, "run", _failing_run)

    exit_code = main(
        ["build-base", "--config", str(EXAMPLE_CONFIG), "--base", "rocm720"]
    )
    assert exit_code == 3
    captured = capsys.readouterr()
    assert "CommandFailed" in captured.err
    assert "exit_code=125" in captured.err


def test_dry_run_never_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("dry-run attempted a subprocess call")

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None)
    monkeypatch.setattr(subprocess, "run", _forbidden)
    exit_code = main(
        ["build-base", "--config", str(EXAMPLE_CONFIG), "--engine=podman", "--dry-run"]
    )
    assert exit_code == 0


# ---------------------------------------------------------------------------
# Pipeline gating (real verify/publish/promote implementations)
# ---------------------------------------------------------------------------


def test_placeholders_refuse_invalid_config() -> None:
    """Even the pipeline subcommands gate on validate+resolve."""
    fixture = SUBPROJECT / "tests" / "fixtures" / "invalid-unknown-source.toml"
    for subcommand in ("verify", "publish", "promote"):
        result = _run_cli(subcommand, "--config", str(fixture))
        assert result.returncode == 1, subcommand
        assert "UnknownSource" in result.stderr


# ---------------------------------------------------------------------------
# Named error classes
# ---------------------------------------------------------------------------


def test_cli_errors_render_named_context() -> None:
    err = EngineNotFound(engine="podman")
    assert str(err) == "EngineNotFound(engine='podman')"
    failed = CommandFailed(command="podman build ...", exit_code=2)
    assert str(failed) == "CommandFailed(command='podman build ...', exit_code=2)"
    assert failed.context == {"command": "podman build ...", "exit_code": 2}
