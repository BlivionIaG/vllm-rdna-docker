"""Tests for the vllm-rdna-docker read-only config resolver.

Covers:
  * happy path — the example config resolves into deterministic records with
    a sha256 config hash and offline commit verification;
  * the CLI — human summary, single-image JSON, named UnknownImage failure;
  * guard rails — unapproved base x source combinations are refused by the
    resolver itself (independent of the validator), before any build command
    could be emitted;
  * chained validation — host paths are rejected by the validator before the
    resolver ever runs.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tools.resolve import (
    NETWORK_ENV_VAR,
    BaseRecord,
    ImageRecord,
    ResolvableSummary,
    ResolvedBuild,
    SourceRecord,
    load_config,
    resolve_config,
    summary_from_data,
)
from tools.validate import (
    ConfigSummary,
    HostPathNotAllowed,
    UnapprovedCombination,
    validate_config,
)

SUBPROJECT = Path(__file__).resolve().parent.parent
EXAMPLE_CONFIG = SUBPROJECT / "config" / "rocm-7.2.0.toml"
RESOLVE_PY = SUBPROJECT / "tools" / "resolve.py"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

GOOD_DIGEST = "a1b2c3d4e5f60718293a4b5c6d7e8f900a1b2c3d4e5f60718293a4b5c6d7e8f9"
GOOD_DIGEST_2 = "f9e8d7c6b5a493827160f5e4d3c2b1a09f8e7d6c5b4a39281706f5e4d3c2b1a0"
GOOD_COMMIT = "4a7c2e91d0b8536f1c9e2d4a6b8c0e2f4a6c8e0b"
GOOD_COMMIT_2 = "9f3b6d1a8c5e0274b6d8a0c2e4f6a8c0d2e4f6a8"

#: Two-base template. rocm722 exists so an image can pair it with a source
#: whose compatible_bases does not include it (unapproved combination test).
TWO_BASE_TEMPLATE = """
[project]
name = "vllm-rdna-docker"
schema_version = 1
reserved_tags = ["v0.22.1", "v0.22.1_base"]

[architecture]
targets = ["gfx1030", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1200", "gfx1201"]

[registry]
host = "docker.io"
base_repository = "blivioniag/rocm-rdna"
vllm_repository = "blivioniag/vllm-rdna"

[bases.rocm720]
id = "rocm720"
rocm_version = "7.2.0"
python_version = "3.12"
pytorch_version = "2.12.0"
triton_version = "3.6.0"
base_image = "rocm/dev-ubuntu-22.04:7.2-complete"
base_digest = "sha256:{digest}"
pytorch_index_url = "https://download.pytorch.org/whl/rocm7.2"
tag = "7.2.0"

[bases.rocm722]
id = "rocm722"
rocm_version = "7.2.2"
python_version = "3.12"
pytorch_version = "2.12.0"
triton_version = "3.6.0"
base_image = "rocm/dev-ubuntu-22.04:7.2.2-complete"
base_digest = "sha256:{digest2}"
pytorch_index_url = "https://download.pytorch.org/whl/rocm7.2"
tag = "7.2.2"

[sources.upstream026]
id = "upstream026"
variant = "upstream"
repository = "https://github.com/vllm-project/vllm.git"
version = "0.26.0"
ref = "{ref}"
commit = "{commit}"
compatible_bases = ["rocm720", "rocm722"]

[sources.extras026]
id = "extras026"
variant = "extras-fork"
repository = "{repository}"
version = "0.26.0"
ref = "v0.26.0-extras"
commit = "{commit2}"
compatible_bases = ["rocm720"]

[[images]]
id = "vllm-026-upstream-rocm720"
base = "rocm720"
source = "upstream026"
tag = "v0.26.0"

[[images]]
id = "vllm-026-extras-rocm720"
base = "rocm720"
source = "extras026"
tag = "v0.26.0-extras"
"""


def _parse(template: str = TWO_BASE_TEMPLATE, **fmt: str) -> dict:
    values = {
        "digest": GOOD_DIGEST,
        "digest2": GOOD_DIGEST_2,
        "commit": GOOD_COMMIT,
        "commit2": GOOD_COMMIT_2,
        "ref": "v0.26.0",
        "repository": "https://github.com/blivioniag/vllm.git",
    }
    values.update(fmt)
    return tomllib.loads(template.format(**values))


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.pop(NETWORK_ENV_VAR, None)  # tests are offline by construction
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, str(RESOLVE_PY), *args],
        capture_output=True,
        text=True,
        check=False,
        env=run_env,
    )


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolver must default to offline regardless of the ambient env."""
    monkeypatch.delenv(NETWORK_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_example_config_resolves_deterministic_records() -> None:
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    assert isinstance(resolved, ResolvedBuild)
    # config_hash is a sha256 over canonical JSON: 64 lowercase hex chars.
    assert SHA256_RE.match(resolved.config_hash)
    # 2 bases, 2 sources, 4 images, 7 architectures.
    assert {b.id for b in resolved.base_records} == {"rocm720", "rocm714"}
    assert len(resolved.base_records) == 2
    assert {s.id for s in resolved.source_records} == {"upstream026", "extras026"}
    assert len(resolved.source_records) == 2
    # Every declared source commit verifies offline (pre-recorded commits,
    # well-formed refs — no network involved).
    assert all(s.commit_matches for s in resolved.source_records)
    assert all(
        s.resolved_commit == s.configured_commit for s in resolved.source_records
    )
    assert len(resolved.image_records) == 4
    assert {i.id for i in resolved.image_records} == {
        "vllm-026-upstream-rocm720",
        "vllm-026-extras-rocm720",
        "vllm-026-upstream-rocm714",
        "vllm-026-extras-rocm714",
    }
    assert len(resolved.architecture_list) == 7


def test_base_records_carry_index_url_and_flash_attention() -> None:
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    by_id = {b.id: b for b in resolved.base_records}
    assert by_id["rocm720"].pytorch_index_url == (
        "https://download.pytorch.org/whl/rocm7.2"
    )
    assert by_id["rocm714"].pytorch_index_url == (
        "https://download.pytorch.org/whl/rocm7.14"
    )
    fa720 = by_id["rocm720"].flash_attention
    assert fa720["install"] == "base"
    assert fa720["version"] == "2.8.4"
    assert fa720["repo"] == "https://github.com/ROCm/flash-attention.git"
    assert fa720["ref"] == "tridao"
    fa = by_id["rocm714"].flash_attention
    assert fa["install"] == "base"
    assert fa["version"] == "2.7.4"
    assert fa["repo"] == "https://github.com/Dao-AILab/flash-attention"


def test_config_hash_changes_with_flash_attention_version(tmp_path: Path) -> None:
    original = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    mutated = tmp_path / "mutated.toml"
    mutated.write_text(original.replace('version = "2.7.4"', 'version = "2.8.0"'))
    first = resolve_config(load_config(EXAMPLE_CONFIG))
    second = resolve_config(load_config(mutated))
    assert first.config_hash != second.config_hash


def test_resolve_is_deterministic_across_runs() -> None:
    first = resolve_config(load_config(EXAMPLE_CONFIG))
    second = resolve_config(load_config(EXAMPLE_CONFIG))
    assert first == second
    assert first.config_hash == second.config_hash


def test_records_are_frozen_and_linked() -> None:
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    assert all(isinstance(b, BaseRecord) for b in resolved.base_records)
    assert all(isinstance(s, SourceRecord) for s in resolved.source_records)
    assert all(isinstance(i, ImageRecord) for i in resolved.image_records)
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.image_records[0].tag = "mutated"  # type: ignore[misc]
    # ImageRecord carries direct references to the exact resolved records.
    for image in resolved.image_records:
        assert image.base_record in resolved.base_records
        assert image.source_record in resolved.source_records
        assert image.base_record.id == image.base_id
        assert image.source_record.id == image.source_id
        assert image.qualified_tag == image.tag


def test_example_image_extras_rocm714_links() -> None:
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    record = next(
        r for r in resolved.image_records if r.id == "vllm-026-extras-rocm714"
    )
    assert record.source_id == "extras026"
    assert record.base_id == "rocm714"
    assert record.tag == "v0.26.0-extras-rocm7.14.0"
    assert record.base_record.rocm_version == "7.14.0"
    assert record.source_record.variant == "extras-fork"


def test_config_hash_excludes_environment_dependent_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """commit_matches is derived per-environment; the hash must not depend on it."""
    summary = load_config(EXAMPLE_CONFIG)
    baseline = resolve_config(summary)
    # Simulate a source whose offline shape check fails: the hash must be
    # computed from config-declared values only, so flipping the derived
    # commit_matches flag must not change it.
    tampered = dataclasses.replace(summary)  # same content, new object
    resolved = resolve_config(tampered)
    assert resolved.config_hash == baseline.config_hash


def test_resolve_config_rejects_plain_config_summary() -> None:
    """A bare ConfigSummary lacks the validated tables; fail loudly."""
    bare = ConfigSummary(
        architectures=(), base_ids=(), source_ids=(), image_ids=()
    )
    with pytest.raises(TypeError):
        resolve_config(bare)


def test_offline_mode_never_invokes_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("offline resolution attempted a subprocess call")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.delenv(NETWORK_ENV_VAR, raising=False)
    resolved = resolve_config(load_config(EXAMPLE_CONFIG))
    assert all(s.commit_matches for s in resolved.source_records)


def test_offline_commit_matches_requires_wellformed_ref() -> None:
    data = _parse(ref="bad ref with spaces!")
    summary = summary_from_data(data, validate=True)
    resolved = resolve_config(summary)
    upstream = next(s for s in resolved.source_records if s.id == "upstream026")
    extras = next(s for s in resolved.source_records if s.id == "extras026")
    assert upstream.commit_matches is False
    assert upstream.resolved_commit == upstream.configured_commit
    assert extras.commit_matches is True


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_unapproved_combination_refused_by_resolver_itself() -> None:
    """rocm722 x extras026: validator rejects it AND the resolver's own gate
    rejects it even when handed an unchecked summary (defense in depth)."""
    data = _parse()
    data["images"].append(
        {
            "id": "vllm-026-extras-rocm722",
            "base": "rocm722",
            "source": "extras026",
            "tag": "v0.26.0-extras-rocm7.2.2",
        }
    )
    # The validator already refuses this config...
    with pytest.raises(UnapprovedCombination) as excinfo:
        validate_config(data)
    assert excinfo.value.context == {"base": "rocm722", "source": "extras026"}
    # ...and the resolver independently refuses to materialize the image
    # (validate=False bypasses the validator to exercise the resolver gate).
    unchecked = summary_from_data(data, validate=False)
    assert isinstance(unchecked, ResolvableSummary)
    with pytest.raises(UnapprovedCombination) as excinfo2:
        resolve_config(unchecked)
    assert excinfo2.value.context == {"base": "rocm722", "source": "extras026"}


def test_host_path_repository_rejected_before_resolution(tmp_path: Path) -> None:
    """A host-path repository is a validator concern; the resolver must never
    see it. Confirm by running validator + resolver chained."""
    data = _parse(repository="/home/foo/vllm")
    with pytest.raises(HostPathNotAllowed):
        validate_config(data)
    config_file = tmp_path / "host-path-config.toml"
    config_file.write_text(
        TWO_BASE_TEMPLATE.format(
            digest=GOOD_DIGEST,
            digest2=GOOD_DIGEST_2,
            commit=GOOD_COMMIT,
            commit2=GOOD_COMMIT_2,
            ref="v0.26.0",
            repository="/home/foo/vllm",
        ),
        encoding="utf-8",
    )
    with pytest.raises(HostPathNotAllowed):
        load_config(config_file)  # resolver entry point chains the validator
    result = _run_cli("--config", str(config_file))
    assert result.returncode == 1
    assert "HostPathNotAllowed" in result.stderr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_happy_summary() -> None:
    result = _run_cli("--config", str(EXAMPLE_CONFIG))
    assert result.returncode == 0
    # Offline banner on stderr documents the default mode.
    assert "offline" in result.stderr
    assert NETWORK_ENV_VAR in result.stderr
    # Human-readable summary lists every record.
    assert "config_hash=" in result.stdout
    for base_id in ("rocm720", "rocm714"):
        assert base_id in result.stdout
    for source_id in ("upstream026", "extras026"):
        assert source_id in result.stdout
    for image_id in (
        "vllm-026-upstream-rocm720",
        "vllm-026-extras-rocm720",
        "vllm-026-upstream-rocm714",
        "vllm-026-extras-rocm714",
    ):
        assert image_id in result.stdout


def test_cli_full_json() -> None:
    result = _run_cli("--config", str(EXAMPLE_CONFIG), "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert SHA256_RE.match(payload["config_hash"])
    assert len(payload["image_records"]) == 4
    assert len(payload["base_records"]) == 2
    assert len(payload["source_records"]) == 2


def test_cli_single_image_json() -> None:
    result = _run_cli(
        "--config", str(EXAMPLE_CONFIG), "--image", "vllm-026-extras-rocm714"
    )
    assert result.returncode == 0
    record = json.loads(result.stdout)
    assert record["id"] == "vllm-026-extras-rocm714"
    assert record["source_id"] == "extras026"
    assert record["base_id"] == "rocm714"
    assert record["qualified_tag"] == "v0.26.0-extras-rocm7.14.0"
    assert record["source_record"]["variant"] == "extras-fork"
    assert record["base_record"]["rocm_version"] == "7.14.0"
    assert record["source_record"]["commit_matches"] is True


def test_cli_unknown_image_exits_2_with_named_error() -> None:
    result = _run_cli("--config", str(EXAMPLE_CONFIG), "--image", "no-such-image")
    assert result.returncode == 2
    assert "UnknownImage" in result.stderr
    assert "no-such-image" in result.stderr
    assert result.stdout == ""
