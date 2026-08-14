"""Tests for the vllm-rdna-docker strict TOML configuration schema.

Covers:
  * happy path — the example config and the valid fixture both validate;
  * every required failure mode — each invalid fixture raises its named error
    with the expected structured context;
  * the CLI — exit codes, named error on stderr, and zero side effects;
  * inline-schema rules that have no dedicated fixture file.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tools.validate import (
    AliasTagMismatch,
    ConfigError,
    ConfigSummary,
    DuplicateArchitecture,
    DuplicateSourceId,
    DuplicateTag,
    HostPathNotAllowed,
    InvalidCommit,
    InvalidDigest,
    InvalidFlashAttentionInstall,
    InvalidVariant,
    MalformedTOML,
    MissingArchitecture,
    MissingField,
    PrimaryArchitectureNotAllowed,
    REQUIRED_ARCHITECTURES,
    ReservedTag,
    UnapprovedCombination,
    UnexpectedArchitecture,
    UnknownBase,
    UnknownField,
    UnknownImage,
    UnknownSection,
    UnknownSource,
    load_config,
    validate_config,
)

SUBPROJECT = Path(__file__).resolve().parent.parent
FIXTURES = SUBPROJECT / "tests" / "fixtures"
EXAMPLE_CONFIG = SUBPROJECT / "config" / "rocm-7.2.0.toml"
SCHEMA_DOC = SUBPROJECT / "config" / "schema.md"
VALIDATE_PY = SUBPROJECT / "tools" / "validate.py"

#: fixture name -> (expected error type, expected context subset)
INVALID_FIXTURES = [
    ("invalid-unknown-source.toml", UnknownSource, {"source": "extras027"}),
    (
        "invalid-duplicate-source-id.toml",
        DuplicateSourceId,
        {"source": "upstream026"},
    ),
    ("invalid-missing-arch.toml", MissingArchitecture, {"arch": "gfx1150"}),
    (
        "invalid-has-primary.toml",
        PrimaryArchitectureNotAllowed,
        {"path": "architecture.primary"},
    ),
    (
        "invalid-has-primary-nested.toml",
        PrimaryArchitectureNotAllowed,
        {"path": "bases.rocm720.primary"},
    ),
    (
        "invalid-unapproved-combination.toml",
        UnapprovedCombination,
        {"base": "rocm722", "source": "extras026"},
    ),
]

MINIMAL_VALID = """
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
base_image = "rocm/dev-ubuntu-22.04:7.2.0-complete"
base_digest = "sha256:{digest}"
pytorch_index_url = "https://download.pytorch.org/whl/rocm7.2"
tag = "7.2.0"

[sources.upstream026]
id = "upstream026"
variant = "upstream"
repository = "https://github.com/vllm-project/vllm.git"
version = "0.26.0"
ref = "v0.26.0"
commit = "{commit}"
compatible_bases = ["rocm720"]

[[images]]
id = "vllm-026-upstream-rocm720"
base = "rocm720"
source = "upstream026"
tag = "v0.26.0"
"""

GOOD_DIGEST = "a1b2c3d4e5f60718293a4b5c6d7e8f900a1b2c3d4e5f60718293a4b5c6d7e8f9"
GOOD_COMMIT = "4a7c2e91d0b8536f1c9e2d4a6b8c0e2f4a6c8e0b"


def _parse(template: str = MINIMAL_VALID, **fmt: str) -> dict:
    values = {"digest": GOOD_DIGEST, "commit": GOOD_COMMIT}
    values.update(fmt)
    return tomllib.loads(template.format(**values))


def _run_cli(config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATE_PY), "--config", str(config)],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_fixture_passes() -> None:
    summary = load_config(FIXTURES / "valid-config.toml")
    assert isinstance(summary, ConfigSummary)
    assert summary.architectures == REQUIRED_ARCHITECTURES
    assert summary.base_ids == ("rocm720",)
    assert summary.source_ids == ("upstream026", "extras026")
    assert summary.image_ids == (
        "vllm-026-upstream-rocm720",
        "vllm-026-extras-rocm720",
    )
    assert summary.aliases == {
        "v0.26.0": "vllm-026-upstream-rocm720",
        "v0.26.0-extras": "vllm-026-extras-rocm720",
    }


def test_example_config_passes_and_has_expected_shape() -> None:
    summary = load_config(EXAMPLE_CONFIG)
    assert set(summary.architectures) == set(REQUIRED_ARCHITECTURES)
    assert len(summary.architectures) == 7
    assert set(summary.base_ids) == {"rocm720", "rocm714"}
    assert set(summary.source_ids) == {"upstream026", "extras026"}
    assert set(summary.image_ids) == {
        "vllm-026-upstream-rocm720",
        "vllm-026-extras-rocm720",
        "vllm-026-upstream-rocm714",
        "vllm-026-extras-rocm714",
    }
    assert summary.image_tags["vllm-026-upstream-rocm720"] == "v0.26.0"
    assert summary.image_tags["vllm-026-extras-rocm720"] == "v0.26.0-extras"
    assert summary.image_tags["vllm-026-upstream-rocm714"] == "v0.26.0-rocm7.14.0"
    assert summary.aliases == {
        "v0.26.0": "vllm-026-upstream-rocm720",
        "v0.26.0-extras": "vllm-026-extras-rocm720",
    }
    assert set(summary.reserved_tags) == {"v0.22.1", "v0.22.1_base"}


def test_per_base_base_image_preserved_on_summary() -> None:
    summary = load_config(EXAMPLE_CONFIG)
    assert summary.base_images["rocm720"] == "rocm/dev-ubuntu-22.04:7.2-complete"
    assert summary.base_images["rocm714"] == "rocm/dev-ubuntu-22.04:7.14.0-full"
    assert len(set(summary.base_images.values())) == 2


def test_example_config_has_no_primary_key_anywhere() -> None:
    data = tomllib.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert key.lower() != "primary", f"primary key found at {key}"
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)


def test_schema_doc_declares_minimum_python() -> None:
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    assert "Minimum Python: 3.11" in text
    assert "tomllib" in text


# ---------------------------------------------------------------------------
# Required failure modes (fixture-driven)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "error_type", "expected_context"),
    INVALID_FIXTURES,
    ids=[name for name, _, _ in INVALID_FIXTURES],
)
def test_invalid_fixtures_raise_named_errors(
    fixture: str,
    error_type: type[ConfigError],
    expected_context: dict,
) -> None:
    with pytest.raises(error_type) as excinfo:
        load_config(FIXTURES / fixture)
    err = excinfo.value
    assert err.name == error_type.__name__
    for key, value in expected_context.items():
        assert err.context.get(key) == value
    # str() renders Name(field=value, ...) so logs are self-describing.
    assert str(err).startswith(f"{error_type.__name__}(")
    for value in expected_context.values():
        assert repr(value) in str(err)


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------


def test_cli_accepts_example_config() -> None:
    result = _run_cli(EXAMPLE_CONFIG)
    assert result.returncode == 0
    assert "OK:" in result.stdout
    assert "vllm-026-upstream-rocm720" in result.stdout


def test_cli_accepts_valid_fixture() -> None:
    result = _run_cli(FIXTURES / "valid-config.toml")
    assert result.returncode == 0
    assert "OK:" in result.stdout


@pytest.mark.parametrize(
    ("fixture", "error_type", "expected_context"),
    INVALID_FIXTURES,
    ids=[name for name, _, _ in INVALID_FIXTURES],
)
def test_cli_rejects_invalid_fixtures_with_named_error(
    fixture: str,
    error_type: type[ConfigError],
    expected_context: dict,
) -> None:
    result = _run_cli(FIXTURES / fixture)
    assert result.returncode == 1
    assert error_type.__name__ in result.stderr
    for value in expected_context.values():
        assert repr(value) in result.stderr
    assert result.stdout == ""


def test_cli_missing_file_exits_nonzero() -> None:
    result = _run_cli(SUBPROJECT / "config" / "does-not-exist.toml")
    assert result.returncode != 0
    assert "error" in result.stderr.lower()


def test_cli_produces_no_side_effects() -> None:
    def snapshot() -> dict[str, tuple[int, int]]:
        snap: dict[str, tuple[int, int]] = {}
        for path in sorted(SUBPROJECT.rglob("*")):
            if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            if path.is_file():
                stat = path.stat()
                snap[str(path.relative_to(SUBPROJECT))] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                )
        return snap

    before = snapshot()
    for config in [EXAMPLE_CONFIG, *sorted(FIXTURES.glob("*.toml"))]:
        _run_cli(config)
    assert snapshot() == before


# ---------------------------------------------------------------------------
# Inline-schema rules (no dedicated fixture file required)
# ---------------------------------------------------------------------------


def test_malformed_toml(tmp_path: Path) -> None:
    bad = tmp_path / "malformed.toml"
    bad.write_text("[project\nname = 'x'\n", encoding="utf-8")
    with pytest.raises(MalformedTOML):
        load_config(bad)


def test_unknown_top_level_section() -> None:
    data = _parse()
    data["mystery"] = {"x": 1}
    with pytest.raises(UnknownSection) as excinfo:
        validate_config(data)
    assert excinfo.value.context["section"] == "mystery"


def test_unknown_field_in_section() -> None:
    data = _parse()
    data["registry"]["bogus"] = "nope"
    with pytest.raises(UnknownField) as excinfo:
        validate_config(data)
    assert excinfo.value.context["field"] == "bogus"


def test_missing_required_field() -> None:
    data = _parse()
    del data["sources"]["upstream026"]["commit"]
    with pytest.raises(MissingField) as excinfo:
        validate_config(data)
    assert excinfo.value.context["field"] == "commit"


def test_unexpected_architecture() -> None:
    data = _parse()
    data["architecture"]["targets"].append("gfx942")
    with pytest.raises(UnexpectedArchitecture) as excinfo:
        validate_config(data)
    assert excinfo.value.context["arch"] == "gfx942"


def test_duplicate_architecture() -> None:
    data = _parse()
    data["architecture"]["targets"].append("gfx1030")
    with pytest.raises(DuplicateArchitecture) as excinfo:
        validate_config(data)
    assert excinfo.value.context["arch"] == "gfx1030"


def test_invalid_variant() -> None:
    data = _parse()
    data["sources"]["upstream026"]["variant"] = "nightly"
    with pytest.raises(InvalidVariant) as excinfo:
        validate_config(data)
    assert excinfo.value.context["variant"] == "nightly"


def test_invalid_commit_short() -> None:
    data = _parse(commit="abc123")
    with pytest.raises(InvalidCommit):
        validate_config(data)


def test_invalid_digest() -> None:
    data = _parse(digest="zz" + GOOD_DIGEST[2:])
    with pytest.raises(InvalidDigest):
        validate_config(data)


def test_unknown_base_in_image() -> None:
    data = _parse()
    data["images"][0]["base"] = "rocm799"
    with pytest.raises(UnknownBase) as excinfo:
        validate_config(data)
    assert excinfo.value.context["base"] == "rocm799"


def test_compatible_bases_rejects_unknown_base() -> None:
    data = _parse()
    data["sources"]["upstream026"]["compatible_bases"] = ["rocm799"]
    with pytest.raises(UnknownBase) as excinfo:
        validate_config(data)
    assert excinfo.value.context["base"] == "rocm799"


def test_duplicate_tag() -> None:
    data = _parse()
    data["images"].append(
        {
            "id": "vllm-026-upstream-rocm720-copy",
            "base": "rocm720",
            "source": "upstream026",
            "tag": "v0.26.0",
        }
    )
    with pytest.raises(DuplicateTag) as excinfo:
        validate_config(data)
    assert excinfo.value.context["tag"] == "v0.26.0"


def test_reserved_tag_on_image() -> None:
    data = _parse()
    data["images"][0]["tag"] = "v0.22.1"
    with pytest.raises(ReservedTag) as excinfo:
        validate_config(data)
    assert excinfo.value.context["tag"] == "v0.22.1"


def test_reserved_tag_on_alias() -> None:
    data = _parse()
    data["aliases"] = {"v0.22.1": "vllm-026-upstream-rocm720"}
    with pytest.raises(ReservedTag):
        validate_config(data)


def test_alias_to_unknown_image() -> None:
    data = _parse()
    data["aliases"] = {"v0.26.0": "no-such-image"}
    with pytest.raises(UnknownImage) as excinfo:
        validate_config(data)
    assert excinfo.value.context["image"] == "no-such-image"


def test_alias_must_match_image_tag() -> None:
    data = _parse()
    data["aliases"] = {"latest": "vllm-026-upstream-rocm720"}
    with pytest.raises(AliasTagMismatch) as excinfo:
        validate_config(data)
    assert excinfo.value.context["alias"] == "latest"


def test_host_path_rejected_in_repository() -> None:
    data = _parse()
    data["sources"]["upstream026"]["repository"] = "/home/builder/vllm"
    with pytest.raises(HostPathNotAllowed):
        validate_config(data)


# ---------------------------------------------------------------------------
# pytorch_index_url + flash_attention rules
# ---------------------------------------------------------------------------


def test_pytorch_index_url_is_required() -> None:
    data = _parse()
    del data["bases"]["rocm720"]["pytorch_index_url"]
    with pytest.raises(MissingField) as excinfo:
        validate_config(data)
    assert excinfo.value.context["field"] == "pytorch_index_url"


def test_pytorch_index_url_host_path_rejected() -> None:
    data = _parse()
    data["bases"]["rocm720"]["pytorch_index_url"] = "/home/builder/wheels"
    with pytest.raises(HostPathNotAllowed):
        validate_config(data)


def test_flash_attention_valid_sub_table() -> None:
    data = _parse()
    data["bases"]["rocm720"]["flash_attention"] = {
        "install": "base",
        "version": "2.7.4",
        "repo": "https://github.com/Dao-AILab/flash-attention",
        "ref": "main",
    }
    summary = validate_config(data)
    assert "rocm720" in summary.base_ids


def test_flash_attention_invalid_install() -> None:
    data = _parse()
    data["bases"]["rocm720"]["flash_attention"] = {"install": "sideways"}
    with pytest.raises(InvalidFlashAttentionInstall) as excinfo:
        validate_config(data)
    assert excinfo.value.context["install"] == "sideways"


def test_flash_attention_version_required_when_installed() -> None:
    data = _parse()
    data["bases"]["rocm720"]["flash_attention"] = {"install": "base"}
    with pytest.raises(MissingField) as excinfo:
        validate_config(data)
    assert excinfo.value.context["field"] == "version"


def test_flash_attention_none_needs_no_version() -> None:
    data = _parse()
    data["bases"]["rocm720"]["flash_attention"] = {"install": "none"}
    summary = validate_config(data)
    assert "rocm720" in summary.base_ids


def test_flash_attention_repo_host_path_rejected() -> None:
    data = _parse()
    data["bases"]["rocm720"]["flash_attention"] = {
        "install": "base",
        "version": "2.7.4",
        "repo": "/home/builder/flash-attention",
    }
    with pytest.raises(HostPathNotAllowed):
        validate_config(data)


def test_example_config_rocm714_carries_flash_attention() -> None:
    data = tomllib.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    fa = data["bases"]["rocm714"]["flash_attention"]
    assert fa["install"] == "base"
    assert fa["repo"] == "https://github.com/Dao-AILab/flash-attention"


def test_error_str_format() -> None:
    err = UnknownSource("extras027")
    assert str(err) == "UnknownSource(source='extras027')"
    combo = UnapprovedCombination(base="rocm722", source="extras026")
    assert str(combo) == "UnapprovedCombination(base='rocm722', source='extras026')"
    assert err.name == "UnknownSource"
