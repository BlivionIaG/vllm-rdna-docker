#!/usr/bin/env python3
"""Strict TOML configuration validator for vllm-rdna-docker.

Enforces the schema documented in ``vllm-rdna-docker/config/schema.md``.

Requirements: Python >= 3.11 (stdlib ``tomllib`` only; no third-party deps).

Properties:
  * read-only — never writes files, never touches the network, never invokes
    a container engine;
  * strict — unknown sections/fields are rejected, not ignored;
  * named errors — every failure raises a ``ConfigError`` subclass whose
    ``str()`` renders ``Name(field=value, ...)`` and whose ``context`` dict
    carries the structured fields.

Usage (CLI):
    python vllm-rdna-docker/tools/validate.py --config <path.toml>

Usage (library):
    from tools.validate import load_config, validate_config, ConfigError
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# All named errors live in tools.errors and are re-exported here so existing
# ``from tools.validate import ConfigError, ReservedTag, ...`` keeps working.
try:  # pytest / package mode (vllm-rdna-docker/ on sys.path)
    from tools.errors import *  # noqa: F401,F403
except ModuleNotFoundError:  # script mode: python tools/validate.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.errors import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# Schema constants (see config/schema.md — keep both in sync in one change)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

#: Fixed project-wide RDNA target set. There is intentionally no "primary"
#: architecture concept; see config/schema.md "Design invariants".
REQUIRED_ARCHITECTURES: tuple[str, ...] = (
    "gfx1030",
    "gfx1100",
    "gfx1101",
    "gfx1150",
    "gfx1151",
    "gfx1200",
    "gfx1201",
)

ALLOWED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "project",
        "architecture",
        "registry",
        "bases",
        "sources",
        "images",
        "aliases",
        "patches",
        "artifacts",
    }
)

REQUIRED_SECTIONS: tuple[str, ...] = (
    "project",
    "architecture",
    "registry",
    "bases",
    "sources",
    "images",
)

ALLOWED_PROJECT_FIELDS: frozenset[str] = frozenset(
    {"name", "schema_version", "description", "reserved_tags"}
)
ALLOWED_ARCHITECTURE_FIELDS: frozenset[str] = frozenset({"targets"})
ALLOWED_REGISTRY_FIELDS: frozenset[str] = frozenset(
    {"host", "base_repository", "vllm_repository"}
)
ALLOWED_BASE_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "rocm_version",
        "python_version",
        "pytorch_version",
        "triton_version",
        "base_image",
        "base_digest",
        "pytorch_index_url",
        "tag",
        "patches",
        "flash_attention",
    }
)
ALLOWED_FLASH_ATTENTION_FIELDS: frozenset[str] = frozenset(
    {"install", "version", "repo", "ref"}
)
ALLOWED_FLASH_ATTENTION_INSTALLS: frozenset[str] = frozenset(
    {"base", "vllm", "none"}
)
ALLOWED_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "variant",
        "repository",
        "version",
        "ref",
        "commit",
        "compatible_bases",
    }
)
ALLOWED_IMAGE_FIELDS: frozenset[str] = frozenset({"id", "base", "source", "tag"})
ALLOWED_PATCH_FIELDS: frozenset[str] = frozenset({"id", "sha256", "description", "base"})
ALLOWED_ARTIFACT_FIELDS: frozenset[str] = frozenset({"id", "url", "sha256"})

ALLOWED_VARIANTS: frozenset[str] = frozenset({"upstream", "extras-fork"})

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Absolute host paths are forbidden: the build machine is external and
#: nothing in a config may depend on host state.
HOST_PATH_PREFIXES: tuple[str, ...] = ("/home/", "/root/", "/Users/")

#: Key forbidden at every nesting level (case-insensitive).
FORBIDDEN_KEY = "primary"



# ---------------------------------------------------------------------------
# Validation summary (returned on success)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigSummary:
    """Structured view of a validated config, for tests and later todos."""

    architectures: tuple[str, ...]
    base_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    base_images: Mapping[str, str] = field(default_factory=dict)
    image_tags: Mapping[str, str] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)
    reserved_tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _check_known_fields(
    section: str, table: Mapping[str, Any], allowed: frozenset[str]
) -> None:
    for key in table:
        if key not in allowed:
            raise UnknownField(section, key)


def _require_str(section: str, table: Mapping[str, Any], field_: str) -> str:
    if field_ not in table:
        raise MissingField(section, field_)
    value = table[field_]
    if type(value) is not str:
        raise InvalidFieldType(section, field_, "string")
    if not value.strip():
        raise MissingField(section, field_)
    return value


def _optional_str(
    section: str, table: Mapping[str, Any], field_: str
) -> str | None:
    if field_ not in table:
        return None
    return _require_str(section, table, field_)


def _require_str_list(
    section: str,
    table: Mapping[str, Any],
    field_: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if field_ not in table:
        raise MissingField(section, field_)
    value = table[field_]
    if not isinstance(value, list) or any(type(v) is not str for v in value):
        raise InvalidFieldType(section, field_, "array<string>")
    if not allow_empty and not value:
        raise MissingField(section, field_)
    if any(not v.strip() for v in value):
        raise MissingField(section, field_)
    return list(value)


def _check_not_host_path(section: str, field_: str, value: str) -> None:
    stripped = value.strip()
    if stripped.startswith(HOST_PATH_PREFIXES):
        raise HostPathNotAllowed(section, field_, value)


# ---------------------------------------------------------------------------
# Primary-key scan (any nesting level, case-insensitive)
# ---------------------------------------------------------------------------


def _scan_forbidden_keys(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and key.lower() == FORBIDDEN_KEY:
                raise PrimaryArchitectureNotAllowed(path=child_path)
            _scan_forbidden_keys(value, child_path)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _scan_forbidden_keys(item, f"{path}[{index}]")


# ---------------------------------------------------------------------------
# Section validators
# ---------------------------------------------------------------------------


def _validate_project(data: Mapping[str, Any]) -> tuple[str, ...]:
    section = "project"
    table = data[section]
    if not isinstance(table, dict):
        raise InvalidFieldType(section, section, "table")
    _check_known_fields(section, table, ALLOWED_PROJECT_FIELDS)
    _require_str(section, table, "name")
    if "schema_version" not in table:
        raise MissingField(section, "schema_version")
    version = table["schema_version"]
    if type(version) is not int:
        raise InvalidFieldType(section, "schema_version", "integer")
    if version != SCHEMA_VERSION:
        raise InvalidSchemaVersion(version)
    _optional_str(section, table, "description")
    reserved: list[str] = []
    if "reserved_tags" in table:
        reserved = _require_str_list(section, table, "reserved_tags", allow_empty=True)
    return tuple(reserved)


def _validate_architecture(data: Mapping[str, Any]) -> tuple[str, ...]:
    section = "architecture"
    table = data[section]
    if not isinstance(table, dict):
        raise InvalidFieldType(section, section, "table")
    _check_known_fields(section, table, ALLOWED_ARCHITECTURE_FIELDS)
    targets = _require_str_list(section, table, "targets")

    seen: set[str] = set()
    for arch in targets:
        if arch in seen:
            raise DuplicateArchitecture(arch)
        seen.add(arch)

    required = set(REQUIRED_ARCHITECTURES)
    for arch in sorted(seen - required):
        raise UnexpectedArchitecture(arch)
    for arch in sorted(required - seen):
        raise MissingArchitecture(arch)
    return tuple(targets)


def _validate_registry(data: Mapping[str, Any]) -> None:
    section = "registry"
    table = data[section]
    if not isinstance(table, dict):
        raise InvalidFieldType(section, section, "table")
    _check_known_fields(section, table, ALLOWED_REGISTRY_FIELDS)
    for f in ("host", "base_repository", "vllm_repository"):
        _require_str(section, table, f)


def _validate_bases(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    section = "bases"
    tables = data[section]
    if not isinstance(tables, dict):
        raise InvalidFieldType(section, section, "table")
    if not tables:
        raise EmptySection(section)

    bases: dict[str, dict[str, Any]] = {}
    for key, table in tables.items():
        entry_section = f"bases.{key}"
        if not isinstance(table, dict):
            raise InvalidFieldType(entry_section, entry_section, "table")
        _check_known_fields(entry_section, table, ALLOWED_BASE_FIELDS)
        base_id = _require_str(entry_section, table, "id")
        if base_id in bases:
            raise DuplicateBaseId(base_id)
        for f in (
            "rocm_version",
            "python_version",
            "pytorch_version",
            "triton_version",
            "base_image",
            "pytorch_index_url",
            "tag",
        ):
            _require_str(entry_section, table, f)
        _check_not_host_path(entry_section, "base_image", table["base_image"])
        _check_not_host_path(
            entry_section, "pytorch_index_url", table["pytorch_index_url"]
        )
        digest = _require_str(entry_section, table, "base_digest")
        if not DIGEST_RE.match(digest):
            raise InvalidDigest(base_id, digest)
        patches: list[str] = []
        if "patches" in table:
            patches = _require_str_list(
                entry_section, table, "patches", allow_empty=True
            )
        if "flash_attention" in table:
            _validate_flash_attention(
                base_id, entry_section, table["flash_attention"]
            )
        record = dict(table)
        record["patches"] = patches
        bases[base_id] = record
    return bases


def _validate_flash_attention(
    base_id: str, entry_section: str, table: Any
) -> None:
    section = f"{entry_section}.flash_attention"
    if not isinstance(table, dict):
        raise InvalidFieldType(section, section, "table")
    _check_known_fields(section, table, ALLOWED_FLASH_ATTENTION_FIELDS)
    install = _require_str(section, table, "install")
    if install not in ALLOWED_FLASH_ATTENTION_INSTALLS:
        raise InvalidFlashAttentionInstall(base_id, install)
    if install != "none":
        _require_str(section, table, "version")
    else:
        if "version" in table:
            _require_str(section, table, "version")
    if "repo" in table:
        repo = _require_str(section, table, "repo")
        _check_not_host_path(section, "repo", repo)
    if "ref" in table:
        _require_str(section, table, "ref")


def _validate_sources(
    data: Mapping[str, Any], base_ids: set[str]
) -> dict[str, dict[str, Any]]:
    section = "sources"
    tables = data[section]
    if not isinstance(tables, dict):
        raise InvalidFieldType(section, section, "table")
    if not tables:
        raise EmptySection(section)

    sources: dict[str, dict[str, Any]] = {}
    for key, table in tables.items():
        entry_section = f"sources.{key}"
        if not isinstance(table, dict):
            raise InvalidFieldType(entry_section, entry_section, "table")
        _check_known_fields(entry_section, table, ALLOWED_SOURCE_FIELDS)
        source_id = _require_str(entry_section, table, "id")
        if source_id in sources:
            raise DuplicateSourceId(source_id)
        variant = _require_str(entry_section, table, "variant")
        if variant not in ALLOWED_VARIANTS:
            raise InvalidVariant(source_id, variant)
        repository = _require_str(entry_section, table, "repository")
        _check_not_host_path(entry_section, "repository", repository)
        _require_str(entry_section, table, "version")
        _require_str(entry_section, table, "ref")
        commit = _require_str(entry_section, table, "commit")
        if not COMMIT_RE.match(commit):
            raise InvalidCommit(source_id, commit)
        compatible = _require_str_list(entry_section, table, "compatible_bases")
        for base_id in compatible:
            if base_id not in base_ids:
                raise UnknownBase(base_id)
        sources[source_id] = dict(table)
    return sources


def _validate_patches(data: Mapping[str, Any], base_ids: set[str]) -> set[str]:
    tables = data.get("patches", {})
    if not isinstance(tables, dict):
        raise InvalidFieldType("patches", "patches", "table")
    patch_ids: set[str] = set()
    for key, table in tables.items():
        entry_section = f"patches.{key}"
        if not isinstance(table, dict):
            raise InvalidFieldType(entry_section, entry_section, "table")
        _check_known_fields(entry_section, table, ALLOWED_PATCH_FIELDS)
        patch_id = _require_str(entry_section, table, "id")
        if patch_id in patch_ids:
            raise DuplicatePatchId(patch_id)
        patch_ids.add(patch_id)
        checksum = _require_str(entry_section, table, "sha256")
        if not SHA256_RE.match(checksum):
            raise InvalidChecksum(entry_section, "sha256")
        _optional_str(entry_section, table, "description")
        base_ref = _optional_str(entry_section, table, "base")
        if base_ref is not None and base_ref not in base_ids:
            raise UnknownBase(base_ref)
    return patch_ids


def _validate_artifacts(data: Mapping[str, Any]) -> None:
    tables = data.get("artifacts", {})
    if not isinstance(tables, dict):
        raise InvalidFieldType("artifacts", "artifacts", "table")
    artifact_ids: set[str] = set()
    for key, table in tables.items():
        entry_section = f"artifacts.{key}"
        if not isinstance(table, dict):
            raise InvalidFieldType(entry_section, entry_section, "table")
        _check_known_fields(entry_section, table, ALLOWED_ARTIFACT_FIELDS)
        artifact_id = _require_str(entry_section, table, "id")
        if artifact_id in artifact_ids:
            raise DuplicateArtifactId(artifact_id)
        artifact_ids.add(artifact_id)
        url = _require_str(entry_section, table, "url")
        _check_not_host_path(entry_section, "url", url)
        checksum = _require_str(entry_section, table, "sha256")
        if not SHA256_RE.match(checksum):
            raise InvalidChecksum(entry_section, "sha256")


def _validate_images(
    data: Mapping[str, Any],
    bases: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    reserved_tags: frozenset[str],
) -> dict[str, dict[str, Any]]:
    section = "images"
    entries = data[section]
    if not isinstance(entries, list):
        raise InvalidFieldType(section, section, "array of tables")
    if not entries:
        raise EmptySection(section)

    images: dict[str, dict[str, Any]] = {}
    tags: set[str] = set()
    for index, entry in enumerate(entries):
        entry_section = f"images[{index}]"
        if not isinstance(entry, dict):
            raise InvalidFieldType(entry_section, entry_section, "table")
        _check_known_fields(entry_section, entry, ALLOWED_IMAGE_FIELDS)
        image_id = _require_str(entry_section, entry, "id")
        if image_id in images:
            raise DuplicateImageId(image_id)
        base_ref = _require_str(entry_section, entry, "base")
        if base_ref not in bases:
            raise UnknownBase(base_ref)
        source_ref = _require_str(entry_section, entry, "source")
        if source_ref not in sources:
            raise UnknownSource(source_ref)
        tag = _require_str(entry_section, entry, "tag")
        if tag in tags:
            raise DuplicateTag(tag)
        if tag in reserved_tags:
            raise ReservedTag(tag)
        compatible = sources[source_ref].get("compatible_bases", [])
        if base_ref not in compatible:
            raise UnapprovedCombination(base=base_ref, source=source_ref)
        tags.add(tag)
        images[image_id] = dict(entry)
    return images


def _validate_aliases(
    data: Mapping[str, Any],
    images: Mapping[str, Mapping[str, Any]],
    reserved_tags: frozenset[str],
) -> dict[str, str]:
    table = data.get("aliases", {})
    if not isinstance(table, dict):
        raise InvalidFieldType("aliases", "aliases", "table")
    aliases: dict[str, str] = {}
    for alias, target in table.items():
        if type(target) is not str or not target.strip():
            raise InvalidFieldType("aliases", alias, "string (image id)")
        if alias in reserved_tags:
            raise ReservedTag(alias)
        if target not in images:
            raise UnknownImage(target)
        if images[target].get("tag") != alias:
            raise AliasTagMismatch(alias=alias, image=target)
        aliases[alias] = target
    return aliases


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def validate_config(data: Mapping[str, Any]) -> ConfigSummary:
    """Validate a parsed TOML document against the strict schema.

    Raises a ``ConfigError`` subclass on the first violation found.
    Returns a ``ConfigSummary`` on success. Read-only; no side effects.
    """
    if not isinstance(data, Mapping):
        raise InvalidFieldType("<root>", "<root>", "table")

    # Invariant: no "primary" key at any nesting level, checked before any
    # semantic rule so the failure is never masked by another error.
    _scan_forbidden_keys(data)

    for section in data:
        if section not in ALLOWED_TOP_LEVEL:
            raise UnknownSection(section)
    for section in REQUIRED_SECTIONS:
        if section not in data:
            raise MissingSection(section)

    reserved_tags = frozenset(_validate_project(data))
    architectures = _validate_architecture(data)
    _validate_registry(data)
    bases = _validate_bases(data)
    base_ids = set(bases)
    patch_ids = _validate_patches(data, base_ids)
    for base_id, record in bases.items():
        for patch_ref in record.get("patches", []):
            if patch_ref not in patch_ids:
                raise UnknownPatch(patch_ref)
    _validate_artifacts(data)
    sources = _validate_sources(data, base_ids)
    images = _validate_images(data, bases, sources, reserved_tags)
    aliases = _validate_aliases(data, images, reserved_tags)

    return ConfigSummary(
        architectures=architectures,
        base_ids=tuple(bases),
        source_ids=tuple(sources),
        image_ids=tuple(images),
        base_images={base_id: b["base_image"] for base_id, b in bases.items()},
        image_tags={img_id: img["tag"] for img_id, img in images.items()},
        aliases=aliases,
        reserved_tags=tuple(sorted(reserved_tags)),
    )


def load_config(path: str | Path) -> ConfigSummary:
    """Parse and validate a TOML config file. Read-only; no side effects."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise MalformedTOML(str(exc)) from exc
    return validate_config(data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="validate.py",
        description=(
            "Validate a vllm-rdna-docker TOML config against the strict "
            "schema (config/schema.md). Read-only; no side effects."
        ),
    )
    parser.add_argument("--config", required=True, help="path to the TOML config")
    args = parser.parse_args(argv)

    try:
        summary = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot read config: {exc}", file=sys.stderr)
        return 2

    print(f"OK: {args.config}")
    print(f"  architectures : {', '.join(summary.architectures)}")
    print(f"  bases         : {', '.join(summary.base_ids)}")
    print(f"  sources       : {', '.join(summary.source_ids)}")
    print(f"  images        : {', '.join(summary.image_ids)}")
    if summary.aliases:
        rendered = ", ".join(f"{k} -> {v}" for k, v in summary.aliases.items())
        print(f"  aliases       : {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
