#!/usr/bin/env python3
"""Immutable data records shared by the validator, resolver, and build CLI.

A single home for every frozen dataclass that crosses a module boundary.
Keeping these out of validate.py / resolve.py means a downstream caller
can ``from tools.records import ResolvedBuild`` without dragging in the
full validator or resolver machinery.

Records defined here:

* :class:`ConfigSummary` — the validated view of a config (no resolution)
* :class:`BaseRecord` — one ``[bases.*]`` entry, flattened
* :class:`SourceRecord` — one ``[sources.*]`` entry plus its commit verdict
* :class:`ImageRecord` — one approved base x source combination, fully linked
* :class:`ResolvedBuild` — the whole resolved build (records + hash)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# ---------------------------------------------------------------------------
# Validator summary
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
# Resolved records (post-resolve)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseRecord:
    """Immutable view of one ``[bases.*]`` entry."""

    id: str
    rocm_version: str
    python_version: str
    pytorch_version: str
    triton_version: str
    base_image: str
    base_digest: str
    pytorch_index_url: str
    tag: str
    patches: tuple[str, ...] = ()
    flash_attention: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceRecord:
    """Immutable view of one ``[sources.*]`` entry plus its commit verdict."""

    id: str
    variant: str
    repository: str
    version: str
    ref: str
    resolved_commit: str
    configured_commit: str
    commit_matches: bool
    compatible_bases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageRecord:
    """One approved base x source combination, fully linked.

    This is the unit consumed by the build CLI: it carries the literal
    ``tag`` / ``qualified_tag`` plus direct references to the resolved base
    and source records, so no downstream step needs to re-join anything.
    """

    id: str
    base_id: str
    source_id: str
    tag: str
    qualified_tag: str
    base_record: BaseRecord
    source_record: SourceRecord


@dataclass(frozen=True)
class ResolvedBuild:
    """Deterministic build description for a whole validated config."""

    architecture_list: tuple[str, ...]
    base_records: tuple[BaseRecord, ...]
    source_records: tuple[SourceRecord, ...]
    image_records: tuple[ImageRecord, ...]
    config_hash: str