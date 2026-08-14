#!/usr/bin/env python3
"""Read-only resolver for vllm-rdna-docker validated configs.

Consumes the output of ``tools.validate.validate_config`` and materializes a
deterministic, immutable build description (``ResolvedBuild``) for every
declared base, source, and approved image combination.

Properties:
  * read-only — never writes files, never clones repositories, never invokes
    a container engine;
  * offline by default — commit verification is config-only: a source's
    ``commit_matches`` is ``True`` iff the configured commit is a 40-char
    lowercase sha AND the configured ref is well-formed
    (``^[A-Za-z0-9._/-]+$``). Setting ``RESOLVE_NETWORK=1`` enables an
    OPTIONAL ``git ls-remote`` check that resolves the ref against the remote
    and compares it to the configured commit. The default never touches the
    network;
  * deterministic — the same config always produces the same records and the
    same ``config_hash`` (sha256 over a canonical JSON projection of the
    config-declared values; the environment-dependent ``commit_matches`` flag
    is deliberately excluded from the hash);
  * fail-fast — an image whose base is not in its source's
    ``compatible_bases`` raises ``UnapprovedCombination`` during resolution,
    before any build command could ever be emitted. (The validator already
    rejects such configs; the resolver repeats the check so it is safe even
    when handed an unchecked summary.)

Usage (CLI):
    python vllm-rdna-docker/tools/resolve.py --config <path.toml> [--json] [--image <id>]

Usage (library):
    from tools.resolve import load_config, resolve_config
    resolved = resolve_config(load_config("config/rocm-7.2.0.toml"))
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # pytest / package mode (vllm-rdna-docker/ on sys.path)
    from tools.validate import (
        COMMIT_RE,
        ConfigError,
        ConfigSummary,
        MalformedTOML,
        UnapprovedCombination,
        UnknownBase,
        UnknownImage,
        UnknownSource,
        validate_config,
    )
except ModuleNotFoundError:  # script mode: python tools/resolve.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.validate import (
        COMMIT_RE,
        ConfigError,
        ConfigSummary,
        MalformedTOML,
        UnapprovedCombination,
        UnknownBase,
        UnknownImage,
        UnknownSource,
        validate_config,
    )

# ---------------------------------------------------------------------------
# Resolution constants
# ---------------------------------------------------------------------------

#: Well-formed git ref names accepted for offline commit verification.
REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")

#: Set RESOLVE_NETWORK=1 to enable the optional `git ls-remote` check.
#: Any other value (or unset) keeps the resolver fully offline.
NETWORK_ENV_VAR = "RESOLVE_NETWORK"

#: Timeout for the optional network check, seconds.
LS_REMOTE_TIMEOUT_S = 30

#: Schema tag embedded in the canonical JSON that feeds config_hash.
CANONICAL_SCHEMA = "vllm-rdna-docker/resolved-build/v1"


# ---------------------------------------------------------------------------
# Resolved records
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

    This is the unit consumed by Todo 3+ (Dockerfiles / build CLI): it carries
    the literal ``tag``/``qualified_tag`` plus direct references to the
    resolved base and source records, so no downstream step needs to re-join
    anything.
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


# ---------------------------------------------------------------------------
# ResolvableSummary — ConfigSummary carrying the validated raw tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvableSummary(ConfigSummary):
    """``ConfigSummary`` enriched with the validated bases/sources/images tables.

    ``ConfigSummary`` intentionally exposes only ids/tags, which is not enough
    to build records. This subclass carries the raw (already schema-checked)
    tables alongside it, so ``resolve_config`` never has to re-parse TOML or
    duplicate any schema logic. Produced by ``load_config`` /
    ``summary_from_data`` in this module.
    """

    bases: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    images: tuple[Mapping[str, Any], ...] = ()


def summary_from_data(
    data: Mapping[str, Any], *, validate: bool = True
) -> ResolvableSummary:
    """Build a ``ResolvableSummary`` from a parsed TOML document.

    With ``validate=True`` (the default, and the only mode used by the CLI)
    the document is checked by ``tools.validate.validate_config`` first; any
    violation raises the corresponding named ``ConfigError`` before any
    resolution happens.

    ``validate=False`` skips the validator and is reserved for tests that
    deliberately feed the resolver a broken summary to prove the resolver's
    own guard rails (e.g. ``UnapprovedCombination``) fire independently.
    """
    bases = {t["id"]: t for t in data.get("bases", {}).values()}
    sources = {t["id"]: t for t in data.get("sources", {}).values()}
    images = tuple(data.get("images", []))
    if validate:
        summary = validate_config(data)
        return ResolvableSummary(
            architectures=summary.architectures,
            base_ids=summary.base_ids,
            source_ids=summary.source_ids,
            image_ids=summary.image_ids,
            base_images=dict(summary.base_images),
            image_tags=dict(summary.image_tags),
            aliases=dict(summary.aliases),
            reserved_tags=summary.reserved_tags,
            bases=bases,
            sources=sources,
            images=images,
        )
    # Unchecked path (tests only): derive the summary fields directly.
    return ResolvableSummary(
        architectures=tuple(data.get("architecture", {}).get("targets", ())),
        base_ids=tuple(bases),
        source_ids=tuple(sources),
        image_ids=tuple(i["id"] for i in images),
        base_images={bid: b.get("base_image", "") for bid, b in bases.items()},
        image_tags={i["id"]: i.get("tag", "") for i in images},
        aliases=dict(data.get("aliases", {})),
        reserved_tags=tuple(data.get("project", {}).get("reserved_tags", ())),
        bases=bases,
        sources=sources,
        images=images,
    )


def load_config(path: str | Path) -> ResolvableSummary:
    """Parse, validate, and wrap a TOML config file. Read-only; no side effects."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise MalformedTOML(str(exc)) from exc
    return summary_from_data(data, validate=True)


# ---------------------------------------------------------------------------
# Commit resolution (offline by default)
# ---------------------------------------------------------------------------


def _ls_remote_commit(repository: str, ref: str) -> str | None:
    """Resolve ``ref`` on ``repository`` via ``git ls-remote``.

    Only ever called when RESOLVE_NETWORK=1. Returns the 40-char commit sha
    (peeled, for annotated tags), or ``None`` on any failure — a failed
    network check is a verification failure, never an exception.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-remote", repository, ref, f"{ref}^{{}}"],
            capture_output=True,
            text=True,
            timeout=LS_REMOTE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    # Annotated tags produce both "<sha> refs/tags/<ref>" and the peeled
    # "<sha> refs/tags/<ref>^{}" line; the peeled sha is the real commit.
    peeled = [line for line in lines if line.rstrip().endswith("^{}")]
    chosen = peeled[0] if peeled else lines[0]
    sha = chosen.split()[0] if chosen.split() else None
    if sha and COMMIT_RE.match(sha):
        return sha
    return None


def _resolve_commit(
    repository: str, ref: str, configured_commit: str, *, network: bool
) -> tuple[str, bool]:
    """Return ``(resolved_commit, commit_matches)`` for one source.

    Offline (default): no network access. The configured commit is the
    resolution of record; ``commit_matches`` is True only when the configured
    commit is a 40-char lowercase sha AND the ref is well-formed.

    Network (RESOLVE_NETWORK=1): additionally resolve the ref with
    ``git ls-remote``; ``commit_matches`` requires the remote sha to equal
    the configured commit.
    """
    shapes_ok = bool(COMMIT_RE.match(configured_commit)) and bool(REF_RE.match(ref))
    if not shapes_ok:
        return configured_commit, False
    if not network:
        return configured_commit, True
    remote = _ls_remote_commit(repository, ref)
    if remote is None:
        return configured_commit, False
    return remote, remote == configured_commit


# ---------------------------------------------------------------------------
# Canonical JSON + config hash
# ---------------------------------------------------------------------------


def _canonical_projection(
    architecture_list: tuple[str, ...],
    base_records: tuple[BaseRecord, ...],
    source_records: tuple[SourceRecord, ...],
    image_records: tuple[ImageRecord, ...],
) -> dict[str, Any]:
    """Deterministic projection of config-declared values.

    Excludes derived/environment-dependent fields (``commit_matches``,
    ``resolved_commit`` when it differs from the configured commit) so the
    hash depends only on the config file, never on the environment.
    Entries are sorted by id so TOML declaration order cannot affect the hash.
    """
    return {
        "schema": CANONICAL_SCHEMA,
        "architectures": sorted(architecture_list),
        "bases": [
            {
                "id": b.id,
                "rocm_version": b.rocm_version,
                "python_version": b.python_version,
                "pytorch_version": b.pytorch_version,
                "triton_version": b.triton_version,
                "base_image": b.base_image,
                "base_digest": b.base_digest,
                "pytorch_index_url": b.pytorch_index_url,
                "tag": b.tag,
                "patches": list(b.patches),
                "flash_attention": dict(sorted(b.flash_attention.items())),
            }
            for b in sorted(base_records, key=lambda b: b.id)
        ],
        "sources": [
            {
                "id": s.id,
                "variant": s.variant,
                "repository": s.repository,
                "version": s.version,
                "ref": s.ref,
                "commit": s.configured_commit,
                "compatible_bases": sorted(s.compatible_bases),
            }
            for s in sorted(source_records, key=lambda s: s.id)
        ],
        "images": [
            {"id": i.id, "base": i.base_id, "source": i.source_id, "tag": i.tag}
            for i in sorted(image_records, key=lambda i: i.id)
        ],
    }


def _config_hash(projection: Mapping[str, Any]) -> str:
    payload = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


def resolve_config(summary: ConfigSummary) -> ResolvedBuild:
    """Resolve a validated config into a deterministic ``ResolvedBuild``.

    ``summary`` must be a ``ResolvableSummary`` produced by this module's
    ``load_config`` / ``summary_from_data`` (which run
    ``tools.validate.validate_config`` exactly once). The resolver never
    parses TOML and never re-runs schema validation.

    Raises ``UnapprovedCombination`` before materializing any image whose
    base is outside its source's ``compatible_bases`` — i.e. before any build
    command could ever be emitted. Read-only; offline unless RESOLVE_NETWORK=1.
    """
    if not isinstance(summary, ResolvableSummary):
        raise TypeError(
            "resolve_config requires a ResolvableSummary produced by "
            "tools.resolve.load_config() or tools.resolve.summary_from_data(); "
            f"got {type(summary).__name__}"
        )

    network = os.environ.get(NETWORK_ENV_VAR) == "1"

    architecture_list = tuple(summary.architectures)

    base_records = tuple(
        BaseRecord(
            id=base_id,
            rocm_version=summary.bases[base_id]["rocm_version"],
            python_version=summary.bases[base_id]["python_version"],
            pytorch_version=summary.bases[base_id]["pytorch_version"],
            triton_version=summary.bases[base_id]["triton_version"],
            base_image=summary.bases[base_id]["base_image"],
            base_digest=summary.bases[base_id]["base_digest"],
            pytorch_index_url=summary.bases[base_id]["pytorch_index_url"],
            tag=summary.bases[base_id]["tag"],
            patches=tuple(summary.bases[base_id].get("patches", ())),
            flash_attention=dict(
                summary.bases[base_id].get("flash_attention", {})
            ),
        )
        for base_id in summary.base_ids
    )
    base_by_id = {b.id: b for b in base_records}

    source_records = tuple(
        SourceRecord(
            id=source_id,
            variant=summary.sources[source_id]["variant"],
            repository=summary.sources[source_id]["repository"],
            version=summary.sources[source_id]["version"],
            ref=summary.sources[source_id]["ref"],
            resolved_commit=resolved,
            configured_commit=summary.sources[source_id]["commit"],
            commit_matches=matches,
            compatible_bases=tuple(
                summary.sources[source_id].get("compatible_bases", ())
            ),
        )
        for source_id in summary.source_ids
        for resolved, matches in [
            _resolve_commit(
                summary.sources[source_id]["repository"],
                summary.sources[source_id]["ref"],
                summary.sources[source_id]["commit"],
                network=network,
            )
        ]
    )
    source_by_id = {s.id: s for s in source_records}

    image_records: list[ImageRecord] = []
    for entry in summary.images:
        image_id = entry["id"]
        base_id = entry["base"]
        source_id = entry["source"]
        if base_id not in base_by_id:
            raise UnknownBase(base_id)
        if source_id not in source_by_id:
            raise UnknownSource(source_id)
        source_record = source_by_id[source_id]
        # Guard rail, independent of the validator: refuse to materialize an
        # image for an unapproved base x source pair, before any build
        # command could ever be emitted.
        if base_id not in source_record.compatible_bases:
            raise UnapprovedCombination(base=base_id, source=source_id)
        tag = entry["tag"]
        image_records.append(
            ImageRecord(
                id=image_id,
                base_id=base_id,
                source_id=source_id,
                tag=tag,
                qualified_tag=tag,
                base_record=base_by_id[base_id],
                source_record=source_record,
            )
        )

    images = tuple(image_records)
    projection = _canonical_projection(
        architecture_list, base_records, source_records, images
    )
    return ResolvedBuild(
        architecture_list=architecture_list,
        base_records=base_records,
        source_records=source_records,
        image_records=images,
        config_hash=_config_hash(projection),
    )


# ---------------------------------------------------------------------------
# CLI rendering
# ---------------------------------------------------------------------------


def _print_human(resolved: ResolvedBuild) -> None:
    print(f"ResolvedBuild(config_hash={resolved.config_hash})")
    print(f"  architectures ({len(resolved.architecture_list)}): "
          + ", ".join(resolved.architecture_list))
    print(f"Bases ({len(resolved.base_records)}):")
    for b in resolved.base_records:
        print(f"  {b.id}: rocm={b.rocm_version} python={b.python_version} "
              f"pytorch={b.pytorch_version} triton={b.triton_version}")
        print(f"    image  = {b.base_image}")
        print(f"    digest = {b.base_digest}")
        patches = ", ".join(b.patches) if b.patches else "(none)"
        print(f"    tag    = {b.tag} patches = {patches}")
    print(f"Sources ({len(resolved.source_records)}):")
    for s in resolved.source_records:
        verdict = "yes" if s.commit_matches else "NO"
        print(f"  {s.id} ({s.variant}) version={s.version}")
        print(f"    repo   = {s.repository}")
        print(f"    ref    = {s.ref}")
        print(f"    commit = {s.configured_commit} "
              f"resolved = {s.resolved_commit} matches = {verdict}")
        print(f"    compatible_bases = [{', '.join(s.compatible_bases)}]")
    print(f"Images ({len(resolved.image_records)}):")
    for i in resolved.image_records:
        print(f"  {i.id}: base={i.base_id} source={i.source_id} "
              f"tag={i.qualified_tag}")


def _build_to_json(resolved: ResolvedBuild) -> dict[str, Any]:
    return {
        "architecture_list": list(resolved.architecture_list),
        "base_records": [asdict(b) for b in resolved.base_records],
        "source_records": [asdict(s) for s in resolved.source_records],
        "image_records": [asdict(i) for i in resolved.image_records],
        "config_hash": resolved.config_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="resolve.py",
        description=(
            "Resolve a validated vllm-rdna-docker TOML config into immutable "
            "base/source/image records. Read-only; offline by default (set "
            "RESOLVE_NETWORK=1 to enable the optional git ls-remote commit "
            "check)."
        ),
    )
    parser.add_argument("--config", required=True, help="path to the TOML config")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the full ResolvedBuild as JSON (ignored with --image, "
        "which always prints one ImageRecord as JSON)",
    )
    parser.add_argument(
        "--image",
        metavar="IMAGE_ID",
        help="print a single ImageRecord as JSON; exit 2 with "
        "UnknownImage(image=<id>) if the id is not declared",
    )
    args = parser.parse_args(argv)

    network = os.environ.get(NETWORK_ENV_VAR) == "1"
    mode = "network (git ls-remote enabled)" if network else "offline (config-only)"
    print(f"resolve.py: {mode} commit verification; set {NETWORK_ENV_VAR}=1 to "
          f"{'disable' if network else 'enable'} git ls-remote checks.",
          file=sys.stderr)

    try:
        summary = load_config(args.config)
        resolved = resolve_config(summary)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot read config: {exc}", file=sys.stderr)
        return 2

    if args.image is not None:
        record = next(
            (r for r in resolved.image_records if r.id == args.image), None
        )
        if record is None:
            print(f"error: {UnknownImage(args.image)}", file=sys.stderr)
            return 2
        print(json.dumps(asdict(record), indent=2, sort_keys=True))
        return 0

    if args.json:
        print(json.dumps(_build_to_json(resolved), indent=2, sort_keys=True))
        return 0

    _print_human(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
