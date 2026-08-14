#!/usr/bin/env python3
"""Engine-neutral build CLI for vllm-rdna-docker. Podman is canonical;
Docker is supported as an adapter (never a second implementation).

Orchestrates the release pipeline:

    validate -> resolve -> build-base / build-vllm -> verify -> publish -> promote

Properties:
  * validate+resolve gate everything — every subcommand runs
    ``tools.validate`` (via ``tools.resolve.load_config``) and
    ``tools.resolve.resolve_config`` first and refuses to proceed if either
    fails. No schema or resolution logic is re-implemented here;
  * deterministic command rendering — the same resolved record always
    produces the same argv. Podman and Docker renderings differ ONLY in the
    executable name and Docker's ``--pull`` flag (Podman does not accept it
    in this project's contract); the ``--file`` / ``--tag`` / ``--build-arg``
    lists are byte-identical;
  * ``--dry-run`` short-circuits before any side effect — no engine is
    invoked. Rendered commands are always printed to stdout (one per line,
    shell-quoted) so CI can capture them; status and errors go to stderr;
  * engine selection — ``--engine=auto`` (default, or ``CONTAINER_ENGINE``
    env var) picks Podman when on PATH, else Docker, else raises
    ``EngineNotFound``. ``--engine=podman`` / ``--engine=docker`` force one
    engine and raise ``EngineNotFound`` if it is missing. Engine presence is
    verified even in dry-run mode so CI command capture fails fast on a
    misconfigured runner;
  * stdlib only (argparse, subprocess, shutil, tomllib, shlex, dataclasses).

Usage:
    python vllm-rdna-docker/tools/build.py <subcommand> --config <path.toml> \
        [--engine auto|podman|docker] [--dry-run] [--image <id>] [--base <id>]

Subcommands:
    validate    run the schema validator and print the ConfigSummary
    resolve     run the resolver and print the ResolvedBuild (human or --json)
    build-base  render (+ execute) the base-image build for each/some bases
    build-vllm  render (+ execute) the application-image build for each/some images
    verify      verify images: OCI labels, architectures, digest, SBOM evidence
    publish     push immutable image tags (dry-run renders; live requires
                registry credentials from the environment)
    promote     promote a config alias via pull -> tag -> push (never rebuilds)
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Callable, Sequence

try:  # pytest / package mode (vllm-rdna-docker/ on sys.path)
    from tools.cli_errors import CommandFailed, EngineNotFound
    from tools.publish import (
        promote_alias,
        publish_image,
        require_credentials,
        UnknownAlias,
    )
    from tools.resolve import (
        BaseRecord,
        ImageRecord,
        ResolvedBuild,
        load_config,
        resolve_config,
    )
    from tools.validate import ConfigError, ReservedTag, UnknownBase, UnknownImage
    from tools.verify import verify_image
except ModuleNotFoundError:  # script mode: python tools/build.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.cli_errors import CommandFailed, EngineNotFound
    from tools.publish import (
        promote_alias,
        publish_image,
        require_credentials,
        UnknownAlias,
    )
    from tools.resolve import (
        BaseRecord,
        ImageRecord,
        ResolvedBuild,
        load_config,
        resolve_config,
    )
    from tools.validate import ConfigError, ReservedTag, UnknownBase, UnknownImage
    from tools.verify import verify_image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Environment override for the default engine. Podman remains canonical:
#: "auto" prefers podman; this var only changes what "the default" is.
ENGINE_ENV_VAR = "CONTAINER_ENGINE"

BASE_DOCKERFILE = "Dockerfile.base"
VLLM_DOCKERFILE = "Dockerfile.vllm"

#: Fixed ARG order for deterministic rendering (never dict iteration).
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
# Command rendering (deterministic)
# ---------------------------------------------------------------------------


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


def _display_path(path: Path) -> str:
    """Render a path relative to the cwd when possible (no host absolutes)."""
    try:
        return os.path.relpath(path, Path.cwd())
    except ValueError:  # different drive (Windows) — fall back to as-is
        return str(path)


# ---------------------------------------------------------------------------
# Shared preparation: validate -> resolve -> registry + project paths
# ---------------------------------------------------------------------------


def _prepare(config_path: str) -> tuple[ResolvedBuild, dict[str, str], Path]:
    """Validate and resolve the config; return (resolved, registry, project_dir).

    The registry section is read from the already-parsed TOML (the resolver's
    records intentionally do not carry it); validation has already guaranteed
    the three fields exist, so this is a plain dict read, never a re-check.
    Raises ConfigError subclasses / OSError; the caller maps them to exits.
    """
    resolved = resolve_config(load_config(config_path))
    with Path(config_path).open("rb") as handle:
        registry = dict(tomllib.load(handle)["registry"])
    project_dir = Path(config_path).resolve().parent.parent
    return resolved, registry, project_dir


def _select_bases(resolved: ResolvedBuild, base_id: str | None) -> list[BaseRecord]:
    if base_id is None:
        return list(resolved.base_records)
    record = next((b for b in resolved.base_records if b.id == base_id), None)
    if record is None:
        raise UnknownBase(base_id)
    return [record]


def _select_images(resolved: ResolvedBuild, image_id: str | None) -> list[ImageRecord]:
    if image_id is None:
        return list(resolved.image_records)
    record = next((i for i in resolved.image_records if i.id == image_id), None)
    if record is None:
        raise UnknownImage(image_id)
    return [record]


def _base_image_ref(registry: dict[str, str], record: BaseRecord) -> str:
    return f"{registry['host']}/{registry['base_repository']}:{record.tag}"


def _vllm_image_ref(registry: dict[str, str], record: ImageRecord) -> str:
    return f"{registry['host']}/{registry['vllm_repository']}:{record.tag}"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _emit_and_run(
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


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    # Delegate to the validator's own CLI so output mirrors validate.py.
    from tools.validate import main as validate_main

    return validate_main(["--config", args.config])


def _cmd_resolve(args: argparse.Namespace) -> int:
    # Delegate to the resolver's own CLI so output mirrors resolve.py.
    from tools.resolve import main as resolve_main

    argv = ["--config", args.config]
    if args.json:
        argv.append("--json")
    return resolve_main(argv)


def _cmd_build_base(args: argparse.Namespace) -> int:
    resolved, registry, project_dir = _prepare(args.config)
    targets = _select_bases(resolved, args.base)
    engine = select_engine(args.engine)
    print(
        f"build-base: engine={engine} (--engine={args.engine}); "
        f"config_hash={resolved.config_hash}",
        file=sys.stderr,
    )
    print(
        f"build-base: {len(targets)} base(s) selected: "
        + ", ".join(b.id for b in targets),
        file=sys.stderr,
    )
    commands = [
        (
            b.id,
            render_build_argv(
                engine,
                _display_path(project_dir / BASE_DOCKERFILE),
                _base_image_ref(registry, b),
                base_build_args(b, resolved.architecture_list),
                _display_path(project_dir),
            ),
        )
        for b in targets
    ]
    return _emit_and_run(commands, dry_run=args.dry_run)


def _cmd_build_vllm(args: argparse.Namespace) -> int:
    resolved, registry, project_dir = _prepare(args.config)
    targets = _select_images(resolved, args.image)
    engine = select_engine(args.engine)
    print(
        f"build-vllm: engine={engine} (--engine={args.engine}); "
        f"config_hash={resolved.config_hash}",
        file=sys.stderr,
    )
    print(
        f"build-vllm: {len(targets)} image(s) selected: "
        + ", ".join(i.id for i in targets),
        file=sys.stderr,
    )
    commands = [
        (
            i.id,
            render_build_argv(
                engine,
                _display_path(project_dir / VLLM_DOCKERFILE),
                _vllm_image_ref(registry, i),
                vllm_build_args(
                    i,
                    _base_image_ref(registry, i.base_record),
                    resolved.config_hash,
                    resolved.architecture_list,
                ),
                _display_path(project_dir),
            ),
        )
        for i in targets
    ]
    return _emit_and_run(commands, dry_run=args.dry_run)


#: Ordered (ARG name, OCI label key) pairs checked by the verify subcommand.
#: The label keys mirror Dockerfile.vllm (vllm.* / config-hash) plus the base
#: labels inherited from Dockerfile.base (base.tag / base.digest). ARCH_LIST
#: is deliberately absent: verify_image checks the architectures label via
#: its dedicated ``architectures`` parameter.
VERIFY_ARG_LABELS: tuple[tuple[str, str], ...] = (
    ("BASE_TAG", "org.opencontainers.image.base.tag"),
    ("BASE_DIGEST", "org.opencontainers.image.base.digest"),
    ("CONFIG_HASH", "org.opencontainers.image.config-hash"),
    ("VLLM_VARIANT", "vllm.variant"),
    ("VLLM_REPOSITORY", "vllm.repository"),
    ("VLLM_COMMIT", "vllm.commit"),
    ("VLLM_REF", "vllm.ref"),
)


def verify_expected_labels(
    record: ImageRecord, resolved: ResolvedBuild
) -> dict[str, str]:
    """Expected OCI labels for one resolved image, keyed by label name.

    Values come from the same resolved records that feed the build ARGs, so
    verify compares the image against exactly what the build declared.
    """
    source = record.source_record
    base = record.base_record
    arg_values = {
        "BASE_TAG": base.tag,
        "BASE_DIGEST": base.base_digest,
        "CONFIG_HASH": resolved.config_hash,
        "VLLM_VARIANT": source.variant,
        "VLLM_REPOSITORY": source.repository,
        "VLLM_COMMIT": source.resolved_commit,
        "VLLM_REF": source.ref,
    }
    return {label: arg_values[arg] for arg, label in VERIFY_ARG_LABELS}


def _cmd_verify(args: argparse.Namespace) -> int:
    resolved, registry, _ = _prepare(args.config)
    targets = _select_images(resolved, args.image)
    engine = select_engine(args.engine)
    print(
        f"verify: engine={engine} (--engine={args.engine}); "
        f"config_hash={resolved.config_hash}",
        file=sys.stderr,
    )
    all_ok = True
    for image in targets:
        ref = _vllm_image_ref(registry, image)
        expected = verify_expected_labels(image, resolved)
        if args.dry_run:
            print(
                f"verify: dry-run: would run '{engine} inspect {ref}' and "
                f"'syft {engine}://{ref} -o spdx-json --quiet' "
                f"({len(expected)} labels + architectures checked)",
                file=sys.stderr,
            )
            continue
        result = verify_image(
            ref,
            expected=expected,
            engine=engine,
            architectures=resolved.architecture_list,
        )
        ok = result.arch_ok and result.label_ok and result.sbom_ok
        print(
            f"verify: {image.id} ({ref}): {'OK' if ok else 'FAIL'} "
            f"arch_ok={result.arch_ok} label_ok={result.label_ok} "
            f"sbom_ok={result.sbom_ok} "
            f"digest={result.digest or '(none)'} "
            f"sbom={result.sbom_path or '(none)'}",
            file=sys.stderr,
        )
        for error in result.errors:
            print(f"  error: {error}", file=sys.stderr)
        if not ok:
            all_ok = False
    return 0 if all_ok else 1


def _check_not_reserved(tag: str, reserved: frozenset[str]) -> None:
    """Refuse to publish/promote a legacy reserved tag (read-only forever)."""
    if tag in reserved:
        raise ReservedTag(tag)


def _cmd_publish(args: argparse.Namespace) -> int:
    resolved, registry, _ = _prepare(args.config)
    summary = load_config(args.config)
    reserved = frozenset(summary.reserved_tags)
    engine = select_engine(args.engine)
    # Credential preflight: a live run fails before the FIRST push, never
    # mid-batch. Values are never read here — only presence is checked.
    require_credentials(dry_run=args.dry_run)

    # Targets: explicit --image/--base selection, else every declared ref.
    targets: list[tuple[str, str]] = []
    if args.image is None and args.base is None:
        targets = [
            *[(b.id, _base_image_ref(registry, b)) for b in resolved.base_records],
            *[(i.id, _vllm_image_ref(registry, i)) for i in resolved.image_records],
        ]
    else:
        if args.base is not None:
            _check_not_reserved(args.base, reserved)
            targets.extend(
                (b.id, _base_image_ref(registry, b))
                for b in _select_bases(resolved, args.base)
            )
        if args.image is not None:
            _check_not_reserved(args.image, reserved)
            targets.extend(
                (i.id, _vllm_image_ref(registry, i))
                for i in _select_images(resolved, args.image)
            )

    print(
        f"publish: engine={engine} (--engine={args.engine}); "
        f"{len(targets)} ref(s) selected"
        + (" [dry-run]" if args.dry_run else ""),
        file=sys.stderr,
    )
    for label, ref in targets:
        # Defense in depth: the computed tag itself must not be reserved,
        # even if the selector id slipped past the check above.
        _check_not_reserved(ref.rsplit(":", 1)[-1], reserved)
        result = publish_image(ref, engine=engine, dry_run=args.dry_run)
        print(result.command)  # stdout: capturable by CI, always emitted
        print(f"[{label}] {result.command}", file=sys.stderr)
        if args.dry_run:
            print(f"[{label}] dry-run: not executing", file=sys.stderr)
        else:
            print(f"[{label}] pushed digest={result.pushed_digest}", file=sys.stderr)
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    resolved, registry, _ = _prepare(args.config)
    summary = load_config(args.config)
    reserved = frozenset(summary.reserved_tags)
    engine = select_engine(args.engine)
    require_credentials(dry_run=args.dry_run)

    if args.alias is not None:
        _check_not_reserved(args.alias, reserved)
        if args.alias not in summary.aliases:
            raise UnknownAlias(args.alias)
        items = {args.alias: summary.aliases[args.alias]}
    else:
        items = dict(summary.aliases)

    print(
        f"promote: engine={engine} (--engine={args.engine}); "
        f"{len(items)} alias(es) selected (pull -> tag -> push; never rebuilds)"
        + (" [dry-run]" if args.dry_run else ""),
        file=sys.stderr,
    )
    for alias, image_id in items.items():
        record = next(i for i in resolved.image_records if i.id == image_id)
        source_ref = _vllm_image_ref(registry, record)
        target_ref = f"{registry['host']}/{registry['vllm_repository']}:{alias}"
        result = promote_alias(
            source_ref, target_ref, engine=engine, dry_run=args.dry_run
        )
        print(result.command)  # stdout: capturable by CI, always emitted
        print(f"[{alias}] {result.command}", file=sys.stderr)
        if args.dry_run:
            print(f"[{alias}] dry-run: not executing", file=sys.stderr)
        else:
            print(f"[{alias}] promoted digest={result.pushed_digest}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description=(
            "Engine-neutral build CLI for vllm-rdna-docker (Podman canonical, "
            "Docker adapter). Every subcommand validates and resolves the "
            "config before doing anything else."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="path to the TOML config")

    def add_engine(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--engine",
            default=os.environ.get(ENGINE_ENV_VAR, "auto"),
            help=(
                "container engine: 'auto' (default; Podman if on PATH, else "
                "Docker), 'podman', or 'docker'. Default may be overridden "
                f"with the {ENGINE_ENV_VAR} environment variable."
            ),
        )

    def add_dry_run(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="print the rendered commands without executing them "
            "(no build/push side effects)",
        )

    p_validate = sub.add_parser("validate", help="validate the config and print the summary")
    add_config(p_validate)
    p_validate.set_defaults(handler=_cmd_validate)

    p_resolve = sub.add_parser("resolve", help="resolve the config and print the build records")
    add_config(p_resolve)
    p_resolve.add_argument("--json", action="store_true", help="print the ResolvedBuild as JSON")
    p_resolve.set_defaults(handler=_cmd_resolve)

    p_build_base = sub.add_parser("build-base", help="build base image(s) (Dockerfile.base)")
    add_config(p_build_base)
    add_engine(p_build_base)
    add_dry_run(p_build_base)
    p_build_base.add_argument("--base", metavar="BASE_ID", help="build a single base (default: all)")
    p_build_base.set_defaults(handler=_cmd_build_base)

    p_build_vllm = sub.add_parser("build-vllm", help="build application image(s) (Dockerfile.vllm)")
    add_config(p_build_vllm)
    add_engine(p_build_vllm)
    add_dry_run(p_build_vllm)
    p_build_vllm.add_argument("--image", metavar="IMAGE_ID", help="build a single image (default: all)")
    p_build_vllm.set_defaults(handler=_cmd_build_vllm)

    p_verify = sub.add_parser("verify", help="verify images (OCI labels, architectures, digest, SBOM)")
    add_config(p_verify)
    add_engine(p_verify)
    add_dry_run(p_verify)
    p_verify.add_argument("--image", metavar="IMAGE_ID", help="verify a single image (default: all)")
    p_verify.set_defaults(handler=_cmd_verify)

    p_publish = sub.add_parser("publish", help="push immutable image/base tags (never reserved tags)")
    add_config(p_publish)
    add_engine(p_publish)
    add_dry_run(p_publish)
    p_publish.add_argument("--base", metavar="BASE_ID", help="publish a single base")
    p_publish.add_argument("--image", metavar="IMAGE_ID", help="publish a single image")
    p_publish.set_defaults(handler=_cmd_publish)

    p_promote = sub.add_parser("promote", help="promote alias(es) via pull -> tag -> push (never rebuilds)")
    add_config(p_promote)
    add_engine(p_promote)
    add_dry_run(p_promote)
    p_promote.add_argument("--alias", metavar="ALIAS", help="promote a single alias (default: all)")
    p_promote.set_defaults(handler=_cmd_promote)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (UnknownBase, UnknownImage) as exc:
        # Selection errors mirror resolve.py's CLI: exit 2, named error.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except EngineNotFound as exc:
        print(
            f"error: {exc} — install Podman (canonical) or Docker, or fix "
            f"--engine/{ENGINE_ENV_VAR}",
            file=sys.stderr,
        )
        return 1
    except CommandFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: cannot read config: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
