#!/usr/bin/env python3
"""Named errors owned by the vllm-rdna-docker CLI.

A single ``ConfigError`` hierarchy covers every failure that can be raised
by the validator, the resolver, the build/publish CLI, and the verify CLI.
``str(err)`` renders ``Name(field=value, ...)``; ``err.context`` carries
the structured fields for programmatic handling.

Layout in this module (mirrors the order in which the original classes
were defined across validate.py / cli_errors.py / publish.py):

* base class
* TOML / structural parsing
* section / field shape
* architecture
* id uniqueness (bases / sources / images / patches / artifacts)
* id resolution (unknown references)
* variant / install / commit / digest / checksum value validity
* matrix approval
* tag / alias rules
* host-path ban
* CLI / engine (build.py)
* publish (publish.py)
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Base class for all named configuration errors."""

    def __init__(self, **context: Any) -> None:
        super().__init__(context)
        self.context = context

    @property
    def name(self) -> str:
        return type(self).__name__

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.name}({inner})"


# ---------------------------------------------------------------------------
# TOML / structural
# ---------------------------------------------------------------------------


class MalformedTOML(ConfigError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail=detail)


class UnknownSection(ConfigError):
    def __init__(self, section: str) -> None:
        super().__init__(section=section)


class MissingSection(ConfigError):
    def __init__(self, section: str) -> None:
        super().__init__(section=section)


class EmptySection(ConfigError):
    def __init__(self, section: str) -> None:
        super().__init__(section=section)


class UnknownField(ConfigError):
    def __init__(self, section: str, field_: str) -> None:
        super().__init__(section=section, field=field_)


class MissingField(ConfigError):
    def __init__(self, section: str, field_: str) -> None:
        super().__init__(section=section, field=field_)


class InvalidFieldType(ConfigError):
    def __init__(self, section: str, field_: str, expected: str) -> None:
        super().__init__(section=section, field=field_, expected=expected)


class InvalidSchemaVersion(ConfigError):
    def __init__(self, found: Any) -> None:
        super().__init__(found=found)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


class PrimaryArchitectureNotAllowed(ConfigError):
    def __init__(self, path: str = "") -> None:
        super().__init__(path=path)


class MissingArchitecture(ConfigError):
    def __init__(self, arch: str) -> None:
        super().__init__(arch=arch)


class UnexpectedArchitecture(ConfigError):
    def __init__(self, arch: str) -> None:
        super().__init__(arch=arch)


class DuplicateArchitecture(ConfigError):
    def __init__(self, arch: str) -> None:
        super().__init__(arch=arch)


# ---------------------------------------------------------------------------
# Id uniqueness
# ---------------------------------------------------------------------------


class DuplicateBaseId(ConfigError):
    def __init__(self, base: str) -> None:
        super().__init__(base=base)


class MissingDefaultBase(ConfigError):
    def __init__(self) -> None:
        super().__init__()


class MultipleDefaultBases(ConfigError):
    def __init__(self, bases: tuple[str, ...]) -> None:
        super().__init__(bases=list(bases))


class DuplicateSourceId(ConfigError):
    def __init__(self, source: str) -> None:
        super().__init__(source=source)


class DuplicateImageId(ConfigError):
    def __init__(self, image: str) -> None:
        super().__init__(image=image)


class DuplicatePatchId(ConfigError):
    def __init__(self, patch: str) -> None:
        super().__init__(patch=patch)


class DuplicateArtifactId(ConfigError):
    def __init__(self, artifact: str) -> None:
        super().__init__(artifact=artifact)


# ---------------------------------------------------------------------------
# Id resolution
# ---------------------------------------------------------------------------


class UnknownBase(ConfigError):
    def __init__(self, base: str) -> None:
        super().__init__(base=base)


class UnknownSource(ConfigError):
    def __init__(self, source: str) -> None:
        super().__init__(source=source)


class UnknownImage(ConfigError):
    def __init__(self, image: str) -> None:
        super().__init__(image=image)


class UnknownPatch(ConfigError):
    def __init__(self, patch: str) -> None:
        super().__init__(patch=patch)


# ---------------------------------------------------------------------------
# Field value validity
# ---------------------------------------------------------------------------


class InvalidVariant(ConfigError):
    def __init__(self, source: str, variant: str) -> None:
        super().__init__(source=source, variant=variant)


class InvalidFlashAttentionInstall(ConfigError):
    def __init__(self, base: str, install: str) -> None:
        super().__init__(base=base, install=install)


class InvalidCommit(ConfigError):
    def __init__(self, source: str, commit: str) -> None:
        super().__init__(source=source, commit=commit)


class InvalidDigest(ConfigError):
    def __init__(self, base: str, digest: str) -> None:
        super().__init__(base=base, digest=digest)


class InvalidChecksum(ConfigError):
    def __init__(self, section: str, field_: str) -> None:
        super().__init__(section=section, field=field_)


# ---------------------------------------------------------------------------
# Matrix / tag / alias
# ---------------------------------------------------------------------------


class UnapprovedCombination(ConfigError):
    def __init__(self, base: str, source: str) -> None:
        super().__init__(base=base, source=source)


class DuplicateTag(ConfigError):
    def __init__(self, tag: str) -> None:
        super().__init__(tag=tag)


class ReservedTag(ConfigError):
    def __init__(self, tag: str) -> None:
        super().__init__(tag=tag)


class AliasTagMismatch(ConfigError):
    def __init__(self, alias: str, image: str) -> None:
        super().__init__(alias=alias, image=image)


# ---------------------------------------------------------------------------
# Host-path ban
# ---------------------------------------------------------------------------


class HostPathNotAllowed(ConfigError):
    def __init__(self, section: str, field_: str, value: str) -> None:
        super().__init__(section=section, field=field_, value=value)


# ---------------------------------------------------------------------------
# Build CLI / engine
# ---------------------------------------------------------------------------


class EngineNotFound(ConfigError):
    """The requested container engine (podman/docker/auto) is not on PATH.

    ``engine`` is the value the caller asked for: ``"podman"`` or
    ``"docker"`` when forced, or ``"auto"`` when neither candidate was
    found.
    """

    def __init__(self, engine: str) -> None:
        super().__init__(engine=engine)


class CommandFailed(ConfigError):
    """A container engine invocation exited non-zero (non-dry-run only)."""

    def __init__(self, command: str, exit_code: int) -> None:
        super().__init__(command=command, exit_code=exit_code)


# ---------------------------------------------------------------------------
# Publish CLI
# ---------------------------------------------------------------------------

#: Accepted credential environment variable pairs, in priority order. The
#: FIRST pair whose variables are both set wins. Only the variable NAMES are
#: ever referenced in errors/logs — never the values.
CREDENTIAL_ENV_PAIRS: tuple[tuple[str, str], ...] = (
    ("DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"),
    ("REGISTRY_USER", "REGISTRY_TOKEN"),
)


class MissingCredentials(ConfigError):
    """Live push requested but no registry credentials are in the environment.

    Raised BEFORE any engine invocation. Carries only the accepted variable
    NAMES — credential values are never included.
    """

    def __init__(self) -> None:
        super().__init__(
            expected=" or ".join(
                f"{user}/{token}" for user, token in CREDENTIAL_ENV_PAIRS
            )
        )


class PushFailed(ConfigError):
    """A push/pull/tag engine invocation exited non-zero.

    ``stderr`` is the engine's captured stderr. Engines receive credentials
    via their own auth files / stdin, so engine stderr does not contain
    credential values under this project's contract.
    """

    def __init__(self, image: str, stderr: str) -> None:
        super().__init__(image=image, stderr=stderr)


class UnknownAlias(ConfigError):
    """The requested alias is not declared in the config's ``[aliases]``."""

    def __init__(self, alias: str) -> None:
        super().__init__(alias=alias)