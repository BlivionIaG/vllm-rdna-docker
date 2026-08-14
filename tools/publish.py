#!/usr/bin/env python3
"""Immutable publication and alias promotion for vllm-rdna-docker.

Pushes immutable build tags and promotes aliases to already-published images.
Promotion NEVER rebuilds: it is strictly ``pull -> tag -> push`` of an image
that the pipeline has already built and verified.

Properties:
  * credential-safe — registry credentials are sourced ONLY from the
    environment (``DOCKERHUB_USERNAME``/``DOCKERHUB_TOKEN`` or
    ``REGISTRY_USER``/``REGISTRY_TOKEN``). Their values are never printed,
    never written to logs, and never embedded in rendered commands. A live
    (non-dry-run) push without credentials raises ``MissingCredentials``
    BEFORE any engine invocation;
  * dry-run first — with ``dry_run=True`` no engine is invoked and no
    credentials are required; the rendered command(s) are returned so the
    CLI can print them;
  * immutable tags — legacy tags (``ConfigSummary.reserved_tags``, e.g.
    ``v0.22.1`` / ``v0.22.1_base``) are refused by the CLI layer in
    ``tools/build.py`` via the validator's ``ReservedTag`` error;
  * stdlib only (os, re, shlex, subprocess, dataclasses).

Usage (library):
    from tools.publish import publish_image, promote_alias
    result = publish_image("docker.io/blivioniag/vllm-rdna:v0.26.0",
                           engine="podman", dry_run=True)
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:  # pytest / package mode (vllm-rdna-docker/ on sys.path)
    from tools.validate import ConfigError
except ModuleNotFoundError:  # script mode: python tools/publish.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.validate import ConfigError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: A sha256 digest token as printed by ``docker push``
#: (``digest: sha256:...``) and ``podman push`` (``Digest: sha256:...``).
#: Matching the bare token accepts both formats.
DIGEST_TOKEN_RE = re.compile(r"sha256:[0-9a-f]{64}")

#: Accepted credential environment variable pairs, in priority order. The
#: FIRST pair whose variables are both set wins. Only the variable NAMES are
#: ever referenced in errors/logs — never the values.
CREDENTIAL_ENV_PAIRS: tuple[tuple[str, str], ...] = (
    ("DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"),
    ("REGISTRY_USER", "REGISTRY_TOKEN"),
)


# ---------------------------------------------------------------------------
# Named errors (ConfigError subclasses: str(err) renders Name(field=value))
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublishResult:
    """Outcome of one ``publish_image`` call."""

    image: str
    pushed_digest: str | None
    dry_run: bool
    command: str


@dataclass(frozen=True)
class PromoteResult:
    """Outcome of one ``promote_alias`` call.

    ``command`` renders the full pull -> tag -> push sequence joined with
    `` && `` (never executed as a shell string; it is a display artifact —
    the live path invokes each step as a separate argv list).
    """

    source: str
    target: str
    pushed_digest: str | None
    command: str


# ---------------------------------------------------------------------------
# Credential gate (values are never read into logs — only presence is checked)
# ---------------------------------------------------------------------------


def credentials_present(environ: Mapping[str, str] | None = None) -> bool:
    """True iff one accepted credential pair is fully set in the environment."""
    env = os.environ if environ is None else environ
    return any(
        bool(env.get(user_var)) and bool(env.get(token_var))
        for user_var, token_var in CREDENTIAL_ENV_PAIRS
    )


def require_credentials(*, dry_run: bool) -> None:
    """Raise ``MissingCredentials`` for a live run without credentials.

    Dry-run never requires credentials. This is called by ``publish_image``
    and ``promote_alias`` themselves, and by the CLI as an upfront preflight
    so a multi-target publish fails before the first push.
    """
    if dry_run:
        return
    if not credentials_present():
        raise MissingCredentials()


# ---------------------------------------------------------------------------
# Digest extraction
# ---------------------------------------------------------------------------


def extract_digest(stdout: str) -> str | None:
    """Return the first ``sha256:...`` token in engine push output, if any."""
    match = DIGEST_TOKEN_RE.search(stdout)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# Publication / promotion
# ---------------------------------------------------------------------------


def _run_step(argv: Sequence[str], image: str) -> subprocess.CompletedProcess[str]:
    """Run one engine step, mapping a non-zero exit to ``PushFailed``."""
    try:
        return subprocess.run(
            list(argv), check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        raise PushFailed(image=image, stderr=exc.stderr or "") from exc


def publish_image(
    image_ref: str, *, engine: str, dry_run: bool = False
) -> PublishResult:
    """Push one immutable image tag with ``<engine> push <image_ref>``.

    Dry-run renders the command and returns without any side effect or
    credential requirement. Live mode requires credentials (see
    ``require_credentials``), pushes with captured text output, and extracts
    the pushed digest from stdout (both ``digest: sha256:...`` and
    ``Digest: sha256:...`` formats are accepted).
    """
    command = shlex.join([engine, "push", image_ref])
    if dry_run:
        return PublishResult(
            image=image_ref, pushed_digest=None, dry_run=True, command=command
        )
    require_credentials(dry_run=False)
    proc = _run_step([engine, "push", image_ref], image_ref)
    return PublishResult(
        image=image_ref,
        pushed_digest=extract_digest(proc.stdout),
        dry_run=False,
        command=command,
    )


def promote_alias(
    source_image: str,
    target_alias: str,
    *,
    engine: str,
    dry_run: bool = False,
) -> PromoteResult:
    """Promote ``target_alias`` onto the already-published ``source_image``.

    NEVER rebuilds: strictly ``pull source -> tag source target -> push
    target``. Dry-run renders the sequence and returns without side effects.
    Live mode requires credentials and returns the digest reported by the
    final push.
    """
    steps = [
        [engine, "pull", source_image],
        [engine, "tag", source_image, target_alias],
        [engine, "push", target_alias],
    ]
    command = " && ".join(shlex.join(step) for step in steps)
    if dry_run:
        return PromoteResult(
            source=source_image,
            target=target_alias,
            pushed_digest=None,
            command=command,
        )
    require_credentials(dry_run=False)
    digest: str | None = None
    for step in steps:
        proc = _run_step(step, step[-1])
        if step[1] == "push":
            digest = extract_digest(proc.stdout)
    return PromoteResult(
        source=source_image,
        target=target_alias,
        pushed_digest=digest,
        command=command,
    )
