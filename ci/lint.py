#!/usr/bin/env python3
"""Static linter for the GitHub Actions workflow adapter.

Enforces the CI contract (see ``ci/README.md``):

  * the workflow invokes the repository-owned CLI — it must contain a
    ``python vllm-rdna-docker/tools/build.py`` invocation (matched against
    the raw file text, so shell indirection cannot hide provider-specific
    build logic);
  * the host-path ban from ``tools/validate.py`` extends to CI — no
    workflow may contain an absolute ``/home/...`` (``/root/``,
    ``/Users/``) path, because the build machine is external and
    Podman-based;
  * the workflow references the required registry secret names
    ``REGISTRY_USER`` and ``REGISTRY_TOKEN`` (values are never printed —
    only the names are checked);
  * the file parses as a valid YAML mapping with a ``jobs`` section.

Every violation is reported as a named error (``HostPathInWorkflow``,
``MissingSecretReference``, ``MissingCliInvocation``, ``WorkflowParseError``)
and the process exits non-zero if any violation is found.

Usage:
    python vllm-rdna-docker/ci/lint.py [workflow.yml ...]

With no arguments, lints the committed adapter
(``.github/workflows/build.yml``). Requires PyYAML (see
``vllm-rdna-docker/requirements-ci.txt``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

import yaml

# ---------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------

#: The single entry point the workflow must call (regex on raw file text).
CLI_PATTERN = re.compile(r"python\s+vllm-rdna-docker/tools/build\.py")

#: Absolute host-path prefixes forbidden anywhere in a workflow file.
#: Mirrors tools/validate.py: the build machine is external and Podman-based.
HOST_PATH_PREFIXES: tuple[str, ...] = ("/home/", "/root/", "/Users/")

#: Registry credential secret NAMES the workflow must reference.
REQUIRED_SECRETS: tuple[str, ...] = ("REGISTRY_USER", "REGISTRY_TOKEN")

#: The committed adapter linted by default.
DEFAULT_WORKFLOWS: tuple[Path, ...] = (
    Path(__file__).resolve().parent.parent / ".github" / "workflows" / "build.yml",
)


# ---------------------------------------------------------------------------
# Named errors
# ---------------------------------------------------------------------------


class WorkflowLintError(Exception):
    """Base class for named CI-contract violations."""

    def __init__(self, workflow: str, detail: str) -> None:
        self.workflow = workflow
        self.detail = detail
        super().__init__(f"{type(self).__name__}(workflow={workflow!r}, detail={detail!r})")


class HostPathInWorkflow(WorkflowLintError):
    """A workflow contains an absolute host path (external-runner ban)."""


class MissingSecretReference(WorkflowLintError):
    """A workflow does not reference a required registry secret name."""


class MissingCliInvocation(WorkflowLintError):
    """A workflow does not invoke the repository-owned build CLI."""


class WorkflowParseError(WorkflowLintError):
    """A workflow file is not valid YAML with a ``jobs`` mapping."""


# ---------------------------------------------------------------------------
# Lint checks
# ---------------------------------------------------------------------------


def lint_text(name: str, text: str) -> list[WorkflowLintError]:
    """Run every contract check against one workflow's raw text.

    ``name`` is a display name (usually the file path) embedded in any
    returned violations. All checks run unconditionally so one lint pass
    surfaces every violation at once.
    """
    violations: list[WorkflowLintError] = []

    try:
        document = yaml.safe_load(text)
        if not isinstance(document, dict) or not isinstance(
            document.get("jobs"), dict
        ):
            violations.append(
                WorkflowParseError(name, "workflow is not a YAML mapping with a 'jobs' section")
            )
    except yaml.YAMLError as exc:
        violations.append(WorkflowParseError(name, f"invalid YAML: {exc}"))

    if not CLI_PATTERN.search(text):
        violations.append(
            MissingCliInvocation(
                name, "no 'python vllm-rdna-docker/tools/build.py' invocation found"
            )
        )

    for prefix in HOST_PATH_PREFIXES:
        if prefix in text:
            violations.append(
                HostPathInWorkflow(
                    name,
                    f"absolute host path prefix {prefix!r} found; the build "
                    "machine is external and Podman-based",
                )
            )

    for secret in REQUIRED_SECRETS:
        if secret not in text:
            violations.append(
                MissingSecretReference(name, f"required secret name {secret!r} not referenced")
            )

    return violations


def lint_file(path: Path) -> list[WorkflowLintError]:
    """Lint one workflow file from disk."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [WorkflowParseError(str(path), f"cannot read file: {exc}")]
    return lint_text(str(path), text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    paths = [Path(a) for a in argv] if argv else list(DEFAULT_WORKFLOWS)
    violations: list[WorkflowLintError] = []
    for path in paths:
        violations.extend(lint_file(path))

    for violation in violations:
        print(f"error: {violation}", file=sys.stderr)
    if violations:
        print(
            f"ci lint: {len(violations)} violation(s) across {len(paths)} workflow(s)",
            file=sys.stderr,
        )
        return 1
    print(f"ci lint: OK ({len(paths)} workflow(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))