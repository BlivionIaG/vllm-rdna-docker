#!/usr/bin/env python3
"""Static Dockerfile linter for vllm-rdna-docker.

Podman-first, Docker-compatible: the committed Dockerfiles must use only
standard Dockerfile syntax. This linter enforces three rules over
``Dockerfile.base`` and ``Dockerfile.vllm`` (or over explicit paths passed
on the command line):

  1. both Dockerfiles are present (``MissingDockerfile``);
  2. no Buildx-only constructs — ``RUN --mount=``, ``RUN --cache-from=``,
     ``RUN --cache-to=``, ``RUN --ssh=``, ``RUN --secret=``
     (``BuildxOnlyConstruct``);
  3. every declared ``ARG`` is documented in the header comment, defined as
     the first contiguous comment block at the top of the file
     (``UndocumentedArg``).

The parser is intentionally tolerant: it splits the file into lines, strips
leading whitespace, and recognises the directives it cares about
(``ARG``, ``RUN``). It is not a full Dockerfile parser and does not need to
be — the rules above are line-local.

Usage:
    python vllm-rdna-docker/docker/lint.py                 # lint the two committed Dockerfiles
    python vllm-rdna-docker/docker/lint.py PATH [PATH...]  # lint explicit files

Exit codes: 0 on pass, 1 on any violation, 2 on unreadable file.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SUBPROJECT = Path(__file__).resolve().parent.parent
DEFAULT_DOCKERFILES = (SUBPROJECT / "Dockerfile.base", SUBPROJECT / "Dockerfile.vllm")

#: Directive recognisers (line-local, tolerant of leading whitespace).
ARG_RE = re.compile(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
RUN_RE = re.compile(r"^RUN\s", re.IGNORECASE)

#: Buildx-only constructs that must never appear in a RUN line. The tuple
#: is (substring-after-normalisation, human name). Whitespace is collapsed
#: before matching so ``RUN  --mount=`` and ``RUN --mount =`` are caught.
BUILDX_ONLY = (
    ("--mount=", "mount"),
    ("--cache-from=", "cache-from"),
    ("--cache-to=", "cache-to"),
    ("--ssh=", "ssh"),
    ("--secret=", "secret"),
)


# ---------------------------------------------------------------------------
# Named errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LintError:
    """Base class for lint violations. ``str(err)`` renders the named error."""

    name: str = "LintError"

    def __str__(self) -> str:  # pragma: no cover - overridden by subclasses
        return self.name


@dataclass(frozen=True)
class MissingDockerfile(LintError):
    path: str = ""
    name: str = "MissingDockerfile"

    def __str__(self) -> str:
        return f"MissingDockerfile(path={self.path!r})"


@dataclass(frozen=True)
class BuildxOnlyConstruct(LintError):
    path: str = ""
    line: int = 0
    construct: str = ""
    name: str = "BuildxOnlyConstruct"

    def __str__(self) -> str:
        return (
            f"BuildxOnlyConstruct(path={self.path!r}, line={self.line}, "
            f"construct={self.construct!r})"
        )


@dataclass(frozen=True)
class UndocumentedArg(LintError):
    path: str = ""
    arg: str = ""
    name: str = "UndocumentedArg"

    def __str__(self) -> str:
        return f"UndocumentedArg(path={self.path!r}, arg={self.arg!r})"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _header_block(lines: Sequence[str]) -> str:
    """Return the first contiguous comment block at the top of the file.

    Leading blank lines are skipped; the block then extends over every
    consecutive ``#`` comment line. The first non-comment, non-blank line
    ends it. Comment text is returned with the leading ``#`` stripped so
    substring checks read naturally.
    """
    out: list[str] = []
    started = False
    for raw in lines:
        stripped = raw.strip()
        if not started:
            if not stripped:
                continue
            if stripped.startswith("#"):
                started = True
                out.append(stripped.lstrip("#").strip())
                continue
            break  # first real directive before any comment: empty header
        if stripped.startswith("#"):
            out.append(stripped.lstrip("#").strip())
            continue
        if not stripped:
            break  # blank line ends the contiguous block
        break
    return "\n".join(out)


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------


def lint_text(text: str, *, path: str = "<inline>") -> list[LintError]:
    """Lint one Dockerfile's contents. Returns a list of violations."""
    errors: list[LintError] = []
    lines = text.splitlines()

    # Rule 2: Buildx-only constructs in RUN lines.
    for lineno, raw in enumerate(lines, start=1):
        collapsed = _collapse_ws(raw.strip())
        if not RUN_RE.match(collapsed):
            continue
        for needle, name in BUILDX_ONLY:
            if needle in collapsed:
                errors.append(
                    BuildxOnlyConstruct(path=path, line=lineno, construct=name)
                )

    # Rule 3: every declared ARG is documented in the header block.
    header = _header_block(lines)
    declared: list[str] = []
    seen: set[str] = set()
    for raw in lines:
        match = ARG_RE.match(raw.strip())
        if match:
            arg = match.group(1)
            if arg not in seen:
                seen.add(arg)
                declared.append(arg)
    for arg in declared:
        if not re.search(rf"\b{re.escape(arg)}\b", header):
            errors.append(UndocumentedArg(path=path, arg=arg))

    return errors


def lint_file(path: Path) -> list[LintError]:
    """Lint one Dockerfile on disk. Missing files yield ``MissingDockerfile``."""
    if not path.is_file():
        return [MissingDockerfile(path=str(path))]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"error: cannot read {path}: {exc}") from exc
    return lint_text(text, path=str(path))


def lint_paths(paths: Sequence[Path]) -> list[LintError]:
    """Lint every path, accumulating violations in file order."""
    errors: list[LintError] = []
    for path in paths:
        errors.extend(lint_file(path))
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lint.py",
        description=(
            "Static Dockerfile linter for vllm-rdna-docker. With no arguments, "
            "lints Dockerfile.base and Dockerfile.vllm; otherwise lints the "
            "explicit paths given."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="DOCKERFILE",
        help="explicit Dockerfile paths (default: Dockerfile.base + Dockerfile.vllm)",
    )
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.paths] if args.paths else list(DEFAULT_DOCKERFILES)
    errors = lint_paths(paths)

    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        print(f"lint: FAIL ({len(errors)} violation(s))", file=sys.stderr)
        return 1

    print(f"lint: OK ({len(paths)} file(s), 0 violations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
