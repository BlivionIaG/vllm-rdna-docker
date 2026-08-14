"""Tests for the vllm-rdna-docker Dockerfile set and static linter.

Covers:
  * the committed Dockerfiles — both pass ``docker/lint.py`` with exit 0;
  * Buildx-only constructs — an inline Dockerfile using ``RUN --mount=``
    (and siblings) is rejected with a named ``BuildxOnlyConstruct`` error;
  * header/arg consistency — an inline Dockerfile whose header comment omits
    a declared ``ARG`` is rejected with a named ``UndocumentedArg`` error;
  * presence rules — a missing file is reported as ``MissingDockerfile``;
  * ``.dockerignore`` — exists and lists the required exclusions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SUBPROJECT = Path(__file__).resolve().parent.parent
LINT_PY = SUBPROJECT / "docker" / "lint.py"
DOCKERFILE_BASE = SUBPROJECT / "Dockerfile.base"
DOCKERFILE_VLLM = SUBPROJECT / "Dockerfile.vllm"
DOCKERIGNORE = SUBPROJECT / ".dockerignore"


def run_lint(*paths: Path) -> subprocess.CompletedProcess[str]:
    """Invoke lint.py as a subprocess; returns the completed process."""
    return subprocess.run(
        [sys.executable, str(LINT_PY), *[str(p) for p in paths]],
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Happy path: the two committed Dockerfiles pass the default lint.
# ---------------------------------------------------------------------------


def test_committed_dockerfiles_exist():
    assert DOCKERFILE_BASE.is_file(), "Dockerfile.base must exist"
    assert DOCKERFILE_VLLM.is_file(), "Dockerfile.vllm must exist"


def test_lint_passes_on_committed_dockerfiles():
    proc = run_lint()  # default: Dockerfile.base + Dockerfile.vllm
    assert proc.returncode == 0, (
        f"lint.py must pass on the committed Dockerfiles; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "lint: OK" in proc.stdout


def test_lint_passes_with_explicit_paths():
    proc = run_lint(DOCKERFILE_BASE, DOCKERFILE_VLLM)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Buildx-only constructs are rejected with a named error.
# ---------------------------------------------------------------------------

_HEADER = "# inline test dockerfile\n#\n#   ARG_FOO  documented\n"


@pytest.mark.parametrize(
    "construct,line",
    [
        ("mount", "RUN --mount=type=cache,target=/root/.cache pip install x"),
        ("cache-from", "RUN --cache-from=type=local,src=/tmp/c pip install x"),
        ("cache-to", "RUN --cache-to=type=local,dest=/tmp/c pip install x"),
        ("ssh", "RUN --ssh=default git clone git@example.com:r.git"),
        ("secret", "RUN --secret=id=token pip install x"),
    ],
)
def test_lint_rejects_buildx_only_constructs(tmp_path: Path, construct: str, line: str):
    dockerfile = tmp_path / "Dockerfile.inline"
    dockerfile.write_text(
        f"{_HEADER}\nARG ARG_FOO\nFROM scratch\n{line}\n",
        encoding="utf-8",
    )
    proc = run_lint(dockerfile)
    assert proc.returncode != 0, "Buildx-only construct must fail the lint"
    assert "BuildxOnlyConstruct" in proc.stderr
    assert f"construct='{construct}'" in proc.stderr


def test_lint_rejects_buildx_mount_with_extra_whitespace(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile.ws"
    dockerfile.write_text(
        f"{_HEADER}\nARG ARG_FOO\nFROM scratch\n"
        "RUN   --mount=type=cache,target=/c   echo hi\n",
        encoding="utf-8",
    )
    proc = run_lint(dockerfile)
    assert proc.returncode != 0
    assert "BuildxOnlyConstruct" in proc.stderr
    assert "construct='mount'" in proc.stderr


# ---------------------------------------------------------------------------
# Header/arg consistency.
# ---------------------------------------------------------------------------


def test_lint_rejects_arg_missing_from_header(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile.undocumented"
    dockerfile.write_text(
        "# header documents only ARG_FOO\n#\n#   ARG_FOO  documented\n"
        "\nARG ARG_FOO\nARG UNDOCUMENTED_ARG\nFROM scratch\n",
        encoding="utf-8",
    )
    proc = run_lint(dockerfile)
    assert proc.returncode != 0
    assert "UndocumentedArg" in proc.stderr
    assert "arg='UNDOCUMENTED_ARG'" in proc.stderr


def test_lint_rejects_dockerfile_without_header(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile.noheader"
    dockerfile.write_text("ARG FOO\nFROM scratch\n", encoding="utf-8")
    proc = run_lint(dockerfile)
    assert proc.returncode != 0
    assert "UndocumentedArg" in proc.stderr
    assert "arg='FOO'" in proc.stderr


def test_lint_passes_when_all_args_documented(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile.ok"
    dockerfile.write_text(
        "# documented dockerfile\n#\n#   FOO  first\n#   BAR  second\n"
        "\nARG FOO\nARG BAR=baz\nFROM scratch\nRUN echo ok\n",
        encoding="utf-8",
    )
    proc = run_lint(dockerfile)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Presence rule.
# ---------------------------------------------------------------------------


def test_lint_rejects_missing_dockerfile(tmp_path: Path):
    proc = run_lint(tmp_path / "Dockerfile.absent")
    assert proc.returncode != 0
    assert "MissingDockerfile" in proc.stderr


# ---------------------------------------------------------------------------
# .dockerignore contract.
# ---------------------------------------------------------------------------


def test_dockerignore_exists_and_lists_required_exclusions():
    assert DOCKERIGNORE.is_file(), ".dockerignore must exist"
    content = DOCKERIGNORE.read_text(encoding="utf-8")
    entries = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    assert ".git" in entries
    assert "tests" in entries
