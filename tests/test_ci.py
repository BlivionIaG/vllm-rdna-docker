"""Tests for the GitHub Actions workflow adapter and the ci/lint.py contract linter.

Covers:
  * the workflow YAML exists and parses as valid YAML with a ``jobs``
    mapping;
  * ``ci/lint.py`` passes on the committed adapter (module and CLI level);
  * ``ci/lint.py`` rejects an inline workflow containing an absolute
    ``/home/foo/bar`` host path with the named ``HostPathInWorkflow`` error;
  * ``ci/lint.py`` rejects an inline workflow omitting ``REGISTRY_USER`` with
    the named ``MissingSecretReference`` error;
  * the YAML contains at least one step calling
    ``python vllm-rdna-docker/tools/build.py`` (raw-text regex and parsed
    step walk).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

SUBPROJECT = Path(__file__).resolve().parent.parent
WORKFLOW = SUBPROJECT / ".github" / "workflows" / "build.yml"

CLI_PATTERN = re.compile(r"python\s+vllm-rdna-docker/tools/build\.py")

# ci/ is not a package; load lint.py by path (mirrors the script-mode import
# pattern used by tools/build.py).
_spec = importlib.util.spec_from_file_location("ci_lint", SUBPROJECT / "ci" / "lint.py")
lint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lint)


def _valid_workflow_text(*, extra_run: str = "", omit_secret: str = "") -> str:
    """Minimal workflow that satisfies every lint check except the tested one."""
    secrets = "\n".join(
        f"          {name}: ${{{{ secrets.{name} }}}}"
        for name in ("REGISTRY_USER", "REGISTRY_TOKEN")
        if name != omit_secret
    )
    return f"""name: inline-test
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Preflight
        env:
{secrets}
        run: python vllm-rdna-docker/tools/build.py validate --config cfg.toml
{extra_run}"""


# ---------------------------------------------------------------------------
# Existence and YAML validity
# ---------------------------------------------------------------------------


def test_workflow_exists_and_parses() -> None:
    assert WORKFLOW.is_file(), f"missing workflow: {WORKFLOW}"
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{WORKFLOW} is not a YAML mapping"
    assert isinstance(document.get("jobs"), dict), f"{WORKFLOW} has no 'jobs' mapping"
    assert "build" in document["jobs"], f"{WORKFLOW} has no 'build' job"


# ---------------------------------------------------------------------------
# lint.py on the committed adapter
# ---------------------------------------------------------------------------


def test_lint_file_passes_on_committed_workflow() -> None:
    assert lint.lint_file(WORKFLOW) == []


def test_lint_cli_passes_on_committed_workflow(capsys: pytest.CaptureFixture[str]) -> None:
    assert lint.main([]) == 0
    assert "ci lint: OK" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# lint.py rejection cases (inline YAML)
# ---------------------------------------------------------------------------


def test_lint_rejects_host_path() -> None:
    text = _valid_workflow_text(
        extra_run="      - name: Bad\n        run: rsync -avz ./ /home/foo/bar/vllm-rdna/\n"
    )
    violations = lint.lint_text("inline-host-path", text)
    assert any(isinstance(v, lint.HostPathInWorkflow) for v in violations)
    assert "/home/foo/bar" in text  # sanity: the fixture really is hostile


def test_lint_rejects_missing_registry_user() -> None:
    text = _valid_workflow_text(omit_secret="REGISTRY_USER")
    violations = lint.lint_text("inline-missing-secret", text)
    assert any(isinstance(v, lint.MissingSecretReference) for v in violations)
    assert not any(isinstance(v, lint.HostPathInWorkflow) for v in violations)
    assert not any(isinstance(v, lint.MissingCliInvocation) for v in violations)


def test_lint_rejects_missing_cli_invocation() -> None:
    text = _valid_workflow_text().replace(
        "python vllm-rdna-docker/tools/build.py validate", "python tools/build.py validate"
    )
    violations = lint.lint_text("inline-missing-cli", text)
    assert any(isinstance(v, lint.MissingCliInvocation) for v in violations)


def test_lint_rejects_invalid_yaml() -> None:
    violations = lint.lint_text("inline-broken", "jobs: [unclosed\n")
    assert any(isinstance(v, lint.WorkflowParseError) for v in violations)


def test_named_error_rendering() -> None:
    err = lint.HostPathInWorkflow("wf.yml", "found /home/")
    rendered = str(err)
    assert "HostPathInWorkflow" in rendered
    assert "wf.yml" in rendered


# ---------------------------------------------------------------------------
# The adapter calls the repository CLI
# ---------------------------------------------------------------------------


def test_workflow_raw_text_calls_cli() -> None:
    assert CLI_PATTERN.search(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_has_step_calling_cli() -> None:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    run_steps = [
        step["run"]
        for job in document["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step, dict) and "run" in step
    ]
    assert any(CLI_PATTERN.search(command) for command in run_steps), (
        f"{WORKFLOW}: no step invokes {CLI_PATTERN.pattern!r}"
    )