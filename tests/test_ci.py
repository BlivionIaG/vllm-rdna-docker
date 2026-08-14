"""Tests for the CI workflow adapters and the ci/lint.py contract linter.

Covers:
  * both workflow YAMLs (GitHub + Gitea) exist and parse as valid YAML with
    a ``jobs`` mapping;
  * ``ci/lint.py`` passes on the committed adapters (module and CLI level);
  * ``ci/lint.py`` rejects an inline workflow containing an absolute
    ``/home/foo/bar`` host path with the named ``HostPathInWorkflow`` error;
  * ``ci/lint.py`` rejects an inline workflow omitting ``REGISTRY_USER`` with
    the named ``MissingSecretReference`` error;
  * both YAMLs contain at least one step calling
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
GITHUB_WORKFLOW = SUBPROJECT / ".github" / "workflows" / "build.yml"
GITEA_WORKFLOW = SUBPROJECT / ".gitea" / "workflows" / "build.yml"
BOTH_WORKFLOWS = (GITHUB_WORKFLOW, GITEA_WORKFLOW)

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


@pytest.mark.parametrize("workflow", BOTH_WORKFLOWS, ids=lambda p: p.parent.parent.name)
def test_workflow_exists_and_parses(workflow: Path) -> None:
    assert workflow.is_file(), f"missing workflow: {workflow}"
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{workflow} is not a YAML mapping"
    assert isinstance(document.get("jobs"), dict), f"{workflow} has no 'jobs' mapping"
    assert "build" in document["jobs"], f"{workflow} has no 'build' job"


# ---------------------------------------------------------------------------
# lint.py on the committed adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", BOTH_WORKFLOWS, ids=lambda p: p.parent.parent.name)
def test_lint_file_passes_on_committed_workflows(workflow: Path) -> None:
    assert lint.lint_file(workflow) == []


def test_lint_cli_passes_on_committed_workflows(capsys: pytest.CaptureFixture[str]) -> None:
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
# Both adapters call the same repository CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", BOTH_WORKFLOWS, ids=lambda p: p.parent.parent.name)
def test_workflow_raw_text_calls_cli(workflow: Path) -> None:
    assert CLI_PATTERN.search(workflow.read_text(encoding="utf-8"))


@pytest.mark.parametrize("workflow", BOTH_WORKFLOWS, ids=lambda p: p.parent.parent.name)
def test_workflow_has_step_calling_cli(workflow: Path) -> None:
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    run_steps = [
        step["run"]
        for job in document["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step, dict) and "run" in step
    ]
    assert any(CLI_PATTERN.search(command) for command in run_steps), (
        f"{workflow}: no step invokes {CLI_PATTERN.pattern!r}"
    )


def test_both_adapters_invoke_identical_cli_commands() -> None:
    """The provider YAMLs must not diverge in what they ask the CLI to do."""
    per_file = []
    for workflow in BOTH_WORKFLOWS:
        document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        commands = sorted(
            step["run"]
            for job in document["jobs"].values()
            for step in job.get("steps", [])
            if isinstance(step, dict)
            and "run" in step
            and CLI_PATTERN.search(step["run"])
        )
        per_file.append(commands)
    assert per_file[0] == per_file[1], (
        "GitHub and Gitea adapters invoke different CLI commands:\n"
        f"  github: {per_file[0]}\n  gitea:  {per_file[1]}"
    )
