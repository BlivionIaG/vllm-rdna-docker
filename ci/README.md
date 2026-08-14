# CI Contract — vllm-rdna-docker

Both providers call the SAME Python CLI. Provider-specific YAML contains only
triggers, runner labels, secrets, and artifact plumbing. The build machine is
external and Podman-based. No host path is hardcoded anywhere.

## Adapters

| Provider | File | Runner labels |
|----------|------|---------------|
| GitHub Actions | `vllm-rdna-docker/.github/workflows/build.yml` | `ubuntu-latest` (overridable via the `VLLM_RDNA_RUNNER` repository variable); GPU smoke on `[self-hosted, linux, rdna]` |
| Gitea Actions | `vllm-rdna-docker/.gitea/workflows/build.yml` | `ubuntu-latest`; GPU smoke on `[self-hosted, linux, rdna]` |

Both adapters invoke identical repository commands:

```bash
python vllm-rdna-docker/tools/build.py build-base --config vllm-rdna-docker/config/rocm-7.2.0.toml [--base rocm720] [--dry-run]
python vllm-rdna-docker/tools/build.py build-vllm --config vllm-rdna-docker/config/rocm-7.2.0.toml --image vllm-026-upstream-rocm720 [--dry-run]
python vllm-rdna-docker/tools/build.py verify     --config vllm-rdna-docker/config/rocm-7.2.0.toml --image vllm-026-upstream-rocm720
```

There is no provider-specific build logic: any change to how images are built
lands in `tools/build.py` (or the Dockerfiles), never in workflow YAML.

## Jobs

- **`build`** — runs on every push to `master` / `v*` tags and on manual
  dispatch (Gitea: every push + manual dispatch). Lints the adapters, runs
  the unit suite (`python -m pytest vllm-rdna-docker/tests/ -q`), then
  exercises the CLI contract end-to-end in dry-run mode. Uploads
  `.omo/evidence/` as the `vllm-rdna-evidence` artifact.
- **`gpu-smoke`** — manual trigger only, runs on the external Podman runner
  (`[self-hosted, linux, rdna]`, the `rdna` label is the hint that this must
  be an RDNA-equipped machine). Executes real (non-dry-run) `build-base
  --base rocm720`, `build-vllm`, and `verify`. A preflight step gates on the
  `REGISTRY_USER` / `REGISTRY_TOKEN` secrets and refuses to build or publish
  without them. Secret values are never printed.

## Secrets

| Name | Purpose |
|------|---------|
| `REGISTRY_USER` | Registry login user for push/promote on the external runner |
| `REGISTRY_TOKEN` | Registry login token (never logged) |

## Static lint

`ci/lint.py` enforces this contract mechanically:

```bash
python vllm-rdna-docker/ci/lint.py            # lints both committed adapters
python vllm-rdna-docker/ci/lint.py FILE ...   # lints specific files
```

Checks (each violation is a named error, exit code 1 if any):

- `MissingCliInvocation` — a workflow never calls
  `python vllm-rdna-docker/tools/build.py` (regex over the raw file text);
- `HostPathInWorkflow` — an absolute `/home/`, `/root/`, or `/Users/` path
  appears in a workflow (the host-path ban from `tools/validate.py` extends
  to CI);
- `MissingSecretReference` — `REGISTRY_USER` or `REGISTRY_TOKEN` is not
  referenced;
- `WorkflowParseError` — the file is not valid YAML with a `jobs` mapping.

CI tooling dependencies are pinned in `vllm-rdna-docker/requirements-ci.txt`.
