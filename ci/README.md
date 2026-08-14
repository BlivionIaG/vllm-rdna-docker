# CI Contract — vllm-rdna-docker

The workflow calls the repository-owned Python CLI. Provider-specific YAML
contains only triggers, runner labels, secrets, and artifact plumbing.
The build machine is external and Podman-based. No host path is hardcoded
anywhere.

## Adapters

| Provider | File | Runner labels |
|----------|------|---------------|
| GitHub Actions | `vllm-rdna-docker/.github/workflows/build.yml` | `ubuntu-latest` (overridable via the `VLLM_RDNA_RUNNER` repository variable); GPU jobs on `[self-hosted, linux, rdna]` |

The adapter invokes the same repository commands:

```bash
python vllm-rdna-docker/tools/build.py build-base --config vllm-rdna-docker/config/rocm-7.2.0.toml [--base <id>] [--dry-run]
python vllm-rdna-docker/tools/build.py build-vllm  --config vllm-rdna-docker/config/rocm-7.2.0.toml --image <id> [--dry-run]
python vllm-rdna-docker/tools/build.py verify      --config vllm-rdna-docker/config/rocm-7.2.0.toml --image <id>
python vllm-rdna-docker/tools/build.py publish    --config vllm-rdna-docker/config/rocm-7.2.0.toml --image <id> [--base <id>]
```

There is no provider-specific build logic: any change to how images are
built lands in `tools/build.py` (or the Dockerfiles), never in workflow YAML.

## Jobs

- **`build`** — runs on `push` to `v*` tags and on manual dispatch. Lints
  the adapter, runs the unit suite (`python -m pytest
  vllm-rdna-docker/tests/ -q`), then exercises the CLI contract end-to-end
  in dry-run mode. Uploads `.omo/evidence/` as the `vllm-rdna-evidence`
  artifact. Branch pushes (master, etc.) never trigger this job.
- **`build-base-on-demand`** — workflow-dispatch only. The `base_id`
  workflow input is a choice between `rocm720` and `rocm714` (the
  `[bases.*]` keys in `config/rocm-7.2.0.toml`). Builds that single rocm
  base on the external Podman runner (`[self-hosted, linux, rdna]`) and
  publishes it to the registry. Credentials preflighted via
  `REGISTRY_USER` / `REGISTRY_TOKEN`.
- **`release`** — tag-release only (`push` to `refs/tags/v*`). Runs on the
  external Podman runner (`[self-hosted, linux, rdna]`, the `rdna` label
  is the hint that this must be an RDNA-equipped machine). Builds the
  application image (`vllm-026-upstream-rocm720`) referencing the
  already-published base, verifies OCI labels + SBOM, then publishes the
  vllm image. Credentials preflighted via `REGISTRY_USER` /
  `REGISTRY_TOKEN`; secret values are never printed.

Triggers summary:

| Event | `build` | `build-base-on-demand` | `release` |
|-------|---------|------------------------|-----------|
| Push to branch | ❌ | ❌ | ❌ |
| Push tag `vX.Y.Z` | ✅ | ❌ | ✅ |
| Workflow dispatch (no input) | ✅ | ❌ | ❌ |
| Workflow dispatch + `base_id` | ✅ | ✅ | ❌ |

## Secrets

| Name | Purpose |
|------|---------|
| `REGISTRY_USER` | Registry login user for push/promote on the external runner |
| `REGISTRY_TOKEN` | Registry login token (never logged) |

## Static lint

`ci/lint.py` enforces this contract mechanically:

```bash
python vllm-rdna-docker/ci/lint.py            # lints the committed adapter
python vllm-rdna-docker/ci/lint.py FILE ...   # lint specific files
```

Checks (each violation is a named error, exit code 1 if any):

- `MissingCliInvocation` — the workflow never calls
  `python vllm-rdna-docker/tools/build.py` (regex over the raw file text);
- `HostPathInWorkflow` — an absolute `/home/`, `/root/`, or `/Users/` path
  appears in the workflow (the host-path ban from `tools/validate.py`
  extends to CI);
- `MissingSecretReference` — `REGISTRY_USER` or `REGISTRY_TOKEN` is not
  referenced;
- `WorkflowParseError` — the file is not valid YAML with a `jobs` mapping.

CI tooling dependencies are pinned in `vllm-rdna-docker/requirements-ci.txt`.