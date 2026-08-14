# vllm-rdna-docker

Standalone, Podman-first build pipeline that produces portable ROCm/RDNA base
images and vLLM application images.

The pipeline is TOML-driven: a single config file declares the ROCm base
stacks, the vLLM sources (upstream releases and independently versioned
extras-fork releases), and the explicit approved base x source matrix. One
engine-neutral Python CLI (`tools/build.py`) validates, resolves, builds,
verifies, publishes, and promotes images. Podman is the canonical engine;
Docker is supported as an adapter of the same CLI, never as a second
implementation.

Nothing in this repository assumes a fixed host path or a specific runner
hostname. The build machine is **external and Podman-based**; CI runners are
selected by labels, and registry credentials arrive exclusively through
secrets.

## Layout

| Path | Purpose |
|------|---------|
| `config/rocm-7.2.0.toml` | Release configuration (bases, sources, images, aliases) |
| `config/schema.md` | Config contract enforced by `tools/validate.py` |
| `tools/validate.py` | Read-only schema validator (named errors) |
| `tools/resolve.py` | Ref/commit resolution and approved-matrix expansion |
| `tools/build.py` | Engine-neutral CLI: validate, resolve, build-base, build-vllm, verify, publish, promote |
| `tools/verify.py` | OCI label / architecture / digest checks + SPDX SBOM evidence |
| `tools/publish.py` | Immutable push and alias promotion (credentials from environment only) |
| `Dockerfile.base` | Portable ROCm base image (standard Dockerfile syntax) |
| `Dockerfile.vllm` | vLLM application image, parameterized by source record |
| `ci/lint.py` | Static lint for the CI adapters (host-path ban, required secrets) |
| `.github/workflows/build.yml` | GitHub Actions adapter (thin) |
| `.gitea/workflows/build.yml` | Gitea Actions adapter (thin) |
| `tests/` | Unit suite: `python -m pytest vllm-rdna-docker/tests/ -q` |

## Configuration

The config schema is documented in [`config/schema.md`](config/schema.md) and
enforced mechanically by `tools/validate.py`. The schema is strict: unknown
sections and unknown fields are rejected with named errors.

Key invariants:

- **Fixed architecture set** — exactly `gfx1030`, `gfx1100`, `gfx1101`,
  `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201`. One portable image; there is
  **no primary architecture** (a `primary` key at any nesting level is a
  validation error).
- **Explicit matrix** — no automatic Cartesian product. Adding a base or a
  source changes nothing until an `[[images]]` entry approves the
  combination, and the base must also appear in the source's
  `compatible_bases`.
- **Immutability** — sources pin a full 40-hex commit; bases pin their own
  `base_image` plus a full `sha256:` digest; artifacts pin a `sha256`
  checksum.
- **No host state** — absolute host paths are rejected everywhere, including
  in CI workflow files.

## Image repositories and tags

| Repository | Content |
|------------|---------|
| `docker.io/blivioniag/rocm-rdna` | ROCm/RDNA base images, one per `[bases.*]` entry |
| `docker.io/blivioniag/vllm-rdna` | vLLM application images, one per `[[images]]` entry |

Current tags (from `config/rocm-7.2.0.toml`):

| Tag | Base | Source |
|-----|------|--------|
| `v0.26.0` | rocm720 | upstream `v0.26.0` |
| `v0.26.0-extras` | rocm720 | extras-fork `v0.26.0-extras` |
| `v0.26.0-rocm7.14.0` | rocm714 | upstream `v0.26.0` |
| `v0.26.0-extras-rocm7.14.0` | rocm714 | extras-fork `v0.26.0-extras` |

The unqualified tags (`v0.26.0`, `v0.26.0-extras`) are aliases promoted onto
the default (rocm720) combinations. Qualified tags (`-rocm7.14.0`)
are produced directly by their images.

### Legacy tags (preserved, never overwritten)

`v0.22.1` and `v0.22.1_base` predate this pipeline. They are listed in
`[project].reserved_tags`; the validator rejects any image or alias that
claims them (`ReservedTag`). The pipeline never rebuilds, retags, or
repurposes them.

## Engine contract: Podman first

- Podman is the canonical engine. `--engine=auto` (default, or the
  `CONTAINER_ENGINE` environment variable) picks Podman when it is on PATH,
  else Docker, else fails with `EngineNotFound`.
- `--engine=podman` / `--engine=docker` force a specific engine and fail if
  it is missing.
- The Dockerfiles use standard Dockerfile syntax only. There is no
  Buildx-only syntax and no Docker-specific extension; the same files build
  under both engines.
- Command rendering is deterministic: Podman and Docker renderings differ
  only in the executable name and Docker's `--pull` flag. `--dry-run` prints
  the rendered commands (shell-quoted, one per line) and performs no side
  effects — not even an engine invocation beyond a presence check.

## Runtime environment defaults

Two RDNA platform defaults are baked into the application image; both are
runtime-overridable:

| Variable | Baked value | Override |
|----------|-------------|----------|
| `VLLM_ROCM_USE_AITER` | `0` | `-e VLLM_ROCM_USE_AITER=1` to opt AITER back in (CDNA-only; off is correct on RDNA) |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` | `-e FLASH_ATTENTION_TRITON_AMD_ENABLE=0` to opt out (Triton FA is the working RDNA backend) |

Deliberately **not** baked:

- `VLLM_RDNA_FORCE_FP16` — left unset. RDNA2 lacks native BF16, but forcing
  FP16 globally changes numerics for every model; opt in per launch if the
  workload needs it.
- hipBLASLt disable variables — documented below, opt-in only.

## Build inputs owned by the config

Per `[bases.*]` entry:

| Field | Consumed as | Notes |
|-------|-------------|-------|
| `pytorch_index_url` | `PYTORCH_INDEX_URL` build arg | Wheel index for `uv pip install --pre --force-reinstall` of torch/triton/torchvision/torchaudio; required, no project default |
| `pytorch_version` / `triton_version` | version pins in the index install | Empty string installs unpinned |
| `flash_attention` | four `FLASH_ATTENTION_*` build args | Optional sub-table; `install = "base"` builds FA into the base image, `"vllm"` into the application image, `"none"` (or absent) skips it. Clone source defaults to `Dao-AILab/flash-attention` at `main` |

The application build derives `--torch-backend` from the base ROCm version
mechanically (`7.2.0` → `rocm7.2`, `7.14.0` → `rocm7.14`) and builds vLLM
editable with `VLLM_USE_PRECOMPILED=0 ENABLE_CK=0
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE ROCM_HOME=/opt/rocm`.

sccache is available behind `USE_SCCACHE=1` build args on both Dockerfiles
(HIP compiler wrappers in `/opt/sccache-wrappers`, off by default).

## hipBLASLt on gfx1030 (documented, not disabled by default)

hipBLASLt on gfx1030 falls back to non-MatrixInstruction kernels (V_DOT2/FMA)
and is generally slower than rocBLAS for standard GEMM. Some ROCm releases
also ship `TensileLibrary_lazy_gfx1030.dat` as a broken symlink, which makes
hipBLASLt's TunableOp path fail with `HIPBLAS_STATUS_INVALID_VALUE`. Real
gfx1030 hipBLASLt libraries exist in the `hipblaslt-library-generator/`
project.

The image does **not** disable hipBLASLt. If you hit the lazy-catalog failure
or a GEMM regression on gfx1030, opt in at launch:

```bash
podman run --rm --device amd.com/gpu=all \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e DISABLE_ADDMM_CUDA_LT=1 \
  -e ROCBLAS_USE_HIPBLASLT=0 \
  -e PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 \
  docker.io/blivioniag/vllm-rdna:v0.26.0-extras ...
```

All four are needed together: `TORCH_BLAS_PREFER_HIPBLASLT=0` alone does not
stop TunableOp from probing hipBLASLt.

## Why bases pin an exact ROCm version

Each `[bases.*]` entry pins an exact ROCm version, its own upstream
`base_image`, and its digest. This is deliberate: RCCL regressed in newer
7.2.x point releases on RDNA multi-GPU (TP>1), so bases are promoted
explicitly after validation instead of tracking a floating minor version. A
new point release gets its own `[bases.*]` entry; existing images keep
building against the old, known-good stack until the new one is validated.

## External build machine

Builds run on an **external Podman-based machine**, not on any host this
repository names:

- No fixed `/home/...` path or runner hostname appears anywhere — configs,
  Dockerfiles, and workflow files are all host-path-free, and this is
  enforced by `tools/validate.py` (`HostPathNotAllowed`) and `ci/lint.py`
  (`HostPathInWorkflow`).
- CI runners are selected by labels. The GPU smoke job requires a
  self-hosted runner labeled `[self-hosted, linux, rdna]` attached to
  RDNA-equipped hardware with Podman and CDI (`--device amd.com/gpu=all`)
  available.
- Registry credentials are supplied exclusively via the `REGISTRY_USER` and
  `REGISTRY_TOKEN` secrets. `publish` and `promote` refuse to run live
  without them, and secret values are never printed.

## End-to-end release procedure

Example: releasing `v0.26.0` / `v0.26.0-extras` on the rocm720 base. All
commands run from the repository root.

1. **Edit the config** — `config/rocm-7.2.0.toml`:
   - add or edit a `[bases.*]` entry with its own `base_image` and
     `base_digest` (per-base pinning; there is no project-level default);
   - add or edit a `[sources.*]` entry with a `ref` and its immutable
     40-hex `commit`;
   - add an `[[images]]` entry for each new approved base x source
     combination (and an `[aliases]` entry if the combination should claim
     an unqualified tag).
2. **Run the unit suite locally:**
   ```bash
   python -m pytest vllm-rdna-docker/tests/ -q
   ```
3. **Validate the config:**
   ```bash
   python vllm-rdna-docker/tools/validate.py \
     --config vllm-rdna-docker/config/rocm-7.2.0.toml
   ```
4. **Resolve the matrix** (verify ref<->commit, tags, and approvals):
   ```bash
   python vllm-rdna-docker/tools/resolve.py \
     --config vllm-rdna-docker/config/rocm-7.2.0.toml \
     [--image vllm-026-extras-rocm720] [--json]
   ```
5. **Build the base image** — on the external Podman runner:
   ```bash
   python vllm-rdna-docker/tools/build.py build-base \
     --config vllm-rdna-docker/config/rocm-7.2.0.toml \
     --base rocm720 [--dry-run]
   ```
6. **Build the application image** — on the external Podman runner:
   ```bash
   python vllm-rdna-docker/tools/build.py build-vllm \
     --config vllm-rdna-docker/config/rocm-7.2.0.toml \
     --image vllm-026-extras-rocm720 [--dry-run]
   ```
7. **Verify the image** — on the external Podman runner (OCI labels,
   architecture metadata, digest identity, non-empty SPDX SBOM):
   ```bash
   python vllm-rdna-docker/tools/build.py verify \
     --config vllm-rdna-docker/config/rocm-7.2.0.toml \
     --image vllm-026-extras-rocm720
   ```
8. **Publish** — on the external Podman runner, with `REGISTRY_USER` and
   `REGISTRY_TOKEN` set in the environment (pushes the immutable tag; never
   logs credentials):
   ```bash
   python vllm-rdna-docker/tools/build.py publish \
     --config vllm-rdna-docker/config/rocm-7.2.0.toml \
     --image vllm-026-extras-rocm720
   ```
9. **Promote aliases** — point the unqualified tag at the verified digest
   (pull + tag + push; never a rebuild):
   ```bash
   python vllm-rdna-docker/tools/build.py promote \
     --config vllm-rdna-docker/config/rocm-7.2.0.toml \
     --alias v0.26.0-extras
   ```

## CI

Two thin adapters invoke the same repository-owned CLI; neither contains
build logic:

- `.github/workflows/build.yml` (GitHub Actions)
- `.gitea/workflows/build.yml` (Gitea Actions)

The `build` job lints the adapters, runs the unit suite, and exercises the
CLI end-to-end in dry-run mode on a generic runner. The `gpu-smoke` job is
manual-trigger only and runs real builds plus `verify` on the external
Podman runner labeled `[self-hosted, linux, rdna]`. Both adapters are gated
on the `REGISTRY_USER` / `REGISTRY_TOKEN` secrets; a preflight refuses to
build or publish when they are absent. Evidence is uploaded as the
`vllm-rdna-evidence` artifact. See [`ci/README.md`](ci/README.md) for the
full contract.

## Lint

```bash
python vllm-rdna-docker/ci/lint.py            # lint both committed adapters
python vllm-rdna-docker/ci/lint.py FILE ...   # lint specific files
```

`ci/lint.py` enforces the CI contract mechanically: every workflow must call
`python vllm-rdna-docker/tools/build.py`, must reference the `REGISTRY_USER`
/ `REGISTRY_TOKEN` secrets, and must not contain absolute host paths.
Violations are named errors with exit code 1.

## Rollback

Publication is immutable; rollback never rebuilds.

- **Alias rollback:** an alias promotion is `podman pull` + `podman tag` +
  `podman push` of a digest that was already verified and published. To move
  an alias back, promote the previous verified digest again — no compilation
  is involved.
- **Image rollback:** to restore a previous image, `podman pull` the
  previous digest, `podman tag` it with the desired tag, and `podman push`.
- **Legacy images:** `v0.22.1` and `v0.22.1_base` are outside this pipeline
  entirely and are never touched by a rollback.

## Adding a new ROCm version

1. Add a `[bases.<key>]` entry with its own `base_image` (for example
   `rocm/dev-ubuntu-22.04:7.x-complete`) and its own pinned `base_digest`.
   Each base pins its upstream image independently.
2. Add the new base id to the `compatible_bases` list of every source that
   is approved to build on it.
3. Add `[[images]]` entries for the new approved combinations, with
   qualified tags (for example `v0.26.0-rocm7.x`).
4. Run the release procedure above (validate, resolve, then build / verify /
   publish on the external runner).

## Adding a new vLLM fork version

1. Add a `[sources.<key>]` entry with `variant = "extras-fork"`, the fork
   `repository`, the compatibility `version`, the `ref` (for example
   `v0.27.0-extras`), and the immutable 40-hex `commit` the ref must resolve
   to.
2. Set `compatible_bases` to the base ids the fork is approved against.
3. Add `[[images]]` entries for the new combinations. Public tags are
   declared explicitly in the image entries — they are never derived from
   branch names.
4. Run the release procedure above.

## Evidence

Verification writes SBOMs and check logs under the project's evidence
directory (`.omo/evidence/`), and CI uploads the same tree as the
`vllm-rdna-evidence` artifact. One log or artifact per release step keeps
every published tag traceable to its config hash, source commit, and base
digest.
