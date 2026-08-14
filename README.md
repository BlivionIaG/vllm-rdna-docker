# vllm-rdna-docker

Builds ROCm/RDNA base images and vLLM application images for AMD consumer
GPUs. Two Dockerfiles, one bake graph, one CI workflow. No Python layer,
no config validator, no custom linter.

## Targets

`docker-bake.hcl` produces six images:

| Target | Tag | Notes |
|---|---|---|
| `base-rocm720` | `docker.io/blivioniag/rocm-rdna:7.2.0` | PyTorch 2.12.0, Triton 3.5.1 |
| `base-rocm714` | `docker.io/blivioniag/rocm-rdna:7.14.0` | PyTorch 2.13.0, Triton 3.7.1 |
| `vllm-0260-rocm720` | `…/vllm-rdna:v0.26.0` | vLLM `v0.26.0` @ `568afb3` |
| `vllm-0260-rocm714` | `…/vllm-rdna:v0.26.0-rocm7.14.0` | same source, ROCm 7.14.0 base |
| `vllm-0260-rocm720-extras` | `…/vllm-rdna:v0.26.0-extras` | fork with extra RDNA kernels |
| `vllm-0260-rocm714-extras` | `…/vllm-rdna:v0.26.0-extras-rocm7.14.0` | same fork, ROCm 7.14.0 base |

Groups: `all` (= `all-bases` + `all-vllm`), `all-bases`, `all-vllm`.

The seven RDNA archs `gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201`
are baked into every image via `PYTORCH_ROCM_ARCH`.

## Build locally

The build graph is `docker-bake.hcl` (the source of truth). Run it with
stock `docker buildx bake`:

```bash
# Build everything
docker buildx bake --file docker-bake.hcl all

# Just the vLLM images
docker buildx bake --file docker-bake.hcl all-vllm

# A single target
docker buildx bake --file docker-bake.hcl vllm-0260-rocm720

# Print the build plan without building
docker buildx bake --file docker-bake.hcl --print all
```

Bake runs the targets in parallel. For Podman locally, add
`--engine podman` if your buildx setup doesn't auto-detect it.

## CI

`.github/workflows/build.yml` runs on:

- **Tag push** (`v*`): builds `all` and pushes to `docker.io/blivioniag/`.
- **Manual dispatch**: pick a single target; optionally push.

Uses stock `docker/setup-buildx-action`, `docker/login-action`,
`docker/metadata-action`, and `docker/bake-action`. GHA cache enabled.

## Build arguments

`Dockerfile.vllm` accepts (all set in `docker-bake.hcl` per target):

| ARG | Purpose |
|---|---|
| `BASE_IMAGE` | Published base image (e.g. `…/rocm-rdna:7.2.0`) |
| `VLLM_REPOSITORY` | vLLM git clone URL |
| `VLLM_REF` | Git ref to clone (tag or branch, e.g. `v0.26.0`) |
| `VLLM_COMMIT` | Full 40-char commit; build fails if HEAD != commit |
| `VLLM_VARIANT` | `upstream` or `extras-fork` (recorded as a label) |
| `IMAGE_TAG` | Published tag for this image |
| `IMAGE_REPOSITORY` | Published repository |
| `PYTORCH_ROCM_ARCH` | Semicolon-joined gfx targets (`ARCH_LIST` forwards) |
| `TORCH_BACKEND` | `uv --torch-backend` value (e.g. `rocm7.2`) |
| `FLASH_ATTENTION_INSTALL` | `base` \| `vllm` \| `none` |
| `FLASH_ATTENTION_REPO` | FA clone URL (default `Dao-AILab/flash-attention`) |
| `FLASH_ATTENTION_REF` | FA branch/tag/commit to clone (default `main`) |
| `USE_SCCACHE` | `1` to wrap HIP compilation in sccache (base must be built with `USE_SCCACHE=1`) |
| `VLLM_PATCH_FILE` | Path to a `.patch` file in `patches/` to apply after the vLLM clone (e.g. `patches/v0.26.0-rocm-platforms.patch`). Empty = no patch. |
| `CONFIG_HASH` | Sha256 of the resolved config (informational label) |

## Adding a new base

1. Copy a `target "base-<id>"` block in `docker-bake.hcl`. Set
   `BASE_IMAGE`, `BASE_DIGEST`, `ROCM_VERSION`, `PYTORCH_VERSION`,
   `TRITON_VERSION`, `PYTORCH_INDEX_URL`, `BASE_TAG` to the values
   you need.
2. Add a `target "vllm-<source>-<id>"` block for each existing source
   (`upstream026`, `extras026`, ...). Set `BASE_IMAGE` to your new
   base's full ref and `TORCH_BACKEND` to `rocm<X>.<Y>` derived from
   the ROCm version.
3. Add the new id to the `all-bases` and `all-vllm` groups.

## Adding a new vLLM source

1. Add a `target "vllm-<source>-<base>"` block for each base. Set
   `VLLM_REPOSITORY`, `VLLM_REF`, `VLLM_COMMIT`, `VLLM_VARIANT`,
   `IMAGE_TAG`.
2. Add the new ids to the `all-vllm` group.

The base default is implicit: the unqualified tag (`v0.26.0`,
`v0.26.0-extras`) is whatever the first base in `docker-bake.hcl`
produces. If you want a different default, reorder the base targets.

## Adding a new RDNA patch

`patches/v0.26.0-rocm-platforms.patch` is the v0.26.0-specific fix.
For a new vLLM release:

1. Clone the upstream tag locally and check out the first commit
   where the issue is hit.
2. Edit the file(s) by hand to the desired end state.
3. Generate the diff:
   ```bash
   git diff -- vllm/platforms/rocm.py > patches/v<N>.<M>.<P>-rocm-platforms.patch
   ```
4. Verify it applies cleanly against a fresh clone:
   ```bash
   git clone --depth 1 --branch v<N>.<M>.<P> https://github.com/vllm-project/vllm.git /tmp/vllm-test
   cd /tmp/vllm-test && git apply --check /path/to/patches/v<N>.<M>.<P>-rocm-platforms.patch
   ```
5. Set `VLLM_PATCH_FILE = "patches/v<N>.<M>.<P>-rocm-platforms.patch"`
   in the `args` block of every `vllm-<source>-<base>` target that
   builds against this vLLM version in `docker-bake.hcl`. Leave it
   empty (`""`) if no patch is needed.

The `COPY patches/ /src/patches/` step in `Dockerfile.vllm` runs
unconditionally (it's cheap); the `git apply` is gated on
`VLLM_PATCH_FILE` being non-empty.

## The v0.26.0 RDNA patch

`Dockerfile.vllm` applies `patches/v0.26.0-rocm-platforms.patch` with
`git apply` after the vLLM clone. The patch fixes the v0.26.0 ROCm
build-context bug in `vllm/platforms/rocm.py`:

1. Removes a `logger.warning_once()` call that triggers a circular
   import at module load
   (vllm.distributed → vllm.utils.system_utils → `from vllm.platforms
   import current_platform`, which is still mid-load).
2. Wraps the module-level GCN arch detection in `try/except` so a
   missing GPU during build doesn't crash the import. Callers
   re-resolve via `current_platform` on first use.

The patch is a standard unified diff. It was generated with
`git diff` from a known-good patched state on a fresh v0.26.0 clone
and verified with `git apply --check`. It's a one-time fix for the
v0.26.0 ROCm circular import bug; drop the file and the `RUN git
apply` line once upstream is fixed.
