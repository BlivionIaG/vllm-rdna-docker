# vllm-rdna-docker

Builds ROCm/RDNA base images and vLLM application images for AMD consumer
GPUs. Two Dockerfiles, one bake graph, one CI workflow. No Python layer,
no config validator, no custom linter.

## What ships

| Image | Tag pattern | Source |
|---|---|---|
| `docker.io/blivioniag/rocm-rdna` | `7.2.0`, `7.14.0` | `Dockerfile.base` |
| `docker.io/blivioniag/vllm-rdna` | `v0.26.0`, `v0.26.0-extras`, plus `-rocm7.14.0` variants | `Dockerfile.vllm` |

The `v0.26.0-extras` variant is a fork with extra kernels.

## Build locally

The build graph is `docker-bake.hcl` (the source of truth). Run it with
stock `docker buildx bake`:

```bash
# Build everything
docker buildx bake --file docker-bake.hcl all

# Just the vLLM images
docker buildx bake --file docker-bake.hcl all-vllm

# A single target
docker buildx bake --file docker-bake.hcl vllm-upstream026-rocm720

# Print the build plan without building
docker buildx bake --file docker-bake.hcl --print all
```

Bake runs the targets in parallel. The seven RDNA targets
(`gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201`) are
baked into each image.

## CI

`.github/workflows/build.yml` runs on:

- **Tag push** (`v*`): builds `all` and pushes to `docker.io/blivioniag/`.
- **Manual dispatch**: pick a single target; optionally push.

Uses stock `docker/setup-buildx-action`, `docker/login-action`,
`docker/metadata-action`, and `docker/bake-action`. GHA cache enabled.

## Adding a new base

1. Copy a `target "base-<id>"` block in `docker-bake.hcl`. Set
   `BASE_IMAGE`, `ROCM_VERSION`, `PYTORCH_VERSION`, `TRITON_VERSION`,
   `PYTORCH_INDEX_URL`, `BASE_TAG` to the values you need.
2. Add a `target "vllm-<source>-<id>"` block for each existing source
   (`upstream026`, `extras026`, ...). Set `BASE_IMAGE` to your new
   base's full ref and `TORCH_BACKEND` to `rocm<X>.<Y>` derived from
   the ROCm version.
3. Add the new ids to the `all-bases` and `all-vllm` groups.

## Adding a new vLLM source

1. Add a `target "vllm-<source>-<base>"` block for each base. Set
   `VLLM_REPOSITORY`, `VLLM_REF`, `VLLM_COMMIT`, `VLLM_VARIANT`,
   `IMAGE_TAG`.
2. Add the new ids to the `all-vllm` group.

The base default is implicit: the unqualified tag (`v0.26.0`,
`v0.26.0-extras`) is whatever the first base in `docker-bake.hcl`
produces. If you want a different default, reorder the base targets.

## The RDNA patch

`Dockerfile.vllm` runs a small Python `RUN` step after the vLLM clone to
patch `vllm/platforms/rocm.py` for the build context:

1. Skips a `logger.warning_once()` call that triggers a circular import
   at module load.
2. Wraps the module-level GCN arch detection in `try/except` so a missing
   GPU during build doesn't crash the import.

The patch is inline in the Dockerfile (no external file, no separate
Python script). It's a one-time fix for the v0.26.0 ROCm circular
import bug; remove it once upstream is fixed.
