# Changelog

All notable changes to the vllm-rdna-docker build pipeline are documented
here. The format follows Keep a Changelog; this project is pre-1.0 and the
config schema is versioned independently (`schema_version = 1`).

## [Unreleased]

### Added

- **`pytorch_index_url` per base** — each `[bases.*]` entry now declares the
  wheel index used for the torch stack (e.g.
  `https://download.pytorch.org/whl/rocm7.2`). Required field; no project
  default. Rendered as `PYTORCH_INDEX_URL` in base builds.
- **Flash Attention per base** — optional
  `[bases.<key>.flash_attention]` sub-table with `install` (`base` | `vllm` |
  `none`), `version`, `repo` (default Dao-AILab/flash-attention), and `ref`
  (default `main`). The resolver carries it into `BaseRecord` and the config
  hash; the CLI renders the four `FLASH_ATTENTION_*` build args for the
  matching image layer only. New named error
  `InvalidFlashAttentionInstall`.
- **deadsnakes Python + build toolchain in `Dockerfile.base`** — Ubuntu
  22.04 bases get Python 3.12 via the deadsnakes PPA (with retry loop),
  `get-pip.py` bootstrap, and the compile toolchain (`packaging`,
  `cmake<4`, `ninja`, `setuptools<80`, `pybind11`, `Cython`) required for
  HIP extension builds, plus `uv`.
- **Wheel-index torch install** — `Dockerfile.base` installs
  torch/triton/torchvision/torchaudio with
  `uv pip install --system --pre --force-reinstall --index-url
  ${PYTORCH_INDEX_URL}`. Version pins apply only when the config field is
  non-empty.
- **AMD SMI wheel** — built from `/opt/rocm/share/amd_smi` and installed in
  the base image; verified by the smoke checks.
- **sccache support** — `USE_SCCACHE=1` installs sccache and creates HIP
  compiler wrappers in `/opt/sccache-wrappers`; the FA build and the vLLM
  extension build opt into the wrappers per compile step via
  `HIP_CLANG_PATH` / `CMAKE_*_COMPILER_LAUNCHER`. Off by default; S3 config
  is only injected when enabled.
- **Editable vLLM build with `--torch-backend`** — `Dockerfile.vllm` builds
  with `VLLM_USE_PRECOMPILED=0 ENABLE_CK=0
  FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE ROCM_HOME=/opt/rocm` and
  `uv pip install --system -e . --no-build-isolation --torch-backend=...`;
  the backend string is derived from the base ROCm version (`7.2.0` →
  `rocm7.2`, `7.14.0` → `rocm7.14`). Smoke checks verify
  `vllm._rocm_C` imports.
- **`HSA_NO_SCRATCH_RECLAIM=1` and `HIP_FORCE_DEV_KERNARG=1`** in
  `Dockerfile.base` (RCCL stability env, upstream parity).

### Changed

- **Runtime environment policy decided** — `Dockerfile.vllm` now bakes two
  RDNA platform defaults, both runtime-overridable: `VLLM_ROCM_USE_AITER=0`
  (AITER is CDNA-only) and `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (Triton
  FA is the working RDNA backend). `VLLM_RDNA_FORCE_FP16` is deliberately
  left unset. hipBLASLt is not disabled by default; the opt-in disable
  variables for gfx1030 (`TORCH_BLAS_PREFER_HIPBLASLT=0`,
  `DISABLE_ADDMM_CUDA_LT=1`, `ROCBLAS_USE_HIPBLASLT=0`,
  `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`) are documented in `README.md`.
- **`PRIMARY_ARCH` purpose clarified** — kept in `Dockerfile.base` as the
  compile anchor for source builds (`HCC_AMDGPU_TARGET`), defaulting to
  gfx1030, with `ARCH_LIST` carrying the full RDNA set to keep build time
  down. It is a build-time selector, not a runtime preference.
- **Per-base ROCm pinning rationale documented** — bases pin an exact ROCm
  version because RCCL regressed in newer 7.2.x point releases on RDNA
  multi-GPU (TP>1); new point releases get their own `[bases.*]` entry and
  are promoted only after validation.

## [0.1.0] - 2026-08-07

Initial release of the standalone Podman-first ROCm/RDNA build pipeline.

### Added

- **Project skeleton and TOML config schema** — `vllm-rdna-docker/` layout
  with a strict, validator-enforced config contract (`config/schema.md`,
  `tools/validate.py`). Fixed seven-target architecture set
  (`gfx1030`, `gfx1100`, `gfx1101`, `gfx1150`, `gfx1151`, `gfx1200`,
  `gfx1201`) with no primary architecture; reserved legacy tags
  (`v0.22.1`, `v0.22.1_base`); named validation errors.
- **Source/base resolver and explicit compatibility matrix** —
  `tools/resolve.py` verifies ref/commit consistency, computes config,
  source, and base IDs, expands the approved `[[images]]` matrix, and
  rejects unapproved base x source combinations before any build command is
  emitted.
- **Podman-first portable Dockerfiles** — `Dockerfile.base` and
  `Dockerfile.vllm` use standard Dockerfile syntax (no Buildx-only
  features), take all parameters as build arguments from the resolved
  records, and build `blivioniag/rocm-rdna` separately from
  `blivioniag/vllm-rdna`.
- **Engine-neutral build CLI** — `tools/build.py` subcommands `validate`,
  `resolve`, `build-base`, `build-vllm`, `verify`, `publish`, and `promote`.
  Podman is canonical; Docker is an adapter selected via `--engine` or
  `CONTAINER_ENGINE`. Deterministic command rendering with `--dry-run`
  support and no side effects.
- **Verification, OCI provenance, and SBOM evidence** — `tools/verify.py`
  checks image labels against resolved config/source/base metadata, asserts
  the configured architecture set, captures manifest digests, and generates
  a non-empty SPDX SBOM (via `syft` when available) under the evidence
  directory.
- **Publication and alias promotion** — `tools/publish.py` pushes immutable
  tags to `docker.io/blivioniag/rocm-rdna` and
  `docker.io/blivioniag/vllm-rdna`, gated on `REGISTRY_USER` /
  `REGISTRY_TOKEN` environment credentials (never logged). Alias promotion
  is pull + tag + push of an already-verified digest, never a rebuild, and
  refuses reserved legacy tags.
- **GitHub and Gitea CI adapters** — thin workflow files
  (`.github/workflows/build.yml`, `.gitea/workflows/build.yml`) that invoke
  the same repository-owned CLI. Static contract lint via `ci/lint.py`
  (host-path ban, required secret references, CLI invocation). GPU smoke job
  targets the external Podman runner labeled `[self-hosted, linux, rdna]`.
- **Operations documentation** — this changelog plus `README.md` covering
  the config schema, image repositories and tags, legacy tag preservation,
  the external Podman build-machine contract, the end-to-end release
  procedure, rollback, and recipes for adding new ROCm versions and new
  extras-fork versions.

[0.1.0]: https://github.com/blivioniag/vllm-rdna-docker/releases/tag/0.1.0
