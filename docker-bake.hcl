# vllm-rdna-docker build graph.
#
# The source of truth. Hand-written because the matrix is small and the
# previous Python emitter was overkill.
#
# Adding a new base: copy a "target \"base-X\"" block, add the new id to
# the "all-bases" group, and add one vllm-<source>-X target per source.
#
# Adding a new vLLM source: add one vllm-<source>-<base> target per base
# and add the new ids to the "all-vllm" group.

# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------

target "base-rocm714" {
  dockerfile = "Dockerfile.base"
  tags       = ["docker.io/blivioniag/rocm-rdna:7.14.0"]
  platforms  = ["linux/amd64"]
  target     = "base"
  args = {
    BASE_IMAGE        = "rocm/dev-ubuntu-22.04:7.14.0-full"
    BASE_DIGEST       = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ROCM_VERSION      = "7.14.0"
    PYTORCH_VERSION   = "2.13.0"
    TRITON_VERSION    = "3.7.1"
    PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/rocm7.14"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    BASE_TAG          = "7.14.0"
  }
}

# ---------------------------------------------------------------------------
# vLLM application images — one per (source, base) pair
# ---------------------------------------------------------------------------

target "vllm-0271-rocm714" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.27.1"]
  platforms  = ["linux/amd64"]
  target     = "vllm"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.14.0"
    VLLM_REPOSITORY  = "https://github.com/vllm-project/vllm.git"
    VLLM_REF         = "v0.27.1"
    VLLM_COMMIT      = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
    VLLM_VARIANT     = "upstream"
    TORCH_BACKEND    = "rocm7.14"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.27.1"
    VLLM_PATCH_FILE  = ""
  }
}

target "vllm-0271-rocm714-extras" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.27.1-extras"]
  platforms  = ["linux/amd64"]
  target     = "vllm"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.14.0"
    VLLM_REPOSITORY  = "https://github.com/BlivionIaG/vllm.git"
    VLLM_REF         = "v0.27.1-extras"
    VLLM_COMMIT      = ""
    VLLM_VARIANT     = "extras-fork"
    TORCH_BACKEND    = "rocm7.14"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.27.1-extras"
    VLLM_PATCH_FILE  = ""
  }
}

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

group "all-bases" {
  targets = ["base-rocm714"]
}

group "all-vllm" {
  targets = [
    "vllm-0271-rocm714",
    "vllm-0271-rocm714-extras",
  ]
}

group "all" {
  targets = ["all-bases", "all-vllm"]
}
