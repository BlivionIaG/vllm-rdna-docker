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

target "base-rocm720" {
  dockerfile = "Dockerfile.base"
  tags       = ["docker.io/blivioniag/rocm-rdna:7.2.0"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE        = "rocm/dev-ubuntu-22.04:7.2-complete"
    BASE_DIGEST       = "sha256:a1b2c3d4e5f60718293a4b5c6d7e8f900a1b2c3d4e5f60718293a4b5c6d7e8f9"
    ROCM_VERSION      = "7.2.0"
    PYTORCH_VERSION   = "2.12.0"
    TRITON_VERSION    = "3.5.1"
    PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/rocm7.2"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    BASE_TAG          = "7.2.0"
  }
}

target "base-rocm714" {
  dockerfile = "Dockerfile.base"
  tags       = ["docker.io/blivioniag/rocm-rdna:7.14.0"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE         = "rocm/dev-ubuntu-22.04:7.14.0-full"
    BASE_DIGEST        = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    ROCM_VERSION       = "7.14.0"
    PYTORCH_VERSION    = "2.12.0+rocm7.14.0"
    TORCHVISION_VERSION = "0.27.0+rocm7.14.0"
    TORCHAUDIO_VERSION  = "2.11.0+rocm7.14.0"
    TRITON_VERSION     = "3.7.1"
    PYTORCH_INDEX_URL  = "https://repo.amd.com/rocm/whl-multi-arch/"
    PYTORCH_ROCM_ARCH  = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    INSTALL_ROCM_SDK_DEVICE = "1"
    BASE_TAG           = "7.14.0"
  }
}

# ---------------------------------------------------------------------------
# vLLM application images — one per (source, base) pair
# ---------------------------------------------------------------------------

target "vllm-0260-rocm720" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.26.0"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.2.0"
    VLLM_REPOSITORY  = "https://github.com/vllm-project/vllm.git"
    VLLM_REF         = "v0.26.0"
    VLLM_COMMIT      = "568afb3a13806beb53bb2e6bd518269357b237c0"
    VLLM_VARIANT     = "upstream"
    TORCH_BACKEND    = "rocm7.2"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.26.0"
    VLLM_PATCH_FILE  = "patches/v0.26.0-rocm-platforms.patch"
  }
}

target "vllm-0260-rocm714" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.26.0-rocm7.14.0"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.14.0"
    VLLM_REPOSITORY  = "https://github.com/vllm-project/vllm.git"
    VLLM_REF         = "v0.26.0"
    VLLM_COMMIT      = "568afb3a13806beb53bb2e6bd518269357b237c0"
    VLLM_VARIANT     = "upstream"
    TORCH_BACKEND    = "rocm7.14"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.26.0-rocm7.14.0"
    VLLM_PATCH_FILE  = "patches/v0.26.0-rocm-platforms.patch"
  }
}

target "vllm-0260-rocm720-extras" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.26.0-extras"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.2.0"
    VLLM_REPOSITORY  = "https://github.com/BlivionIaG/vllm.git"
    VLLM_REF         = "v0.26.0-extras"
    VLLM_COMMIT      = "9f3b6d1a8c5e0274b6d8a0c2e4f6a8c0d2e4f6a8"
    VLLM_VARIANT     = "extras-fork"
    TORCH_BACKEND    = "rocm7.2"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.26.0-extras"
    VLLM_PATCH_FILE  = "patches/v0.26.0-rocm-platforms.patch"
  }
}

target "vllm-0260-rocm714-extras" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.26.0-extras-rocm7.14.0"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.2.0"
    VLLM_REPOSITORY  = "https://github.com/BlivionIaG/vllm.git"
    VLLM_REF         = "v0.26.0-extras"
    VLLM_COMMIT      = "9f3b6d1a8c5e0274b6d8a0c2e4f6a8c0d2e4f6a8"
    VLLM_VARIANT     = "extras-fork"
    TORCH_BACKEND    = "rocm7.2"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.26.0-extras"
    VLLM_PATCH_FILE  = "patches/v0.26.0-rocm-platforms.patch"
  }
}

target "vllm-0271-rocm720" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.27.1"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.2.0"
    VLLM_REPOSITORY  = "https://github.com/vllm-project/vllm.git"
    VLLM_REF         = "v0.27.1"
    VLLM_COMMIT      = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
    VLLM_VARIANT     = "upstream"
    TORCH_BACKEND    = "rocm7.2"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.27.1"
    VLLM_PATCH_FILE  = "patches/v0.27.1-rocm-platforms.patch"
  }
}

target "vllm-0271-rocm714" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.27.1-rocm7.14.0"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.14.0"
    VLLM_REPOSITORY  = "https://github.com/vllm-project/vllm.git"
    VLLM_REF         = "v0.27.1"
    VLLM_COMMIT      = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
    VLLM_VARIANT     = "upstream"
    TORCH_BACKEND    = "rocm7.14"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.27.1-rocm7.14.0"
    VLLM_PATCH_FILE  = "patches/v0.27.1-rocm-platforms.patch"
    VLLM_FINAL_PATCH_FILE = "patches/v0.27.1-rocm-platform-detect.patch"
  }
}

target "vllm-0271-rocm720-extras" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.27.1-extras"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.2.0"
    VLLM_REPOSITORY  = "https://github.com/BlivionIaG/vllm.git"
    VLLM_REF         = "v0.27.1-extras"
    VLLM_COMMIT      = ""
    VLLM_VARIANT     = "extras-fork"
    TORCH_BACKEND    = "rocm7.2"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.27.1-extras"
    VLLM_PATCH_FILE  = "patches/v0.27.1-rocm-platforms.patch"
  }
}

target "vllm-0271-rocm714-extras" {
  dockerfile = "Dockerfile.vllm"
  tags       = ["docker.io/blivioniag/vllm-rdna:v0.27.1-extras-rocm7.14.0"]
  platforms  = ["linux/amd64"]
  target     = "final"
  args = {
    BASE_IMAGE       = "docker.io/blivioniag/rocm-rdna:7.14.0"
    VLLM_REPOSITORY  = "https://github.com/BlivionIaG/vllm.git"
    VLLM_REF         = "rdna2_extras"
    VLLM_COMMIT      = "3e05abc9bdb92100b0fec7a91e856c147dc6849c"
    VLLM_VARIANT     = "extras-fork"
    TORCH_BACKEND    = "rocm7.14"
    PYTORCH_ROCM_ARCH = "gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    IMAGE_TAG        = "v0.27.1-extras-rocm7.14.0"
    VLLM_PATCH_FILE  = "patches/v0.27.1-extras-rocm-platforms.patch"
    VLLM_FINAL_PATCH_FILE = "patches/v0.27.1-extras-platform-detect.patch"
  }
}

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

group "all-bases" {
  targets = ["base-rocm720", "base-rocm714"]
}

group "all-vllm" {
  targets = [
    "vllm-0260-rocm720",
    "vllm-0260-rocm714",
    "vllm-0260-rocm720-extras",
    "vllm-0260-rocm714-extras",
    "vllm-0271-rocm720",
    "vllm-0271-rocm714",
    "vllm-0271-rocm720-extras",
    "vllm-0271-rocm714-extras",
  ]
}

group "all" {
  targets = ["all-bases", "all-vllm"]
}
