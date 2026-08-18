"""Hip runtime eager-init for vLLM workers on ROCm/RDNA.

The upstream v0.27.1 platform layer asserts
``local_rank < torch.accelerator.device_count()`` inside vLLM's worker
``init_device()``. Without an eager ``torch.cuda.init()`` at interpreter
startup, ``torch.cuda.device_count()`` returns 0 in the container (the HIP
runtime is lazily initialized only on first explicit CUDA call), so the
assertion fails:

    AssertionError: DP adjusted local rank 3 is out of bounds for 0 devices.

This mirrors the working bare-metal venv where sitecustomize forces rocBLAS
at startup (which eagerly initializes HIP).
"""
import os
import warnings

try:
    import torch
except Exception:
    torch = None

if torch is not None and torch.version.hip and torch.cuda.is_available():
    # Eagerly initialize the HIP runtime so torch.cuda.device_count() (and
    # torch.accelerator.device_count()) return the real GPU count before
    # vLLM's worker asserts local_rank < device_count(). Mirrors the working
    # bare-metal venv where sitecustomize forces rocBLAS at startup.
    try:
        torch.cuda.init()
    except Exception:
        pass
    try:
        os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
        torch.backends.cuda.preferred_blas_library("hipblas")
    except Exception:
        pass