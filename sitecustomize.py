"""Hip runtime eager-init for vLLM workers on ROCm/RDNA.

vLLM v0.27.1 workers assert ``local_rank < torch.accelerator.device_count()``
in ``init_device()``. Without eager ``torch.cuda.init()`` at interpreter start
the count reads 0 and the assert fails:

    AssertionError: DP adjusted local rank N is out of bounds for 0 devices.

Critical: do NOT gate the init on ``torch.cuda.is_available()``. vLLM sets
``PYTORCH_NVML_BASED_CUDA_CHECK=1`` at import; spawned workers inherit it and
``torch.cuda.is_available()`` then uses the NVML path, which returns False on
ROCm (no NVIDIA NVML). The HIP runtime is present whenever
``torch.version.hip`` is set, so force the init regardless.
"""
import os
import warnings

try:
    import torch
except Exception:
    torch = None

if torch is not None and torch.version.hip:
    try:
        torch.cuda.init()
    except Exception:
        pass
    try:
        os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
        torch.backends.cuda.preferred_blas_library("hipblas")
    except Exception:
        pass