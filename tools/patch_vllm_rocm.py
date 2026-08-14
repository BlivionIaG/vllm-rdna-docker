#!/usr/bin/env python3
"""Patch vllm/platforms/rocm.py to fix the v0.26.0 circular import bug.

Two changes to vllm/platforms/rocm.py:

1. _get_gcn_arch() fallback path: remove the `logger.warning_once(...)`
   call. That call triggers `vllm.distributed.parallel_state` →
   `vllm.utils.system_utils` → `from vllm.platforms import current_platform`
   while vllm/platforms/__init__.py is still loading and hasn't exported
   `current_platform` yet. The warning is purely informational — the
   fallback to torch.cuda is the actual behavior.

2. Module-level execution: wrap the `_GCN_ARCH = _get_gcn_arch()` and
   subsequent `_ON_GFX*` assignments in a try/except. In the build
   context (no GPU), amdsmi and torch.cuda both fail; the flags should
   default to False/empty strings rather than crashing the import.

Usage: python patch_vllm_rocm.py <path-to-rocm.py>
"""
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <path-to-rocm.py>", file=sys.stderr)
    sys.exit(1)

path = Path(sys.argv[1])
src = path.read_text()

# --- Patch 1: remove the logger.warning_once() call in _get_gcn_arch() ---
# The multi-line call is the only place logger.warning_once is used in this
# function's fallback path. Replace it with a comment + pass so the
# fallback to torch.cuda still works.
old_warning = '''        logger.warning_once(
            "Failed to get GCN arch via amdsmi, falling back to torch.cuda. "
            "This will initialize CUDA and may cause "
            "issues if CUDA_VISIBLE_DEVICES is not set yet."
        )'''
new_warning = '''        # vllm v0.26.0 ROCm bug: logger.warning_once() here triggers
        # vllm.distributed.parallel_state → vllm.utils.system_utils →
        # from vllm.platforms import current_platform, which is not yet
        # exported while vllm/platforms/__init__.py is still loading.
        # The warning is informational; skip it to avoid the circular import.'''

if old_warning not in src:
    print("ERROR: warning_once block not found", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_warning, new_warning)

# --- Patch 2: wrap module-level execution in try/except ---
# The original code runs `_GCN_ARCH = _get_gcn_arch()` and the resulting
# `_ON_GFX*` flags at module load time. In the build context (no GPU),
# both amdsmi and torch.cuda fail. We default the flags to safe empty
# values so the module imports cleanly.
old_block = '''# Resolve once at module load. Uses amdsmi (no CUDA init) so Ray workers
# can still set CUDA_VISIBLE_DEVICES after import.
# These are plain Python bools — fully torch.compile/Dynamo safe.
_GCN_ARCH = _get_gcn_arch()

_ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
_ON_GFX11 = "gfx11" in _GCN_ARCH
_ON_GFX1100 = "gfx1100" in _GCN_ARCH
_ON_GFX1151 = "gfx1151" in _GCN_ARCH
_ON_GFX12X = any(arch in _GCN_ARCH for arch in ["gfx12"])
_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])
_ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
_ON_GFX90A = "gfx90a" in _GCN_ARCH
_ON_GFX942 = "gfx942" in _GCN_ARCH
_ON_GFX950 = "gfx950" in _GCN_ARCH'''

new_block = '''# Lazy execution to avoid circular import at module load time
# (vllm v0.26.0 ROCm bug). Wrap in try/except so a missing GPU or
# broken amdsmi in the build context doesn't crash the import.
# The flags are eager-resolved by callers (vllm/platforms/__init__.py
# imports `current_platform` which retries this on first use).
try:
    _GCN_ARCH = _get_gcn_arch()
    _ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
    _ON_GFX11 = "gfx11" in _GCN_ARCH
    _ON_GFX1100 = "gfx1100" in _GCN_ARCH
    _ON_GFX1151 = "gfx1151" in _GCN_ARCH
    _ON_GFX12X = any(arch in _GCN_ARCH for arch in ["gfx12"])
    _ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])
    _ON_GFX9 = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
    _ON_GFX90A = "gfx90a" in _GCN_ARCH
    _ON_GFX942 = "gfx942" in _GCN_ARCH
    _ON_GFX950 = "gfx950" in _GCN_ARCH
except Exception:
    # amdsmi and torch.cuda both failed (build context, no GPU).
    # Leave flags at safe empty defaults; callers should re-resolve
    # via current_platform instead of relying on these flags.
    _GCN_ARCH = ""
    _ON_GFX1X = False
    _ON_GFX11 = False
    _ON_GFX1100 = False
    _ON_GFX1151 = False
    _ON_GFX12X = False
    _ON_MI3XX = False
    _ON_GFX9 = False
    _ON_GFX90A = False
    _ON_GFX942 = False
    _ON_GFX950 = False'''

if old_block not in src:
    print("ERROR: module-level block not found", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_block, new_block)

path.write_text(src)
print(f"Patched {path}")
print("  - removed logger.warning_once() in _get_gcn_arch() fallback")
print("  - wrapped module-level GCN arch detection in try/except")
