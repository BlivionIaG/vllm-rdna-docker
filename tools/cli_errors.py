"""Named errors owned by the vllm-rdna-docker build CLI (``tools/build.py``).

These are CLI-only failures — they can only be raised once a config has
already passed ``tools.validate`` and ``tools.resolve``, so they live outside
the validator. Every error subclasses ``tools.validate.ConfigError`` so
``str(err)`` renders ``Name(field=value, ...)`` and ``err.context`` carries
the structured fields, exactly like the validator/resolver errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pytest / package mode (vllm-rdna-docker/ on sys.path)
    from tools.validate import ConfigError
except ModuleNotFoundError:  # script mode: python tools/build.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.validate import ConfigError


class EngineNotFound(ConfigError):
    """The requested container engine (podman/docker/auto) is not on PATH.

    ``engine`` is the value the caller asked for: ``"podman"`` or
    ``"docker"`` when forced, or ``"auto"`` when neither candidate was
    found.
    """

    def __init__(self, engine: str) -> None:
        super().__init__(engine=engine)


class CommandFailed(ConfigError):
    """A container engine invocation exited non-zero (non-dry-run only)."""

    def __init__(self, command: str, exit_code: int) -> None:
        super().__init__(command=command, exit_code=exit_code)
