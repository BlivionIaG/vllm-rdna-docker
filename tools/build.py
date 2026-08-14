#!/usr/bin/env python3
"""Engine-neutral build CLI for vllm-rdna-docker. Podman is canonical;
Docker is supported as an adapter (never a second implementation).

Subcommands: validate, resolve, build-base, build-vllm, verify, publish,
promote. All orchestration lives in ``tools._build_cli``; this module is
a thin entry point so the user-facing CLI keeps its familiar location
(``python vllm-rdna-docker/tools/build.py ...``).

Detailed usage, properties, and exit codes: ``tools/_build_cli.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:  # pytest / package mode (vllm-rdna-docker/ on sys.path)
    from tools._build_cli import main
except ModuleNotFoundError:  # script mode: python tools/build.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools._build_cli import main


if __name__ == "__main__":
    raise SystemExit(main())