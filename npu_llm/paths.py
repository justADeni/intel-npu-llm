"""Filesystem layout for models, NPU blob cache and HuggingFace downloads.

Everything lives under a single user-writable home directory so the tool works
the same whether it is run from a git checkout or installed with ``pip``.

Layout (default ``<platform cache>/npu-llm`` or ``$NPU_LLM_HOME``)::

    <home>/
      models/        # OpenVINO IR (int4) exports, one dir per model
      blobs/         # NPU-compiled .blob cache (keyed by model + context length)

(HuggingFace source downloads use the standard HF hub cache, not this dir.)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from platformdirs import user_cache_dir

_APP = "npu-llm"


def home() -> Path:
    """Root directory for all npu-llm data. Override with ``$NPU_LLM_HOME``."""
    override = os.environ.get("NPU_LLM_HOME")
    root = Path(override).expanduser() if override else Path(user_cache_dir(_APP, appauthor=False))
    root.mkdir(parents=True, exist_ok=True)
    return root


def models_dir() -> Path:
    d = home() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def blobs_dir() -> Path:
    d = home() / "blobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_dirname(repo_id: str) -> str:
    """Turn a HF repo id (``org/name``) into a nested, filesystem-safe path fragment."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", repo_id)


def model_path(repo_id: str) -> Path:
    """Directory where the OpenVINO IR for ``repo_id`` is (or will be) stored."""
    return models_dir() / safe_dirname(repo_id)


def blob_path(repo_id: str, ctx: int) -> Path:
    """Path of the cached NPU-compiled blob for ``repo_id`` at context length ``ctx``."""
    key = f"{safe_dirname(repo_id)}_ctx{ctx}.blob"
    return blobs_dir() / key


def is_prepared(repo_id: str) -> bool:
    """True if an OpenVINO IR export already exists locally for ``repo_id``."""
    return (model_path(repo_id) / "openvino_model.xml").is_file()
