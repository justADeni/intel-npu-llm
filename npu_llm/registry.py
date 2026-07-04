"""Model alias registry and repo-id resolution.

A short ``alias`` (e.g. ``tinyllama``) maps to a HuggingFace repo id plus a bit
of metadata. Anything that is not a known alias is treated as a raw HF repo id
and validated online before use.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo: str
    params: str  # human-readable size, e.g. "1.1B"
    note: str = ""
    # Repos that already ship OpenVINO IR (int4) — download only, never re-export.
    prequantized: bool = False


# Curated, NPU-friendly defaults. Users can still pass any raw HF repo id.
REGISTRY: dict[str, ModelSpec] = {
    m.alias: m
    for m in [
        ModelSpec("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B",
                  "tiny & fast, great for smoke-testing the NPU"),
        ModelSpec("qwen3-4b-thinking", "Qwen/Qwen3-4B-Thinking-2507", "4B",
                  "reasoning model; emits <think> traces"),
        ModelSpec("phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "3.8B",
                  "strong small general model"),
        ModelSpec("phi-4-mini", "microsoft/Phi-4-mini-instruct", "3.8B",
                  "newer Phi small model"),
        ModelSpec("mistral-7b", "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov", "7B",
                  "non-gated, pre-quantized int4 for NPU (upper size for a 13-TOPS NPU)",
                  prequantized=True),
    ]
}


def looks_prequantized(repo: str) -> bool:
    """Heuristic: does this repo already ship OpenVINO IR weights?"""
    low = repo.lower()
    return "openvino" in low or low.endswith("-ov") or "int4-ov" in low or "-ov-" in low


def resolve(name: str) -> tuple[str, ModelSpec | None]:
    """Resolve an alias-or-repo to ``(repo_id, spec_or_None)``.

    Known aliases return their :class:`ModelSpec`; raw repo ids return ``None``
    for the spec (the caller validates them online).
    """
    spec = REGISTRY.get(name.lower())
    if spec is not None:
        return spec.repo, spec
    return name, None


# --- online validation / gating (used only for non-alias / raw repos) ---------

def model_exists(repo: str) -> bool:
    from huggingface_hub import model_info
    from huggingface_hub.utils import HfHubHTTPError, HFValidationError

    try:
        model_info(repo)
        return True
    except HFValidationError:
        return False
    except HfHubHTTPError as e:
        if "404" in str(e):
            return False
        raise


def is_gated(repo: str) -> bool:
    from huggingface_hub import model_info

    try:
        info = model_info(repo)
        gated = getattr(info, "gated", None)
        if gated:  # newer hub exposes .gated directly ("auto"/"manual"/False)
            return True
        return bool(info.cardData and info.cardData.get("gated"))
    except Exception:
        return False


def _hf_logged_in() -> bool:
    try:
        from huggingface_hub import whoami

        whoami()
        return True
    except Exception:
        return False


def ensure_hf_login() -> None:
    """Interactively register a HF token if not already logged in (for gated repos)."""
    if _hf_logged_in():
        return
    from huggingface_hub import login

    from .console import error, good, warn

    warn("This model is gated. A HuggingFace access token is required.")
    for _ in range(3):
        token = input("Enter HuggingFace token (or 'exit'): ").strip()
        if token == "exit":
            raise SystemExit(0)
        try:
            login(token=token, add_to_git_credential=False)
            if _hf_logged_in():
                good("Logged in to HuggingFace.")
                return
        except Exception:
            pass
        error("Login failed, try again.")
    raise SystemExit("Could not authenticate with HuggingFace.")
