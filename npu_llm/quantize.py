"""Download + quantize HuggingFace models to OpenVINO IR (sym-int4 by default).

Two paths:
* **Prequantized** repos (already OpenVINO IR) are downloaded as-is.
* Everything else is exported with ``optimum-cli export openvino`` at the
  requested weight format. The default, ``sym-int4``, is what the Intel NPU
  wants for fast weight-compressed inference.
"""

from __future__ import annotations

import subprocess
import sys

from . import paths, registry
from .console import console, good, info, warn


# Map our friendly quant names to optimum-cli flags.
_QUANT_FLAGS: dict[str, list[str]] = {
    "sym-int4": ["--weight-format", "int4", "--sym", "--ratio", "1.0", "--group-size", "128"],
    "int4": ["--weight-format", "int4", "--ratio", "1.0", "--group-size", "128"],
    "int8": ["--weight-format", "int8"],
}


def _download_prequantized(repo: str, out_dir) -> None:
    from huggingface_hub import snapshot_download

    info(f"Downloading pre-optimized OpenVINO model {repo} ...")
    # Use the default HF hub cache; a custom cache_dir has been observed to
    # interact badly with hf-xet and corrupt downloads.
    snapshot_download(repo_id=repo, local_dir=str(out_dir))


def _export(repo: str, out_dir, quant: str) -> None:
    flags = _QUANT_FLAGS.get(quant)
    if flags is None:
        raise ValueError(f"Unknown quant '{quant}'. Choose from {', '.join(_QUANT_FLAGS)}.")

    warn(f"Exporting + quantizing {repo} to {quant} (RAM/CPU intensive, one-time) ...")
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli",
        "export", "openvino",
        "-m", repo,
        *flags,
        "--trust-remote-code",
        str(out_dir),
    ]
    console.print(f"[muted]$ {' '.join(cmd)}[/muted]")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"optimum-cli export failed (exit {result.returncode}). "
            "The model may be unsupported for NPU export."
        )


def prepare(name: str, quant: str = "sym-int4", *, spec: registry.ModelSpec | None = None) -> str:
    """Ensure an OpenVINO IR export exists for ``name``; return the resolved repo id.

    ``name`` may be a registry alias or a raw HF repo id. Idempotent: if the IR
    already exists on disk it returns immediately.
    """
    repo, resolved_spec = registry.resolve(name)
    spec = spec or resolved_spec
    out_dir = paths.model_path(repo)

    if paths.is_prepared(repo):
        good(f"Model already prepared: {repo}")
        return repo

    prequant = (spec.prequantized if spec else False) or registry.looks_prequantized(repo)

    # Validate + gate-check only for unknown/raw repos (aliases are trusted).
    if spec is None:
        if not registry.model_exists(repo):
            raise SystemExit(f"Model '{repo}' not found on HuggingFace (check spelling / connection).")
        if registry.is_gated(repo):
            registry.ensure_hf_login()
    else:
        # A gated alias (e.g. Mistral/Llama) still needs a token.
        if registry.is_gated(repo):
            registry.ensure_hf_login()

    out_dir.mkdir(parents=True, exist_ok=True)
    if prequant:
        _download_prequantized(repo, out_dir)
    else:
        _export(repo, out_dir, quant)

    if not paths.is_prepared(repo):
        raise RuntimeError(f"Preparation finished but no openvino_model.xml found in {out_dir}.")
    good(f"Prepared {repo} -> {out_dir}")
    return repo
