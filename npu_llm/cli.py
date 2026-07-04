"""Command-line interface for npu-llm.

Composable flags — a single invocation can prepare a model, run it, serve an
OpenAI-compatible API and show a live dashboard, e.g.::

    npu-llm --model tinyllama --serve 4444 --display
"""

from __future__ import annotations

import argparse
import sys

from . import __version__, paths, quantize, registry
from .console import console, error, good, info, muted, warn


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="npu-llm",
        description="Run, quantize and serve LLMs on the Intel NPU (OpenVINO GenAI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  npu-llm --list\n"
            "  npu-llm --pull tinyllama\n"
            "  npu-llm --model tinyllama\n"
            "  npu-llm --model tinyllama --serve 4444 --display\n"
            "  npu-llm --model Qwen/Qwen3-4B-Thinking-2507 --ctx 2048\n"
        ),
    )
    p.add_argument("-m", "--model", metavar="ALIAS|HF_REPO", help="model alias or HuggingFace repo id")
    p.add_argument("--serve", metavar="PORT", type=int, help="expose an OpenAI-compatible API on PORT")
    p.add_argument("--host", default="127.0.0.1", help="host for --serve (default 127.0.0.1)")
    p.add_argument("--display", action="store_true", help="show a live NPU/usage dashboard")
    p.add_argument("--pull", metavar="ALIAS|HF_REPO", help="download + quantize a model, then exit")
    p.add_argument("--list", action="store_true", help="list built-in aliases and cached models")
    p.add_argument("--info", action="store_true", help="print NPU hardware summary and exit")
    p.add_argument("--ctx", type=int, default=1024, help="max prompt/context length (default 1024)")
    p.add_argument(
        "--compact",
        type=int,
        metavar="PCT",
        help="in chat, auto-summarize the conversation when it reaches PCT%% of --ctx "
        "(e.g. 80), preventing context-overflow resets",
    )
    p.add_argument(
        "--quant",
        default="sym-int4",
        choices=["sym-int4", "int4", "int8"],
        help="weight quantization for export (default sym-int4)",
    )
    p.add_argument("--device", default="NPU", help="OpenVINO device: NPU (default), CPU, GPU")
    p.add_argument("-V", "--version", action="version", version=f"npu-llm {__version__}")
    return p


def _cmd_list() -> None:
    from rich.table import Table

    prepared = {p.name for p in paths.models_dir().iterdir()} if paths.models_dir().exists() else set()

    t = Table(title="Built-in model aliases", title_style="info")
    t.add_column("alias", style="good")
    t.add_column("params", style="muted", justify="right")
    t.add_column("HuggingFace repo")
    t.add_column("cached", justify="center")
    t.add_column("note", style="muted")
    for spec in registry.REGISTRY.values():
        cached = "✓" if paths.safe_dirname(spec.repo) in prepared else ""
        t.add_row(spec.alias, spec.params, spec.repo, cached, spec.note)
    console.print(t)

    # Any locally-prepared models that aren't aliases.
    alias_dirs = {paths.safe_dirname(s.repo) for s in registry.REGISTRY.values()}
    extra = sorted(prepared - alias_dirs)
    if extra:
        console.print("\n[info]Other cached models:[/info]")
        for name in extra:
            muted(f"  {name}")
    muted(f"\ncache dir: {paths.home()}")


def _pick_model() -> str:
    """Interactive fallback picker when no --model is given."""
    from rich.table import Table

    specs = list(registry.REGISTRY.values())
    t = Table(title="Select a model", title_style="info")
    t.add_column("#", justify="right", style="accent")
    t.add_column("alias", style="good")
    t.add_column("params", justify="right", style="muted")
    t.add_column("note", style="muted")
    for i, spec in enumerate(specs, 1):
        t.add_row(str(i), spec.alias, spec.params, spec.note)
    console.print(t)
    while True:
        choice = input("Select a number, or paste a HuggingFace repo id (or 'exit'): ").strip()
        if choice == "exit":
            raise SystemExit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(specs):
            return specs[int(choice) - 1].alias
        if "/" in choice or choice.lower() in registry.REGISTRY:
            return choice
        error("Invalid selection.")


def _run(model: str, args) -> None:
    from .engine import NPUEngine
    from .hardware import require_npu

    if args.device.upper() == "NPU":
        require_npu()

    # Prepare (download + quantize) if needed.
    repo = quantize.prepare(model, args.quant)

    engine = NPUEngine(repo, ctx=args.ctx, device=args.device.upper())
    engine.load()

    if args.serve:
        _run_serve(engine, args)
    else:
        _run_chat(engine, args)


def _run_serve(engine, args) -> None:
    from .server import serve

    if args.compact is not None:
        muted("--compact applies to interactive chat only; the API is stateless per request.")

    if args.display:
        from .display import Dashboard
        from .monitor import get_monitor

        dash = Dashboard(engine, get_monitor(), subtitle=f"serving :{args.serve}")
        dash.start_background()
        try:
            serve(engine, host=args.host, port=args.serve, log_level="critical")
        finally:
            dash.stop()
    else:
        serve(engine, host=args.host, port=args.serve, log_level="warning")


def _run_chat(engine, args) -> None:
    # Rich Live + input() conflict, so with --display we print per-turn stats
    # rather than a full-screen dashboard.
    if args.display:
        info("Live dashboard is used with --serve; in interactive chat, per-turn stats are shown.")
    compact = args.compact
    if compact is not None and not (1 <= compact <= 99):
        warn("--compact must be a percent between 1 and 99; ignoring.")
        compact = None
    engine.chat(stats_after=args.display, compact_pct=compact)


def _run_monitor_only(args) -> None:
    """`--display` with no model: hardware-only NPU monitor."""
    from .display import Dashboard
    from .engine import NPUEngine
    from .hardware import require_npu
    from .monitor import get_monitor

    if args.device.upper() == "NPU":
        require_npu()
    stub = NPUEngine("(no model)", ctx=0, device=args.device.upper())
    good("NPU monitor — press Ctrl+C to exit.")
    Dashboard(stub, get_monitor(), subtitle="monitor").run()


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    try:
        if args.info:
            from .hardware import print_info

            print_info()
            return

        if args.list:
            _cmd_list()
            return

        if args.pull:
            from .hardware import has_npu

            quantize.prepare(args.pull, args.quant)
            if not has_npu():
                muted("(no NPU detected — model prepared but can only run once an NPU is present)")
            return

        if args.model:
            _run(args.model, args)
            return

        if args.display and not args.serve:
            _run_monitor_only(args)
            return

        # No actionable flags → interactive picker → chat.
        model = _pick_model()
        _run(model, args)
    except KeyboardInterrupt:
        console.print("\n[muted]bye[/muted]")
        sys.exit(0)


if __name__ == "__main__":
    main()
