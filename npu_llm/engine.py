"""NPU inference engine: compile-with-blob-cache, generate, chat REPL.

Wraps ``openvino_genai.LLMPipeline`` on the ``NPU`` device. The first run for a
given (model, context-length) pair compiles the model and exports a ``.blob``;
subsequent runs load that blob directly, which is dramatically faster.
"""

from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass, field

from . import paths
from .console import console, good, info


@dataclass
class GenStats:
    """Live + last-generation metrics, shared with the dashboard."""
    model: str = ""
    ctx: int = 0
    device: str = "NPU"
    load_seconds: float = 0.0
    # rolling live view (updated per streamed token)
    generating: bool = False
    live_tokens: int = 0
    live_tok_per_s: float = 0.0
    # last completed generation
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_ttft_ms: float = 0.0
    last_tok_per_s: float = 0.0
    total_output_tokens: int = 0
    total_generations: int = 0


class NPUEngine:
    def __init__(self, repo: str, ctx: int = 1024, device: str = "NPU"):
        self.repo = repo
        self.ctx = int(ctx)
        self.device = device
        self.pipe = None
        self.stats = GenStats(model=repo, ctx=self.ctx, device=device)
        # NPU processes one generation at a time; serialize all callers.
        self.lock = threading.Lock()
        self._config = None

    # -- loading ---------------------------------------------------------------

    def _pipeline_config(self) -> dict:
        blob = paths.blob_path(self.repo, self.ctx)
        blob.parent.mkdir(parents=True, exist_ok=True)
        if self.device != "NPU":
            return {}
        if blob.is_file():
            return {
                "BLOB_PATH": str(blob),
                "GENERATE_HINT": "FAST_COMPILE",
                "WEIGHTS_PATH": str(paths.model_path(self.repo) / "openvino_model.bin"),
                "MAX_PROMPT_LEN": self.ctx,
            }
        return {
            "EXPORT_BLOB": "YES",
            "BLOB_PATH": str(blob),
            "GENERATE_HINT": "FAST_COMPILE",
            "MAX_PROMPT_LEN": self.ctx,
        }

    def load(self) -> None:
        import openvino_genai

        model_dir = paths.model_path(self.repo)
        if not (model_dir / "openvino_model.xml").is_file():
            raise FileNotFoundError(
                f"No OpenVINO IR for {self.repo}. Run `npu-llm --pull {self.repo}` first."
            )

        cached = paths.blob_path(self.repo, self.ctx).is_file()
        if self.device == "NPU" and not cached:
            info(
                "First run for this model/context: compiling and caching for the NPU. "
                "This can take minutes (larger = longer); later starts are much faster."
            )
        info(f"Loading {self.repo} on {self.device} (ctx={self.ctx}) ...")

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        t0 = time.time()
        self.pipe = openvino_genai.LLMPipeline(str(model_dir), self.device, self._pipeline_config())
        self.stats.load_seconds = round(time.time() - t0, 1)
        good(f"Model ready in {self.stats.load_seconds}s.")

    # -- generation config -----------------------------------------------------

    def default_config(self, max_new_tokens: int | None = None):
        import openvino_genai

        cfg = openvino_genai.GenerationConfig()
        cfg.do_sample = True
        cfg.temperature = 0.7
        cfg.top_p = 0.9
        cfg.top_k = 40
        cfg.repetition_penalty = 1.15
        if max_new_tokens:
            cfg.max_new_tokens = int(max_new_tokens)
        return cfg

    # -- core generate ---------------------------------------------------------

    def _count_input_tokens(self, prompt: str) -> int:
        try:
            return int(self.pipe.get_tokenizer().encode(prompt).input_ids.shape[-1])
        except Exception:
            return 0

    def generate(self, prompt: str, config=None, on_token=None):
        """Generate a completion for ``prompt``.

        ``on_token(text)`` is called for each streamed subword (for printing or
        pushing to an SSE queue). Returns the full decoded text and updates
        ``self.stats`` (TTFT, tokens, tok/s) measured directly from the stream —
        ``openvino_genai`` returns a plain ``str`` here, so we don't rely on a
        ``perf_metrics`` attribute (it's used only as an optional refinement).
        """
        if self.pipe is None:
            raise RuntimeError("Engine not loaded; call load() first.")
        cfg = config or self.default_config()

        in_tokens = self._count_input_tokens(prompt)
        self.stats.generating = True
        self.stats.live_tokens = 0
        start = time.time()
        first_token_t: list[float | None] = [None]

        def _streamer(subword: str):
            if first_token_t[0] is None:
                first_token_t[0] = time.time()
            self.stats.live_tokens += 1
            elapsed = time.time() - start
            if elapsed > 0:
                self.stats.live_tok_per_s = round(self.stats.live_tokens / elapsed, 1)
            if on_token is not None:
                on_token(subword)
            return False  # continue

        try:
            result = self.pipe.generate(prompt, cfg, _streamer)
        finally:
            self.stats.generating = False

        end = time.time()
        text = str(result)
        out_tokens = self.stats.live_tokens

        first_t = first_token_t[0] or end
        ttft_ms = (first_t - start) * 1000.0
        decode_dur = end - first_t
        if out_tokens > 1 and decode_dur > 0:
            tok_s = (out_tokens - 1) / decode_dur   # decode throughput, excludes TTFT
        elif out_tokens and (end - start) > 0:
            tok_s = out_tokens / (end - start)
        else:
            tok_s = 0.0

        # Optional refinement if a build ever returns perf_metrics.
        pm = getattr(result, "perf_metrics", None)
        if pm is not None:
            try:
                out_tokens = int(pm.get_num_generated_tokens()) or out_tokens
            except Exception:
                pass

        s = self.stats
        s.last_input_tokens = in_tokens
        s.last_output_tokens = out_tokens
        s.last_ttft_ms = round(ttft_ms, 1)
        s.last_tok_per_s = round(tok_s, 1)
        s.total_output_tokens += out_tokens
        s.total_generations += 1
        return text

    # -- interactive chat ------------------------------------------------------

    def chat(self, config=None, stats_after: bool = False, compact_pct: int | None = None) -> None:
        """Interactive REPL. Commands: exit / reset / npuinfo.

        When ``compact_pct`` is set, the running context is tracked and, once it
        reaches that percentage of ``ctx``, the conversation is summarized by the
        model itself and the session restarts from that summary. This keeps a
        long chat alive indefinitely instead of hitting a hard context overflow.
        """
        from .hardware import print_info

        cfg = config or self.default_config()
        threshold = int(self.ctx * compact_pct / 100) if compact_pct else None

        console.print(
            "[good]Chat ready.[/good] Commands: [accent]exit[/accent] quit, "
            "[accent]reset[/accent] clear context, [accent]npuinfo[/accent] NPU stats."
        )
        if threshold:
            console.print(
                f"[muted]auto-compact at ~{compact_pct}% of {self.ctx} ctx (~{threshold} tokens)[/muted]"
            )
        console.print()

        transcript: list[tuple[str, str]] = []  # (role, content) kept for summarization
        running = 0  # approximate tokens held in the current KV context

        self.pipe.start_chat()
        try:
            while True:
                try:
                    raw = input("\nyou › ")
                except EOFError:
                    break
                cmd = raw.strip().lower()
                if cmd == "exit":
                    break
                if cmd == "npuinfo":
                    print_info()
                    continue
                if cmd == "reset":
                    self._restart_session()
                    transcript, running = [], 0
                    good("Context reset.")
                    continue
                if not raw.strip():
                    continue

                in_toks = self._count_input_tokens(raw)

                console.print("[muted]llm ›[/muted] ", end="")
                try:
                    reply = self.generate(
                        raw, cfg, on_token=lambda t: console.print(t, end="", markup=False)
                    )
                except RuntimeError:
                    # Safety net if the compaction estimate was off: hard reset.
                    console.print()
                    console.print("[warn]Context overflowed — resetting.[/warn]")
                    self._restart_session()
                    transcript, running = [], 0
                    continue
                console.print()
                if stats_after:
                    self._print_turn_stats()

                transcript.append(("user", raw))
                transcript.append(("assistant", reply))
                running += in_toks + self.stats.last_output_tokens

                if threshold and running >= threshold:
                    pct = round(100 * running / self.ctx)
                    console.print(f"[warn]⟳ context ~{pct}% full — compacting…[/warn]")
                    summary = self._compact(transcript)
                    transcript, running = [], self._count_input_tokens(summary)
                    good("compacted — continuing.")
        finally:
            self._safe_finish_chat()

    def _restart_session(self, system_message: str = "") -> None:
        self._safe_finish_chat()
        self.pipe.start_chat(system_message) if system_message else self.pipe.start_chat()

    def _compact(self, transcript: list[tuple[str, str]]) -> str:
        """Summarize the conversation and restart the session seeded with that summary.

        The summary is installed as the new session's *system message* so the
        model treats it as background context rather than a message to reply to.
        Returns the summary text.
        """
        self._safe_finish_chat()  # free the KV context for a clean summarization pass
        summary = self._summarize(transcript)
        seed = f"Summary of the earlier conversation so far (use as background context):\n{summary}"
        self.pipe.start_chat(seed)
        return summary

    def _summarize(self, transcript: list[tuple[str, str]]) -> str:
        def build(turns):
            convo = "\n".join(
                f"{'User' if r == 'user' else 'Assistant'}: {c}" for r, c in turns
            )
            return (
                "Summarize the conversation below as concise notes, preserving key facts, the "
                "user's goals, decisions made, and any details needed to continue it seamlessly.\n\n"
                f"Conversation:\n{convo}\n\nSummary:"
            )

        turns = list(transcript)
        prompt = build(turns)
        budget = max(128, self.ctx - 96)
        # Drop oldest turn-pairs until the summarization prompt itself fits the context.
        while len(turns) > 2 and self._count_input_tokens(prompt) > budget:
            turns = turns[2:]
            prompt = build(turns)

        scfg = self.default_config(max_new_tokens=min(256, max(64, self.ctx // 4)))
        scfg.do_sample = False  # deterministic summary
        # Stateless one-shot (we are outside start_chat here); keeps user-facing stats untouched.
        return str(self.pipe.generate(prompt, scfg)).strip()

    def _print_turn_stats(self) -> None:
        s = self.stats
        console.print(
            f"[muted]· {s.last_output_tokens} tok · {s.last_tok_per_s} tok/s · "
            f"TTFT {s.last_ttft_ms:.0f} ms · in {s.last_input_tokens} tok[/muted]"
        )

    def _safe_finish_chat(self) -> None:
        try:
            self.pipe.finish_chat()
        except Exception:
            pass
