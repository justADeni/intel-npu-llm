"""OpenAI-compatible HTTP API backed by the NPU engine.

Endpoints:
* ``GET  /v1/models``            — list the loaded model
* ``POST /v1/chat/completions``  — chat, streaming (SSE) or not
* ``POST /v1/completions``       — legacy text completion
* ``GET  /health``               — liveness

The NPU runs a single generation at a time, so every request holds
``engine.lock`` for the duration of its generation; concurrent requests queue.
"""

from __future__ import annotations

import json
import queue
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import __version__
from .console import good, info


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = []
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | str | None = None


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str] = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False


def _now() -> int:
    return int(time.time())


def _apply_template(engine, messages: list[dict]) -> str:
    """Render chat messages to a prompt using the model's own chat template."""
    try:
        tok = engine.pipe.get_tokenizer()
        return tok.apply_chat_template(messages, True, "")
    except Exception:
        # Fallback: plain role-tagged concatenation.
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        parts.append("assistant:")
        return "\n".join(parts)


def _build_config(engine, temperature, top_p, max_tokens):
    default_max = max(16, min(engine.ctx - 8, 512))
    cfg = engine.default_config(max_new_tokens=max_tokens or default_max)
    if temperature is not None:
        cfg.temperature = float(temperature)
        cfg.do_sample = float(temperature) > 0.0
    if top_p is not None:
        cfg.top_p = float(top_p)
    return cfg


def create_app(engine) -> FastAPI:
    app = FastAPI(title="npu-llm", version=__version__)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.repo, "device": engine.device}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.repo,
                    "object": "model",
                    "created": _now(),
                    "owned_by": "npu-llm",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        messages = [m.model_dump() for m in req.messages]
        prompt = _apply_template(engine, messages)
        cfg = _build_config(engine, req.temperature, req.top_p, req.max_tokens)
        model_id = req.model or engine.repo
        cid = f"chatcmpl-{_now()}"

        if req.stream:
            return StreamingResponse(
                _stream_chat(engine, prompt, cfg, cid, model_id),
                media_type="text/event-stream",
            )

        with engine.lock:
            text = engine.generate(prompt, cfg)
        s = engine.stats
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": _now(),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": s.last_input_tokens,
                    "completion_tokens": s.last_output_tokens,
                    "total_tokens": s.last_input_tokens + s.last_output_tokens,
                },
            }
        )

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
        cfg = _build_config(engine, req.temperature, req.top_p, req.max_tokens)
        model_id = req.model or engine.repo
        cid = f"cmpl-{_now()}"
        with engine.lock:
            text = engine.generate(prompt, cfg)
        s = engine.stats
        return JSONResponse(
            {
                "id": cid,
                "object": "text_completion",
                "created": _now(),
                "model": model_id,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": s.last_input_tokens,
                    "completion_tokens": s.last_output_tokens,
                    "total_tokens": s.last_input_tokens + s.last_output_tokens,
                },
            }
        )

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stream_chat(engine, prompt, cfg, cid, model_id):
    """Sync generator yielding OpenAI chat.completion.chunk SSE frames.

    The engine lock is held by the *worker thread* for the duration of the
    generation — not across this generator's yields. That way, if the client
    disconnects and this generator is abandoned, the lock is still released when
    generation finishes, instead of leaking and blocking every future request.
    """
    q: queue.Queue = queue.Queue()
    DONE = object()

    def worker():
        with engine.lock:
            try:
                engine.generate(prompt, cfg, on_token=q.put)
            except Exception as e:  # surface as a final error token
                q.put(f"\n[error: {e}]")
            finally:
                q.put(DONE)

    threading.Thread(target=worker, daemon=True).start()

    base = {"id": cid, "object": "chat.completion.chunk", "created": _now(), "model": model_id}
    # role frame
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    while True:
        item = q.get()
        if item is DONE:
            break
        yield _sse({**base, "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}]})
    yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    yield "data: [DONE]\n\n"


def serve(engine, host: str = "127.0.0.1", port: int = 8000, *, log_level: str = "warning"):
    import uvicorn

    app = create_app(engine)
    good(f"OpenAI-compatible API on http://{host}:{port}  (model: {engine.repo})")
    info(f"  try: curl http://{host}:{port}/v1/models")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
