"""OpenAI-compatible HTTP server (completions and chat completions, streaming via SSE)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from kvserve.async_engine import AsyncLLMEngine
from kvserve.config import EngineConfig
from kvserve.engine import RequestOutput
from kvserve.sequence import SamplingParams


class _SamplingFields(BaseModel):
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = -1
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    # Non-OpenAI extensions, matching vLLM's names so benchmark clients work unchanged.
    ignore_eos: bool = False
    stop_token_ids: list[int] = []

    def sampling_params(self, default_max_tokens: int) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens or default_max_tokens,
            stop_token_ids=tuple(self.stop_token_ids),
            ignore_eos=self.ignore_eos,
        )

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options.get("include_usage"))


class CompletionRequest(_SamplingFields):
    prompt: str | list[int]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage]
    max_completion_tokens: int | None = Field(default=None, ge=1)


class IncrementalDetokenizer:
    """Turns a growing token list into text deltas without splitting multi-byte characters."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.text = ""

    def delta(self, token_ids: list[int], final: bool) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if text.endswith("�") and not final:  # incomplete UTF-8 sequence; wait for more
            return ""
        out, self.text = text[len(self.text) :], text
        return out


def startup_banner(engine: AsyncLLMEngine, url: str) -> str:
    runner = engine.engine.runner
    kv_tokens = runner.num_kv_blocks * engine.config.block_size
    graphs = f"decode batch <= {runner.graphs.max_batch}" if runner.graphs else "off"
    return (
        f"kvserve ready at {url}\n"
        f"  model     {engine.config.model}\n"
        f"  device    {engine.config.device} ({str(runner.dtype).removeprefix('torch.')}), "
        f"attention: {runner.attn_backend.__name__}, cuda graphs: {graphs}\n"
        f"  kv cache  {runner.num_kv_blocks} blocks x {engine.config.block_size} = {kv_tokens:,} tokens\n"
        f"  try       {url}/docs   (interactive API)   {url}/metrics"
    )


def create_app(config: EngineConfig, url: str | None = None) -> FastAPI:
    state: dict[str, AsyncLLMEngine] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        state["engine"] = AsyncLLMEngine(config)
        if url:
            print(startup_banner(state["engine"], url), flush=True)
        yield
        state["engine"].shutdown()

    app = FastAPI(title="kvserve", lifespan=lifespan)
    model_name = config.model

    def engine() -> AsyncLLMEngine:
        return state["engine"]

    @app.get("/health")
    async def health() -> Response:
        ok = engine().healthy
        return JSONResponse({"status": "ok" if ok else "error"}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def prometheus() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "kvserve"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw: Request) -> Response:
        params = req.sampling_params(default_max_tokens=16)
        return await _respond(raw, req, req.prompt, params, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw: Request) -> Response:
        req.max_tokens = req.max_completion_tokens or req.max_tokens
        messages = [m.model_dump() for m in req.messages]
        prompt = engine().tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if not isinstance(prompt, list):  # some tokenizer versions return a BatchEncoding
            prompt = prompt["input_ids"]
        params = req.sampling_params(default_max_tokens=config.max_model_len)
        return await _respond(raw, req, prompt, params, chat=True)

    async def _respond(
        raw: Request, req: _SamplingFields, prompt: str | list[int], params: SamplingParams, chat: bool
    ) -> Response:
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        detok = IncrementalDetokenizer(engine().tokenizer)
        stream = engine().generate(prompt, params)

        try:  # surface validation errors (e.g. prompt too long) as 400 before streaming starts
            first = await anext(stream)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        async def outputs() -> AsyncIterator[RequestOutput]:
            yield first
            async for out in stream:
                yield out

        def chunk(text: str, out: RequestOutput) -> dict:
            reason = out.finish_reason.value if out.finish_reason else None
            if chat:
                choice = {"index": 0, "delta": {"content": text}, "finish_reason": reason}
                return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                        "choices": [choice]}  # fmt: skip
            choice = {"index": 0, "text": text, "finish_reason": reason}
            return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [choice]}  # fmt: skip

        def usage(out: RequestOutput) -> dict:
            n = len(out.output_token_ids)
            return {"prompt_tokens": out.num_prompt_tokens, "completion_tokens": n,
                    "total_tokens": out.num_prompt_tokens + n,
                    "prompt_tokens_details": {"cached_tokens": out.num_cached_tokens}}  # fmt: skip

        if req.stream:

            async def sse() -> AsyncIterator[str]:
                last = None
                async for out in outputs():
                    if await raw.is_disconnected():
                        await stream.aclose()  # aborts the request in the engine
                        return
                    last = out
                    text = detok.delta(out.output_token_ids, out.finished)
                    if chat and len(out.output_token_ids) == 1:
                        role = chunk("", out)
                        role["choices"][0]["delta"] = {"role": "assistant", "content": ""}
                        role["choices"][0]["finish_reason"] = None
                        yield f"data: {json.dumps(role)}\n\n"
                    if text or out.finished:
                        yield f"data: {json.dumps(chunk(text, out))}\n\n"
                if last is not None and req.include_usage:
                    body = chunk("", last) | {"choices": [], "usage": usage(last)}
                    yield f"data: {json.dumps(body)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        last = None
        async for out in outputs():
            last = out
        assert last is not None
        text = engine().tokenizer.decode(last.output_token_ids, skip_special_tokens=True)
        body = chunk(text, last)
        if chat:
            body["object"] = "chat.completion"
            choice = body["choices"][0]
            choice["message"] = {"role": "assistant", "content": choice.pop("delta")["content"]}
        body["usage"] = usage(last)
        return JSONResponse(body)

    return app
