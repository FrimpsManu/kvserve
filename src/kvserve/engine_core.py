"""Engine core process: owns LLMEngine and runs the step loop, nothing else.

The API server (engine_client.ProcessEngineClient) does HTTP, tokenization,
detokenization and SSE in its own process. The two talk over ZeroMQ IPC sockets:

    server -> core   ("add", request_id, token_ids, SamplingParams)
                     ("abort", request_id)
                     ("shutdown",)
    core -> server   ("ready", info)
                     ("step", outputs, report)   one message per engine step, where
                         outputs = [(request_id, new_token_ids, finished, finish_reason,
                                     num_prompt_tokens, num_cached_tokens), ...]
                         report  = scheduler/step stats for telemetry
                     ("gauges", report)         queue sizes changed while idle
                     ("reject", request_id, message)
                     ("error", traceback)

Batching every request's new tokens into one message per step keeps IPC cost
proportional to steps, not to tokens x requests.
"""

from __future__ import annotations

import dataclasses
import traceback
from typing import Any

import zmq

from kvserve.async_engine import engine_info
from kvserve.config import EngineConfig
from kvserve.engine import LLMEngine


def _report(engine: LLMEngine, with_step: bool) -> dict[str, Any]:
    sched = engine.scheduler
    report: dict[str, Any] = {
        "running": len(sched.running),
        "waiting": len(sched.waiting),
        "kv_usage": engine.kv.usage,
        "step_paths": dict(engine.runner.step_paths),
    }
    step = engine.last_step
    if with_step and step is not None:
        report["step"] = dataclasses.astuple(step)
    return report


def run_engine_core(config: EngineConfig, input_addr: str, output_addr: str) -> None:
    """Entry point of the engine process (started with the 'spawn' method)."""
    ctx = zmq.Context()
    inbox = ctx.socket(zmq.PULL)
    inbox.connect(input_addr)
    outbox = ctx.socket(zmq.PUSH)
    outbox.connect(output_addr)
    try:
        engine = LLMEngine(config)
        outbox.send_pyobj(("ready", engine_info(engine)))
        _loop(engine, inbox, outbox)
    except BaseException:
        outbox.send_pyobj(("error", traceback.format_exc()))
        raise
    finally:
        outbox.close(linger=1000)
        inbox.close(linger=0)
        ctx.term()


def _loop(engine: LLMEngine, inbox: zmq.Socket, outbox: zmq.Socket) -> None:
    while True:
        # Block while idle; while busy, drain everything that arrived during the step.
        commands = [] if engine.has_unfinished() else [inbox.recv_pyobj()]
        while True:
            try:
                commands.append(inbox.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                break
        for cmd in commands:
            kind = cmd[0]
            if kind == "shutdown":
                return
            if kind == "add":
                _, request_id, token_ids, params = cmd
                try:
                    engine.add_request(token_ids, params, request_id)
                except ValueError as e:
                    outbox.send_pyobj(("reject", request_id, str(e)))
            elif kind == "abort":
                engine.abort(cmd[1])

        if not engine.has_unfinished():
            if commands:  # e.g. an abort emptied the batch: refresh the dashboard gauges
                outbox.send_pyobj(("gauges", _report(engine, with_step=False)))
            continue

        outputs = engine.step()
        packed = [
            (o.request_id, o.new_token_ids, o.finished, o.finish_reason.value if o.finish_reason else None,
             o.num_prompt_tokens, o.num_cached_tokens)
            for o in outputs
        ]  # fmt: skip
        outbox.send_pyobj(("step", packed, _report(engine, with_step=True)))
