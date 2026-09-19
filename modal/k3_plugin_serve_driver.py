#!/usr/bin/env python3
"""Collective HTTP serving driver for Kimi K3 on 2x H200:8 (PP=2 x TP=8),
plugin lane: steering comes from the installed weightless-steer plugin
(registry shadow via the sitecustomize shim), NOT the file-patching hotfix.

Runs under torchrun on ALL 16 processes (2 nodes x 8), same
external_launcher shape as the validated capture/eval lane
(refusal-research/experiments/20260903-kimi-k3-glp, boot #16 + 4 eval arms).

Why not `vllm serve`: this fork's external_launcher serving shape starts a
full frontend+engine on every process, all binding :8000 via SO_REUSEPORT;
a client request lands on a random one and only global rank 0's engine can
schedule SPMD work, so 7/8 of requests hang forever (boot #10 evidence).
Offline LLM() called collectively on every rank is the native mode.

So this driver is a lockstep shim: every rank builds the same LLM(), then
loops. Rank 0 runs the HTTP server (OpenAI-compatible /v1/chat/completions,
non-streaming) and fans each request out to the 15 followers over ZMQ
REQ/REP (poll/answer: followers send READY, rank 0 answers with the work
or "wait"). All ranks then call llm.generate() with identical token ids and
sampling params; only rank 0's output is returned to the client.

One request in flight at a time (collective lockstep); concurrent HTTP
requests queue in arrival order. Requests carry rendered token ids, not
text, so every rank generates on byte-identical input.

Env:
  RANK / WORLD_SIZE / MASTER_ADDR   torchrun (MASTER_ADDR is i6pn IPv6)
  MAX_MODEL_LEN                     default 65536
  SERVED_MODEL_NAME                 default kimi-k3
  LOCKSTEP_PORT                     default 29557
"""

import json
import os
import queue
import threading
import time
import uuid

import torch  # noqa: F401  (engine import order parity with the eval lane)
import zmq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL_ID = "moonshotai/Kimi-K3"
SERVED_MODEL = os.environ.get("SERVED_MODEL_NAME", "kimi-k3")
LOCKSTEP_PORT = int(os.environ.get("LOCKSTEP_PORT", "29557"))

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
MASTER = os.environ.get("MASTER_ADDR", "::1")


def log0(msg):
    if RANK == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------- bridge

class Bridge:
    """Rank-0 only: HTTP handlers enqueue, the engine loop dequeues."""

    def __init__(self):
        self.incoming = queue.Queue()
        self.pending = {}  # id -> dict(event, result, error)
        self.current = None  # request being offered to followers
        self.lock = threading.Lock()

    def submit(self, ids, sp):
        rid = uuid.uuid4().hex
        entry = {"event": threading.Event(), "result": None, "error": None}
        with self.lock:
            self.pending[rid] = entry
        self.incoming.put({"id": rid, "ids": ids, "sp": sp})
        return rid, entry

    def offer(self):
        """The request the engine loop is currently dispatching, if any."""
        with self.lock:
            if self.current is None and not self.incoming.empty():
                try:
                    self.current = self.incoming.get_nowait()
                except queue.Empty:
                    self.current = None
            return self.current

    def finish(self, result=None, error=None):
        with self.lock:
            req = self.current
            self.current = None
        if req is None:
            return
        entry = self.pending.pop(req["id"], None)
        if entry is not None:
            entry["result"] = result
            entry["error"] = error
            entry["event"].set()


# ---------------------------------------------------------------- engine

def build_llm():
    # Steering is armed by the weightless-steer plugin: the sitecustomize
    # shim (PYTHONPATH=/opt/weightless-shim) already registered the
    # KimiLinearForCausalLM shadow in EVERY interpreter and pre-parsed the
    # vector fail-closed, so reaching LLM() means the shim markers printed.
    t0 = time.time()
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        distributed_executor_backend="external_launcher",
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "65536")),
        max_num_seqs=32,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.92,
        enforce_eager=False,
        # copy_inputs=True: the V1 batch queue double-buffers input tensors,
        # so with copy_inputs=False (default) batched traffic trips
        # "Input tensor addresses changed between capture and replay".
        compilation_config={"cudagraph_mode": "PIECEWISE",
                            "cudagraph_copy_inputs": True},
        distributed_timeout_seconds=1800,
        cpu_distributed_timeout_seconds=1800,
        disable_log_stats=True,
    )
    log0(f"serve: engine up in {time.time() - t0:.0f}s")
    return llm


def generate_one(llm, ids, sp):
    params = SamplingParams(
        temperature=sp.get("temperature", 0.6),
        top_p=sp.get("top_p", 0.95),
        max_tokens=min(int(sp.get("max_tokens", 4096)), 16384),
        seed=sp.get("seed"),
    )
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], params)
    o = outs[0].outputs[0]
    return {
        "text": o.text,
        "finish_reason": o.finish_reason,
        "completion_tokens": len(o.token_ids),
        "prompt_tokens": len(outs[0].prompt_token_ids),
    }


# ---------------------------------------------------------------- http (rank 0)

def start_http(llm, bridge):
    app = FastAPI()
    tok = llm.get_tokenizer()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": SERVED_MODEL}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": SERVED_MODEL, "object": "model", "owned_by": "weightless"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if body.get("stream"):
            return JSONResponse(status_code=400, content={"error": {
                "message": "streaming is not supported on this lane"}})
        messages = body.get("messages") or []
        kwargs = body.get("chat_template_kwargs") or {}
        try:
            ids = tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, **kwargs)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"error": {
                "message": f"chat template failed: {exc}"}})
        sp = {k: body[k] for k in ("temperature", "top_p", "max_tokens",
                                   "seed") if k in body}
        rid, entry = bridge.submit(ids, sp)
        if not entry["event"].wait(timeout=1800):
            with bridge.lock:
                bridge.pending.pop(rid, None)
            return JSONResponse(status_code=504, content={"error": {
                "message": "generation timed out (30 min)"}})
        if entry["error"] is not None:
            return JSONResponse(status_code=500, content={"error": {
                "message": str(entry["error"])}})
        r = entry["result"]
        return {
            "id": f"chatcmpl-{rid[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVED_MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": r["text"]},
                "finish_reason": r["finish_reason"],
            }],
            "usage": {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
            },
        }

    import uvicorn
    cfg = uvicorn.Config(app, host="0.0.0.0", port=8000,
                         log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()
    log0("serve: HTTP listening on :8000")


# ---------------------------------------------------------------- loops

def start_tcp_proxy(listen_port, target_host, target_port):
    """Follower ranks listen on :8000 too, piping to the leader.

    Modal Servers health-check (and may route to) EVERY container in the
    cluster; a clustered container whose port never opens is terminated.
    The leader is the only rank with the HTTP bridge, so followers forward
    raw TCP to it (protocol-agnostic, no HTTP parsing)."""
    import socket

    def pipe(src, dst):
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass

    def handle(client):
        try:
            upstream = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            upstream.connect((target_host, target_port))
        except OSError:
            client.close()
            return
        threading.Thread(target=pipe, args=(client, upstream),
                         daemon=True).start()
        threading.Thread(target=pipe, args=(upstream, client),
                         daemon=True).start()

    srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    srv.bind(("::", listen_port))
    srv.listen(64)
    def accept_loop():
        while True:
            client, _ = srv.accept()
            threading.Thread(target=handle, args=(client,),
                             daemon=True).start()
    threading.Thread(target=accept_loop, daemon=True).start()


def run_leader(llm):
    bridge = Bridge()
    start_http(llm, bridge)
    ctx = zmq.Context()
    rep = ctx.socket(zmq.REP)
    rep.ipv6 = True
    rep.bind(f"tcp://[::]:{LOCKSTEP_PORT}")
    log0(f"serve: lockstep REP on port {LOCKSTEP_PORT}, "
         f"followers={WORLD - 1}")
    while True:
        msg = rep.recv_json()  # {"ready": rank}
        req = bridge.offer()
        if req is None:
            rep.send_json({"op": "wait"})
            continue
        work = {"op": "gen", "ids": req["ids"], "sp": req["sp"]}
        rep.send_json(work)
        # Dispatch the same work to the remaining followers as their READYs
        # arrive (all are parked in the poll loop, so this is fast).
        for _ in range(WORLD - 2):
            rep.recv_json()
            rep.send_json(work)
        try:
            bridge.finish(result=generate_one(llm, req["ids"], req["sp"]))
        except Exception as exc:
            bridge.finish(error=exc)
            raise


def run_follower(llm):
    ctx = zmq.Context()
    req = ctx.socket(zmq.REQ)
    req.ipv6 = True
    req.connect(f"tcp://[{MASTER}]:{LOCKSTEP_PORT}")
    start_tcp_proxy(8000, MASTER, 8000)
    print(f"serve: rank {RANK} proxying :8000 -> [{MASTER}]:8000",
          flush=True)
    while True:
        req.send_json({"ready": RANK})
        work = req.recv_json()
        if work["op"] == "wait":
            time.sleep(0.05)
            continue
        if work["op"] == "gen":
            generate_one(llm, work["ids"], work["sp"])


def main():
    llm = build_llm()
    # Boot evidence, fail-closed: with WEIGHTLESS_STEER_PATH set the registry
    # MUST resolve KimiLinearForCausalLM to the steered adapter in this very
    # process (external_launcher: the torchrun processes ARE the workers, so
    # this is the registry state the model build just resolved through). The
    # sitecustomize shim's stderr markers are the per-interpreter channel;
    # this is the post-build assertion. (ModelRegistry.models maps arch ->
    # _LazyRegisteredModel whose .module_name/.class_name name the target.)
    if os.environ.get("WEIGHTLESS_STEER_PATH", "").strip():
        from vllm.model_executor.models import ModelRegistry
        entry = ModelRegistry.models.get("KimiLinearForCausalLM")
        if isinstance(entry, str):
            target = entry
        else:
            target = f"{getattr(entry, 'module_name', '?')}:" \
                     f"{getattr(entry, 'class_name', '?')}"
            if target == "?:?":
                target = repr(entry)
        print(f"serve: rank {RANK} registry KimiLinearForCausalLM -> "
              f"{target}", flush=True)
        if "weightless_steer" not in target:
            raise SystemExit(
                "WEIGHTLESS_STEER_PATH is set but KimiLinearForCausalLM did "
                "NOT resolve to the steered adapter — refusing to serve "
                "unsteered")
    if RANK == 0:
        run_leader(llm)
    else:
        run_follower(llm)


if __name__ == "__main__":
    main()
