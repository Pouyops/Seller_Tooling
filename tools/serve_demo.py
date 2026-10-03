#!/usr/bin/env python
"""Everything a demo needs in one process: API + GPU worker + web page, no Docker required.

    python tools/serve_demo.py            # then open http://127.0.0.1:8000/

`make serve` is the real layout (separate API and worker processes, Valkey as the queue). This is
the convenience version: if no Redis/Valkey is reachable it runs the queue in-process so the page
still works, and it says which it used. Same application code either way.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from st_inference.api import create_app  # noqa: E402
from st_inference.settings import ServiceSettings  # noqa: E402
from st_inference.worker import Worker  # noqa: E402


async def make_redis(url: str):
    import redis.asyncio as aioredis

    try:
        client = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2)
        await client.ping()
        return client, f"Valkey/Redis at {url}"
    except Exception:
        import fakeredis

        return fakeredis.FakeAsyncRedis(decode_responses=True), "in-process (no Redis reachable — demo only)"


async def run(args) -> None:
    import uvicorn

    settings = ServiceSettings(matting_model=args.model, max_batch=args.max_batch, log_json=False,
                               log_level=args.log_level)
    redis, queue_kind = await make_redis(settings.redis_url)
    app = create_app(settings, redis_client=redis, configure_logging=False)
    worker = Worker(settings, redis_client=redis, consumer="demo")

    async def worker_loop():
        while not worker.stopping.is_set():
            try:
                await worker.step(block_ms=250)
            except Exception as e:  # keep the demo alive
                print(f"[demo] worker: {type(e).__name__}: {str(e)[:160]}", flush=True)
                await asyncio.sleep(1)

    # The heartbeat key expires after a few seconds, and step() only refreshes it on state changes,
    # so /healthz would report "no worker" on a healthy idle system without this.
    tasks = [asyncio.create_task(worker_loop()), asyncio.create_task(worker._heartbeat_loop())]
    task = tasks[0]
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level.lower(), access_log=False)
    server = uvicorn.Server(config)
    print(f"[demo] model={args.model}  queue={queue_kind}")
    print(f"[demo] open  http://{'127.0.0.1' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}/")
    print("[demo] the first photo also loads the model, so it takes ~20 s; later ones are quick.")
    try:
        await server.serve()
    finally:
        worker.stopping.set()
        for t in tasks:
            t.cancel()
        worker._unload_all()
        await redis.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="birefnet_lite")
    ap.add_argument("--max-batch", type=int, default=2)
    ap.add_argument("--log-level", default="warning")
    try:
        asyncio.run(run(ap.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
