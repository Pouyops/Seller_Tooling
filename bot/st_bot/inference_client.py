"""HTTP client for the inference API, built for an unreliable network: retries with backoff and
idempotent submissions (the API dedupes by content hash, so a retried upload never runs twice)."""

from __future__ import annotations

import asyncio
import random

import httpx


class InferenceRejected(Exception):
    def __init__(self, status: int, code: str, message: str = ""):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code = status, code


class InferenceUnavailable(Exception):
    pass


class InferenceClient:
    def __init__(self, base_url: str, *, timeout_s: float = 30.0, retries: int = 6, transport: httpx.AsyncBaseTransport | None = None):
        self.retries = retries
        self.http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _request(self, method: str, url: str, **kw) -> httpx.Response:
        delay = 0.5
        last: Exception | None = None
        for _ in range(self.retries):
            try:
                resp = await self.http.request(method, url, **kw)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = e
            else:
                if resp.status_code in (502, 503, 504):
                    last = InferenceUnavailable(f"{resp.status_code} from inference API")
                elif resp.status_code >= 400 and resp.status_code not in (404, 409):
                    try:
                        body = resp.json()
                    except ValueError:
                        body = {}
                    raise InferenceRejected(resp.status_code, body.get("error", "error"), body.get("message", ""))
                else:
                    return resp
            await asyncio.sleep(delay + random.uniform(0, delay / 2))
            delay = min(delay * 2, 15)
        raise InferenceUnavailable(str(last))

    async def submit(self, image: bytes, *, filename: str = "photo.jpg", background: str = "transparent",
                     user_id: int | None = None, priority: str = "normal", wait_s: float = 0) -> dict:
        files = {"file": (filename, image, "application/octet-stream")}
        data = {"background": background, "priority": priority, "wait": str(wait_s)}
        headers = {"x-user-id": str(user_id)} if user_id is not None else {}
        resp = await self._request("POST", "/v1/matting", files=files, data=data, headers=headers)
        return resp.json()

    async def get(self, job_id: str) -> dict | None:
        resp = await self._request("GET", f"/v1/jobs/{job_id}")
        return None if resp.status_code == 404 else resp.json()

    async def wait(self, job_id: str, timeout_s: float, poll_s: float = 0.5) -> dict | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        job = None
        while True:
            job = await self.get(job_id)
            if job is None or job["status"] in ("done", "failed") or loop.time() >= deadline:
                return job
            await asyncio.sleep(poll_s)
            poll_s = min(poll_s * 1.5, 3.0)

    async def download(self, job_id: str) -> tuple[bytes, str]:
        resp = await self._request("GET", f"/v1/jobs/{job_id}/result")
        if resp.status_code != 200:
            raise InferenceRejected(resp.status_code, "not_ready")
        return resp.content, resp.headers.get("content-type", "image/png")
