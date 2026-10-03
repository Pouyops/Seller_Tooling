"""API + worker behaviour with an in-memory Redis and a fake CPU model (no GPU, no weights)."""

import io

import httpx
import numpy as np
import pytest
from PIL import Image
from redis.exceptions import ConnectionError as RedisConnectionError

fakeredis = pytest.importorskip("fakeredis")

from st_common.jobs import DONE, QUEUED  # noqa: E402
from st_inference.api import create_app  # noqa: E402
from st_inference.models.base import Timing  # noqa: E402
from st_inference.settings import ServiceSettings  # noqa: E402
from st_inference.worker import Worker  # noqa: E402


class FakeModel:
    def __init__(self, name="fake", device="cpu", fail=None):
        self.name, self.device, self.precision = name, device, "fp32"
        self.last_timing = Timing(1, 2, 3)
        self.calls: list[int] = []
        self.fail = fail  # callable(batch_len) -> Exception | None

    def load(self):
        return self

    def unload(self):
        pass

    def predict(self, images):
        self.calls.append(len(images))
        if self.fail:
            e = self.fail(len(images))
            if e:
                raise e
        return [(im.mean(axis=2) > 100).astype(np.float32) for im in images]


def jpeg(w=64, h=48, color=(230, 200, 180), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.zeros((h, w, 3), np.uint8)
    arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = color
    arr = (arr + rng.integers(0, 20, arr.shape)).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return ServiceSettings(data_dir=tmp_path, matting_model="fake", allowed_models="fake,fake2", batch_wait_ms=0,
                           max_batch=4, log_json=False, max_upload_mb=1, visibility_timeout_ms=50, gpu_retry_max_s=0.01)


@pytest.fixture
async def redis():
    r = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest.fixture
async def client(settings, redis):
    app = create_app(settings, redis_client=redis, configure_logging=False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            c.app = app
            yield c


def make_worker(settings, redis, model=None, probe=lambda: "cpu"):
    model = model or FakeModel()

    def factory(name, dev):
        model.device = dev
        return model

    w = Worker(settings, redis_client=redis, model_factory=factory, device_probe=probe, consumer="w-test")
    return w, model


async def submit(client, data=None, **form):
    return await client.post("/v1/matting", files={"file": ("p.jpg", data or jpeg(), "image/jpeg")}, data=form)


async def test_submit_queue_and_dedupe(client):
    r = await submit(client)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == QUEUED and body["deduplicated"] is False and body["input"]["width"] == 64
    assert r.headers["location"] == f"/v1/jobs/{body['job_id']}"
    r2 = await submit(client)
    assert r2.json()["job_id"] == body["job_id"] and r2.json()["deduplicated"] is True
    assert (await client.get(f"/v1/jobs/{body['job_id']}")).json()["status"] == QUEUED
    assert (await client.get(f"/v1/jobs/{body['job_id']}/result")).status_code == 409


@pytest.mark.parametrize(
    "form, data, status, code",
    [
        ({"model": "rmbg2"}, None, 400, "unknown_model"),
        ({"priority": "urgent"}, None, 400, "bad_priority"),
        ({"background": "#12"}, None, 400, "bad_background"),
        ({}, b"not an image", 415, "not_an_image"),
        ({}, b"x" * (2 * 2**20), 413, "too_large"),
    ],
    ids=["unknown_model", "bad_priority", "bad_background", "not_an_image", "too_large"],
)
async def test_submit_validation(client, form, data, status, code):
    r = await submit(client, data=data, **form)
    assert r.status_code == status and r.json()["error"] == code


async def test_unknown_job_ids(client):
    assert (await client.get("/v1/jobs/../../etc")).status_code == 404
    assert (await client.get("/v1/jobs/" + "0" * 32)).status_code == 404


async def test_worker_processes_batch_and_results_are_served(client, settings, redis):
    ids = [(await submit(client, data=jpeg(seed=i))).json()["job_id"] for i in range(3)]
    worker, model = make_worker(settings, redis)
    assert await worker.step(block_ms=0) == 3
    assert model.calls == [3]  # one GPU batch
    for jid in ids:
        view = (await client.get(f"/v1/jobs/{jid}")).json()
        assert view["status"] == DONE and view["result"]["batch_size"] == 3
        res = await client.get(view["result"]["url"])
        assert res.status_code == 200 and res.headers["content-type"] == "image/png"
        img = Image.open(io.BytesIO(res.content))
        assert img.mode == "RGBA" and img.size == (64, 48)
        alpha = np.asarray(img)[..., 3]
        assert alpha[24, 32] == 255 and alpha[2, 2] == 0
        mask = Image.open(io.BytesIO((await client.get(view["result"]["mask_url"])).content))
        assert mask.mode == "L"


async def test_wait_returns_200_when_already_done(client, settings, redis):
    data = jpeg(seed=9)
    await submit(client, data=data)
    worker, _ = make_worker(settings, redis)
    await worker.step(block_ms=0)
    r = await submit(client, data=data, wait="1")
    assert r.status_code == 200 and r.json()["status"] == DONE


async def test_white_background_returns_jpeg(client, settings, redis):
    jid = (await submit(client, background="white")).json()["job_id"]
    worker, _ = make_worker(settings, redis)
    await worker.step(block_ms=0)
    res = await client.get(f"/v1/jobs/{jid}/result")
    assert res.headers["content-type"] == "image/jpeg"
    px = np.asarray(Image.open(io.BytesIO(res.content)))
    assert px[1, 1].min() > 240  # background composited on white


async def test_gpu_down_jobs_wait_in_queue_and_health_is_degraded(client, settings, redis):
    jid = (await submit(client)).json()["job_id"]
    gpu = {"up": False}
    worker, model = make_worker(settings, redis, probe=lambda: "cuda" if gpu["up"] else None)
    assert await worker.step(block_ms=0) == 0
    assert worker.state == "waiting_for_gpu" and model.calls == []
    h = await client.get("/healthz")
    assert h.status_code == 200 and h.json()["status"] == "degraded" and h.json()["gpu_workers_ready"] == 0
    assert (await client.get(f"/v1/jobs/{jid}")).json()["status"] == QUEUED
    r = await submit(client, data=jpeg(seed=5))
    assert r.status_code == 202  # still accepting work
    gpu["up"] = True
    assert await worker.step(block_ms=0) == 2
    h = await client.get("/healthz")
    assert h.json()["status"] == "ok" and h.json()["workers"][0]["state"] == "ready"


async def test_cuda_failure_releases_jobs_without_spending_attempts(client, settings, redis):
    jid = (await submit(client)).json()["job_id"]
    boom = {"on": True}
    model = FakeModel(fail=lambda n: RuntimeError("CUDA error: an illegal memory access was encountered") if boom["on"] else None)
    worker, _ = make_worker(settings, redis, model=model)
    assert await worker.step(block_ms=0) == 0
    assert worker.state == "gpu_error"
    view = (await client.get(f"/v1/jobs/{jid}")).json()
    assert view["status"] == QUEUED and view["attempts"] == 0
    boom["on"] = False
    assert await worker.step(block_ms=0) == 1


async def test_oom_splits_batch(client, settings, redis):
    for i in range(4):
        await submit(client, data=jpeg(seed=20 + i))
    model = FakeModel(fail=lambda n: RuntimeError("CUDA out of memory. Tried to allocate") if n > 1 else None)
    worker, _ = make_worker(settings, redis, model=model)
    assert await worker.step(block_ms=0) == 4
    assert model.calls == [4, 2, 1, 1, 2, 1, 1]


async def test_non_device_error_is_retried_then_failed(client, settings, redis):
    settings.max_attempts = 2
    jid = (await submit(client)).json()["job_id"]
    model = FakeModel(fail=lambda n: ValueError("bad tensor shape"))
    worker, _ = make_worker(settings, redis, model=model)
    worker.queue.max_attempts = 2
    await worker.step(block_ms=0)
    assert (await client.get(f"/v1/jobs/{jid}")).json()["status"] == QUEUED
    await worker.step(block_ms=0)
    view = (await client.get(f"/v1/jobs/{jid}")).json()
    assert view["status"] == "failed" and "bad tensor shape" in view["error"]


async def test_corrupt_input_fails_permanently(client, settings, redis):
    jid = (await submit(client)).json()["job_id"]
    job = await client.app.state.st.queue.get(jid)
    sha, ext = job["input_key"].rsplit(".", 1)
    client.app.state.st.blobs.path_for(sha, ext).write_bytes(b"truncated")
    worker, model = make_worker(settings, redis)
    await worker.step(block_ms=0)
    view = (await client.get(f"/v1/jobs/{jid}")).json()
    assert view["status"] == "failed" and view["error"].startswith("decode_failed") and model.calls == []


async def test_redis_outage_spools_to_disk_and_replays(client, settings, redis, monkeypatch):
    st = client.app.state.st

    async def down(*a, **k):
        raise RedisConnectionError("connection refused")

    monkeypatch.setattr(st.queue, "submit", down)
    monkeypatch.setattr(st.queue, "get", down)
    r = await submit(client, data=jpeg(seed=77))
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    assert r.json()["status"] == QUEUED and len(st.spool) == 1
    view = await client.get(f"/v1/jobs/{jid}")
    assert view.status_code == 200 and view.json()["spooled"] is True
    monkeypatch.undo()
    assert await st.spool.drain(st.queue) == 1
    assert len(st.spool) == 0
    assert (await st.queue.get(jid))["status"] == QUEUED


async def test_crashed_worker_job_is_recovered_by_another_worker(client, settings, redis):
    import asyncio

    jid = (await submit(client)).json()["job_id"]
    dead = await client.app.state.st.queue.claim("matting", "worker-lost-power", 1)
    assert dead and dead[0].job_id == jid
    await asyncio.sleep(0.1)  # visibility timeout is 50 ms in these settings
    worker, _ = make_worker(settings, redis)
    assert await worker.step(block_ms=0) == 1
    assert (await client.get(f"/v1/jobs/{jid}")).json()["status"] == DONE


async def test_metrics_and_health_endpoints(client):
    await submit(client)
    m = await client.get("/metrics")
    assert m.status_code == 200
    assert "st_api_requests_total" in m.text and "st_api_jobs_submitted_total" in m.text
    h = (await client.get("/healthz")).json()
    assert h["redis"] is True and h["queue"]["queued"] >= 1


def test_imaging_exif_orientation_and_alpha_flatten():
    from st_inference.imaging import decode_rgb, parse_background, probe_image

    arr = np.zeros((40, 80, 3), np.uint8)
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90° CW on display
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    assert decode_rgb(buf.getvalue()).shape == (80, 40, 3)
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    b2 = io.BytesIO()
    rgba.save(b2, "PNG")
    assert decode_rgb(b2.getvalue())[0, 0].tolist() == [255, 255, 255]
    assert probe_image(b2.getvalue(), 1000)["ext"] == "png"
    assert parse_background("#FFcc00") == (255, 204, 0) and parse_background("transparent") is None
    big = io.BytesIO()
    Image.new("RGB", (200, 200)).save(big, "PNG")
    from st_inference.imaging import ImageRejected

    with pytest.raises(ImageRejected):
        probe_image(big.getvalue(), 1000)
    assert decode_rgb(big.getvalue(), max_side=50).shape == (50, 50, 3)
