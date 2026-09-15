import asyncio

import pytest

fakeredis = pytest.importorskip("fakeredis")

from st_common.jobs import DONE, FAILED, QUEUED, RUNNING, JobQueue, JobSpec  # noqa: E402


@pytest.fixture
async def r():
    client = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


def spec(n=0, **kw):
    return JobSpec(kind="matting", model="birefnet_lite", input_key=f"{n:064x}.png", **kw)


async def test_submit_is_idempotent(r):
    q = JobQueue(r)
    jid, status, created = await q.submit(spec(1, owner="u1"))
    assert (status, created) == (QUEUED, True)
    jid2, status2, created2 = await q.submit(spec(1, owner="u2"))  # owner not part of identity
    assert (jid2, status2, created2) == (jid, QUEUED, False)
    assert (await q.depth("matting"))["queued"] == 1


async def test_job_id_depends_on_params():
    assert spec(1).job_id() != spec(1, params={"size": 512}).job_id()
    assert spec(1, params={"a": 1, "b": 2}).job_id() == spec(1, params={"b": 2, "a": 1}).job_id()


async def test_claim_complete_roundtrip(r):
    q = JobQueue(r)
    jid, *_ = await q.submit(spec(1, params={"bg": "white"}))
    jobs = await q.claim("matting", "w1", count=8)
    assert [j.job_id for j in jobs] == [jid]
    job = jobs[0]
    assert job.attempts == 1 and job.params == {"bg": "white"} and job.model == "birefnet_lite"
    assert (await q.get(jid))["status"] == RUNNING
    assert (await q.depth("matting")) == {"queued": 0, "running": 1, "dead": 0}

    await q.complete(job, {"output_key": "abc.png", "latency_ms": 12.5})
    got = await q.get(jid)
    assert got["status"] == DONE and got["result"]["output_key"] == "abc.png"
    assert (await q.depth("matting")) == {"queued": 0, "running": 0, "dead": 0}
    # resubmitting finished work returns the cached job, no new message
    _, status, created = await q.submit(spec(1, params={"bg": "white"}))
    assert (status, created) == (DONE, False)
    assert await q.claim("matting", "w1", count=8) == []


async def test_batch_claim_respects_count_and_priority(r):
    q = JobQueue(r)
    normal = [(await q.submit(spec(i)))[0] for i in range(5)]
    high = (await q.submit(spec(99, priority="high")))[0]
    first = await q.claim("matting", "w1", count=3)
    assert first[0].job_id == high
    assert [j.job_id for j in first[1:]] == normal[:2]
    rest = await q.claim("matting", "w1", count=10)
    assert [j.job_id for j in rest] == normal[2:]


async def test_crashed_worker_jobs_are_reclaimed(r):
    q = JobQueue(r, visibility_timeout_ms=50)
    jid, *_ = await q.submit(spec(7))
    lost = await q.claim("matting", "worker-that-dies", count=1)
    assert lost[0].job_id == jid
    # nothing is acked; simulate power loss and a new worker coming up after the timeout
    assert await q.claim("matting", "w2", count=1) == []
    await asyncio.sleep(0.12)
    recovered = await q.claim("matting", "w2", count=1)
    assert [j.job_id for j in recovered] == [jid]
    assert recovered[0].attempts == 2
    await q.complete(recovered[0], {"ok": True})
    assert (await q.get(jid))["status"] == DONE


async def test_retry_then_dead_letter(r):
    q = JobQueue(r, max_attempts=2)
    jid, *_ = await q.submit(spec(3))
    j = (await q.claim("matting", "w", count=1))[0]
    await q.fail(j, "cuda error", retry=True)
    assert (await q.get(jid))["status"] == QUEUED
    j = (await q.claim("matting", "w", count=1))[0]
    assert j.attempts == 2
    await q.fail(j, "cuda error again", retry=True)  # attempts exhausted -> dead
    got = await q.get(jid)
    assert got["status"] == FAILED and "again" in got["error"]
    assert (await q.depth("matting")) == {"queued": 0, "running": 0, "dead": 1}
    # a failed job may be resubmitted explicitly
    _, status, created = await q.submit(spec(3))
    assert (status, created) == (QUEUED, True)


async def test_release_does_not_spend_attempts(r):
    q = JobQueue(r, max_attempts=1)
    jid, *_ = await q.submit(spec(4))
    for _ in range(3):  # GPU disappears repeatedly; job must not be dead-lettered
        jobs = await q.claim("matting", "w", count=1)
        assert jobs and jobs[0].attempts == 1
        await q.release(jobs)
    assert (await q.get(jid))["status"] == QUEUED


async def test_complete_before_ack_crash_is_not_rerun(r):
    q = JobQueue(r, visibility_timeout_ms=30)
    jid, *_ = await q.submit(spec(5))
    job = (await q.claim("matting", "w1", count=1))[0]
    # result written + status updated, but the process dies before XACK
    await r.hset(q.job_key(jid), "status", DONE)
    await asyncio.sleep(0.08)
    assert await q.claim("matting", "w2", count=1) == []
    assert (await q.depth("matting"))["running"] == 0


async def test_wait_returns_on_terminal(r):
    q = JobQueue(r)
    jid, *_ = await q.submit(spec(6))

    async def worker():
        await asyncio.sleep(0.05)
        j = (await q.claim("matting", "w", count=1))[0]
        await q.complete(j, {"ok": 1})

    t = asyncio.create_task(worker())
    got = await q.wait(jid, timeout_s=2, poll_s=0.01)
    await t
    assert got["status"] == DONE
    assert (await q.wait("nope", timeout_s=0.01)) is None


async def test_invalid_priority(r):
    with pytest.raises(ValueError):
        await JobQueue(r).submit(spec(1, priority="urgent"))
