"""Disk spool for submissions accepted while Redis/Valkey is unreachable.

The API writes the job spec to disk (fsync'ed) and answers 202, same as a normal submission. A
housekeeping task replays the spool into the queue when the connection returns. Replay is
idempotent because job ids are content hashes: if we crash after enqueueing but before deleting
the spool file, the second replay just finds the existing job.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from st_common.jobs import JobSpec
from st_common.storage import atomic_write_bytes


class Spool:
    def __init__(self, data_dir: Path | str):
        self.dir = Path(data_dir) / "spool"
        self.dir.mkdir(parents=True, exist_ok=True)

    def put(self, spec: JobSpec) -> str:
        job_id = spec.job_id()
        if not self.has(job_id):
            atomic_write_bytes(self.dir / f"{time.time_ns()}_{job_id}.json", json.dumps(asdict(spec), ensure_ascii=False).encode())
        return job_id

    def has(self, job_id: str) -> bool:
        return any(self.dir.glob(f"*_{job_id}.json"))

    def items(self) -> list[tuple[Path, JobSpec]]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append((p, JobSpec(**json.loads(p.read_text(encoding="utf-8")))))
            except (json.JSONDecodeError, TypeError):
                p.rename(p.with_suffix(".corrupt"))
        return out

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.glob("*.json"))

    def writable(self) -> bool:
        probe = self.dir / ".probe"
        try:
            probe.write_bytes(b"1")
            probe.unlink()
            return True
        except OSError:
            return False

    async def drain(self, queue) -> int:
        n = 0
        for path, spec in self.items():
            await queue.submit(spec)
            path.unlink(missing_ok=True)
            n += 1
        return n
