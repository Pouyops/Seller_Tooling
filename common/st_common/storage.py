"""Content-addressed blob store on local disk, crash-safe.

Writes go to a temp file, are fsync'ed, then atomically renamed into place, so a power cut
leaves either the old state or the complete new file, never a truncated blob. Because names are
SHA-256 of the content, re-uploads dedupe and a job's result can be checked by existence.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BlobRef:
    sha256: str
    ext: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.sha256}.{self.ext}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: Path, data: bytes, *, fsync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise



class BlobStore:
    def __init__(self, root: Path | str, *, fsync: bool = True):
        self.root = Path(root)
        self.fsync = fsync
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str, ext: str) -> Path:
        ext = ext.lstrip(".").lower()
        return self.root / sha256[:2] / sha256[2:4] / f"{sha256}.{ext}"

    def put_bytes(self, data: bytes, ext: str) -> BlobRef:
        sha = sha256_bytes(data)
        path = self.path_for(sha, ext)
        if not path.exists():
            atomic_write_bytes(path, data, fsync=self.fsync)
        return BlobRef(sha, ext.lstrip(".").lower(), path)

    def put_at(self, sha256: str, ext: str, data: bytes) -> BlobRef:
        """Store derived data (e.g. a result) under a caller-chosen key."""
        path = self.path_for(sha256, ext)
        atomic_write_bytes(path, data, fsync=self.fsync)
        return BlobRef(sha256, ext.lstrip(".").lower(), path)

    def exists(self, sha256: str, ext: str) -> bool:
        return self.path_for(sha256, ext).exists()

    def get_bytes(self, sha256: str, ext: str) -> bytes:
        return self.path_for(sha256, ext).read_bytes()
