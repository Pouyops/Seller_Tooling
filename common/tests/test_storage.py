from st_common.storage import BlobStore, atomic_write_bytes, sha256_bytes


def test_put_dedupes_and_layout(tmp_path):
    store = BlobStore(tmp_path, fsync=False)
    a = store.put_bytes(b"hello", "PNG")
    b = store.put_bytes(b"hello", ".png")
    assert a == b
    assert a.sha256 == sha256_bytes(b"hello")
    assert a.path == tmp_path / a.sha256[:2] / a.sha256[2:4] / f"{a.sha256}.png"
    assert store.exists(a.sha256, "png")
    assert store.get_bytes(a.sha256, "png") == b"hello"
    assert a.key == f"{a.sha256}.png"


def test_put_at_overwrites_atomically(tmp_path):
    store = BlobStore(tmp_path)
    store.put_at("ab" * 32, "json", b"1")
    ref = store.put_at("ab" * 32, "json", b"2")
    assert ref.path.read_bytes() == b"2"
    leftovers = [p for p in ref.path.parent.iterdir() if p.name.startswith(".tmp-")]
    assert leftovers == []


def test_atomic_write_cleans_up_on_failure(tmp_path, monkeypatch):
    import os

    target = tmp_path / "x" / "out.bin"

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    try:
        atomic_write_bytes(target, b"data")
    except OSError:
        pass
    assert not target.exists()
    assert list(target.parent.iterdir()) == []
