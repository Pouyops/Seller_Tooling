#!/usr/bin/env python
"""Task runner behind `make <target>` (Linux/WSL/CI) and `make.cmd <target>` (Windows). See ADR-002.

Targets
  setup        install packages (editable), download licence-cleared weights, build the benchmark set
  test         unit tests (no GPU / weights / network needed)
  test-all     unit + GPU/weights integration tests
  bench        full model comparison -> eval/results.md (resumable)
  bench-quick  2 models x 24 images, for a smoke check
  bench-kaggle run the benchmark on a Kaggle T4 and pull results back
               (`bench-kaggle push|status|fetch`; needs ~/.kaggle/kaggle.json)
  serve        start Valkey (docker), the inference API and a GPU worker
  worker       run only a GPU worker
  api          run only the API
  bot          run the Telegram bot (needs ST_TELEGRAM_TOKEN)
  loadtest     load-test a running API; writes docs/loadtest.md
  export       ONNX export (+ TensorRT engine if available) for the default model
  llm          start the local llama.cpp server for the listing generator
  up / down    docker compose (Valkey, Prometheus)
  weights      download weights only
  dataset      build the synthetic benchmark set only
  report       re-render eval/results.md from the latest run
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
COMPOSE = ["docker", "compose", "-f", str(ROOT / "infra" / "docker-compose.yml")]


def load_env() -> None:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split(" #", 1)[0].strip()
            os.environ.setdefault(k.strip(), v)
    if "ST_DATA_DIR" not in os.environ:
        default = Path("D:/seller-tooling") if os.name == "nt" and Path("D:/seller-tooling").exists() else ROOT / ".data"
        os.environ["ST_DATA_DIR"] = str(default)
    os.environ.setdefault("HF_HOME", str(Path(os.environ["ST_DATA_DIR"]) / "hf"))
    os.environ.setdefault("PYTHONUTF8", "1")


def sh(cmd: list[str], check: bool = True, **kw) -> int:
    print("+", " ".join(map(str, cmd)), flush=True)
    rc = subprocess.call(cmd, cwd=kw.pop("cwd", ROOT), **kw)
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


def data_dir() -> Path:
    return Path(os.environ["ST_DATA_DIR"])


def dataset_dir() -> Path:
    return data_dir() / "data" / "bench" / "synth-v1-s1403-n200"


# ---------------------------------------------------------------------------------------------
def t_weights(names: list[str] | None = None) -> None:
    import hashlib

    from st_inference.models.registry import REGISTRY, default_models_dir

    root = default_models_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name in names or list(REGISTRY):
        spec = REGISTRY[name]
        target = root / spec.weights
        if target.exists():
            print(f"[weights] {name}: present")
            continue
        d = spec.download
        print(f"[weights] {name}: downloading from {d.get('repo') or d.get('url')}")
        if d["type"] == "hf":
            from huggingface_hub import snapshot_download

            snapshot_download(d["repo"], allow_patterns=d.get("allow_patterns"), local_dir=str(target))
        else:
            import urllib.request

            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            urllib.request.urlretrieve(d["url"], tmp)
            md5 = hashlib.md5(tmp.read_bytes()).hexdigest()
            if md5 != d["md5"]:
                tmp.unlink()
                raise SystemExit(f"[weights] {name}: md5 mismatch {md5} != {d['md5']}")
            tmp.replace(target)


def t_dataset() -> None:
    if (dataset_dir() / "manifest.jsonl").exists():
        print(f"[dataset] present at {dataset_dir()}")
        return
    sh([PY, "-m", "st_eval.synth", "--out", str(dataset_dir()), "--n", "200", "--seed", "1403"])


def t_setup() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Python >= 3.11 required")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is not installed. Install the CUDA build first, e.g.:\n"
              "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        raise SystemExit(1)
    sh([PY, "-m", "pip", "install", "-e", "common[test]", "-e", "inference[onnx]", "-e", "eval", "-e", "bot"])
    sh([PY, "-m", "pip", "install", "--no-deps", "transparent-background==1.3.4", "easydict", "pyyaml"])
    t_weights()
    t_dataset()
    print("[setup] done. Next: `make test`, `make bench`, `make serve`.")


def t_test(all_: bool = False) -> None:
    args = [PY, "-m", "pytest"]
    if not all_:
        args += ["-m", "not gpu and not weights and not redis and not llm"]
    sh(args)


def _bench_out() -> Path:
    return ROOT / "eval" / "runs" / dataset_dir().name


def t_bench(extra: list[str]) -> None:
    t_dataset()
    t_weights()
    sh([PY, "-m", "st_eval.bench", "--dataset", str(dataset_dir()), "--out", str(_bench_out()), "--report", "eval/results.md", *extra])


def t_bench_quick() -> None:
    t_dataset()
    t_weights(["birefnet_lite", "inspyrenet_fast"])
    out = ROOT / "eval" / "runs" / "quick"
    sh([PY, "-m", "st_eval.bench", "--dataset", str(dataset_dir()), "--models", "birefnet_lite,inspyrenet_fast", "--limit", "24",
        "--batch-sizes", "1,4", "--sweep-seconds", "5", "--out", str(out), "--report", str(out / "results.md")])


def t_report() -> None:
    sh([PY, "-m", "st_eval.report", str(_bench_out()), "--out", "eval/results.md"])


def _docker_ok() -> bool:
    return shutil.which("docker") is not None and subprocess.call(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def t_up(services: list[str] | None = None) -> None:
    if not _docker_ok():
        print("[up] docker not available; start a Redis-protocol server yourself and set ST_REDIS_URL")
        return
    sh([*COMPOSE, "up", "-d", *(services or ["valkey"])])


def t_down() -> None:
    sh([*COMPOSE, "down"])


def t_serve() -> None:
    t_up(["valkey"])
    host, port = os.environ.get("ST_API_HOST", "0.0.0.0"), os.environ.get("ST_API_PORT", "8000")
    procs = [
        subprocess.Popen([PY, "-m", "st_inference.worker"], cwd=ROOT),
        subprocess.Popen([PY, "-m", "uvicorn", "st_inference.api:app", "--host", host, "--port", port], cwd=ROOT),
    ]
    print(f"[serve] API on http://{host}:{port}  (docs: /docs, health: /healthz, metrics: /metrics). Ctrl+C to stop.")
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def main(argv: list[str]) -> None:
    load_env()
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(__doc__)
        return
    target, rest = argv[0], argv[1:]
    actions = {
        "setup": t_setup,
        "test": lambda: t_test(False),
        "test-all": lambda: t_test(True),
        "bench": lambda: t_bench(rest),
        "bench-quick": t_bench_quick,
        "bench-kaggle": lambda: sh([PY, str(ROOT / "tools" / "kaggle_bench.py"), *(rest or ["push"])]),
        "report": t_report,
        "weights": lambda: t_weights(rest or None),
        "dataset": t_dataset,
        "serve": t_serve,
        "worker": lambda: sh([PY, "-m", "st_inference.worker", *rest]),
        "api": lambda: sh([PY, "-m", "uvicorn", "st_inference.api:app", "--host", os.environ.get("ST_API_HOST", "0.0.0.0"),
                           "--port", os.environ.get("ST_API_PORT", "8000")]),
        "bot": lambda: sh([PY, "-m", "st_bot", *rest]),
        "loadtest": lambda: sh([PY, "-m", "st_inference.loadtest", *rest]),
        "export": lambda: sh([PY, "-m", "st_inference.export", *rest]),
        "llm": lambda: sh([PY, "-m", "st_inference.listing.llm_server", *rest]),
        "up": lambda: t_up(rest or None),
        "down": t_down,
        "lint": lambda: sh([PY, "-m", "ruff", "check", "."], check=False),
    }
    if target not in actions:
        raise SystemExit(f"unknown target {target!r}\n{__doc__}")
    actions[target]()


if __name__ == "__main__":
    main(sys.argv[1:])
