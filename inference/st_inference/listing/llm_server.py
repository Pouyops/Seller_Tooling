"""Start the local llama.cpp server for listing generation:  python -m st_inference.listing.llm_server

Model files live in ``$ST_DATA_DIR/models/llm``. Only licence-cleared weights (docs/licenses.md §4)
belong there; the launcher refuses anything on the DO-NOT-SHIP list.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from st_common.licensing import check_model_allowed

DEFAULT_GGUF = "Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_MMPROJ = "mmproj-F16.gguf"


def models_dir() -> Path:
    return Path(os.environ.get("ST_DATA_DIR", Path.cwd() / ".data")) / "models" / "llm"


def find_server() -> str | None:
    env = os.environ.get("ST_LLAMA_SERVER")
    if env and Path(env).exists():
        return env
    found = shutil.which("llama-server")
    if found:
        return found
    for candidate in (Path(os.environ.get("ST_DATA_DIR", ".")) / "bin" / "llama" / "llama-server.exe",
                      Path(os.environ.get("ST_DATA_DIR", ".")) / "bin" / "llama" / "llama-server"):
        if candidate.exists():
            return str(candidate)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("ST_LLM_GGUF", DEFAULT_GGUF))
    ap.add_argument("--mmproj", default=os.environ.get("ST_LLM_MMPROJ", DEFAULT_MMPROJ))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ngl", type=int, default=99, help="layers on GPU; lower it on a small card")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()

    server = find_server()
    if not server:
        raise SystemExit("llama-server not found. Set ST_LLAMA_SERVER, or download a release from "
                         "https://github.com/ggml-org/llama.cpp/releases into $ST_DATA_DIR/bin/llama")
    model = models_dir() / args.model
    if not model.exists():
        raise SystemExit(f"{model} not found. See docs/licenses.md §4 for cleared models, then download the GGUF.")
    check_model_allowed(args.model)

    cmd = [server, "-m", str(model), "--host", "127.0.0.1", "--port", str(args.port),
           "-c", str(args.ctx), "-ngl", str(args.ngl), "--jinja"]
    mmproj = models_dir() / args.mmproj
    if mmproj.exists():
        cmd += ["--mmproj", str(mmproj)]  # vision projector: lets the model actually see the product
    cmd += [a for a in args.extra if a != "--"]
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
