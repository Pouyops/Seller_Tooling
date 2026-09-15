# Architecture Decision Records

Short ADRs for every non-obvious technical choice. Format: context → decision →
consequences → alternative considered. Newest at the bottom.

---

## ADR-001 — Monorepo with a shared `common/` package
**Context.** The brief fixes top-level dirs `/inference /bot /web /eval /docs /infra`.
Persian text normalization, config, logging, and the queue/job schema are needed by
the API, the GPU worker, the bot and the eval harness.
**Decision.** Add `/common` (`st_common`), a small dependency-light package. Each of
`common`, `inference`, `eval`, `bot` is its own pip-installable package; one dev venv
installs all of them editable.
**Consequences.** The bot does not import torch. Production images install only what they need.
**Alternative.** Copy the normalizer into each package (drift risk), or put it in `inference` (drags torch into the bot).

## ADR-002 — `tools/tasks.py` is the single task runner; `Makefile` and `make.cmd` are thin shims
**Context.** Build box is Windows without GNU make; CI/production is Linux.
**Decision.** Targets are implemented once in Python. `make bench` (Linux/WSL) and
`.\make.cmd bench` (Windows) call the same function.
**Alternative.** `just`/`invoke`/`nox` — extra tool to install for every contributor.

## ADR-003 — Large artifacts outside the repo under `ST_DATA_DIR`
**Context.** Weights (~1 GB per matting model, several GB per LLM), the benchmark set, TensorRT
engines and job blobs must not enter git. The build machine's system drive has <7 GB free.
**Decision.** Everything large lives under `ST_DATA_DIR` (default `D:/seller-tooling` here,
`/data` in containers). Benchmark images are *regenerated deterministically* from a seed, so the
set is reproducible without shipping files.

## ADR-004 — Valkey instead of Redis server
**Context.** Redis ≥ 7.4 is no longer BSD (RSALv2/SSPLv1, and AGPLv3 from Redis 8). Internal
unmodified use is very likely fine, but it is one more licence to explain to counsel.
**Decision.** Run **Valkey 8** (BSD-3, Linux Foundation fork, wire-compatible) in `infra/`.
Client is `redis-py` (MIT). Code says "Redis" because the protocol is Redis.
**Alternative.** Redis 7.2 (last BSD release) — works, but frozen.
