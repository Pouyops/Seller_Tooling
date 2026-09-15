# All real logic lives in tools/tasks.py so that `make <target>` (Linux/WSL/CI) and
# `.\make.cmd <target>` (Windows, no GNU make) run exactly the same code paths.
PY ?= python

.PHONY: setup test bench bench-quick serve worker bot loadtest export llm lint up down help

help:
	@$(PY) tools/tasks.py help

setup:
	$(PY) tools/tasks.py setup

test:
	$(PY) tools/tasks.py test

bench:
	$(PY) tools/tasks.py bench

bench-quick:
	$(PY) tools/tasks.py bench-quick

serve:
	$(PY) tools/tasks.py serve

worker:
	$(PY) tools/tasks.py worker

bot:
	$(PY) tools/tasks.py bot

loadtest:
	$(PY) tools/tasks.py loadtest

export:
	$(PY) tools/tasks.py export

llm:
	$(PY) tools/tasks.py llm

lint:
	$(PY) tools/tasks.py lint

up:
	$(PY) tools/tasks.py up

down:
	$(PY) tools/tasks.py down
