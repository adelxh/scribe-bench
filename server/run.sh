#!/bin/sh
cd "$(dirname "$0")"
exec .venv/bin/uvicorn app.main:app --reload --port "${PORT:-5001}"
