#!/usr/bin/env bash
# Serve an OpenAI-compatible vLLM endpoint on one A6000 (planner.md §5).
# The reader (mem27.reader.vllm_reader) uses vLLM offline for throughput, but this
# endpoint is what All-Mem / Graphiti would target at integration.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
GPU="${2:-0}"
PORT="${3:-8000}"
CUDA_VISIBLE_DEVICES="$GPU" .venv_vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --port "$PORT"
