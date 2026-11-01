# Source this before any work: source env.sh
# ROOT defaults to this repo; set ROOT before sourcing to use another checkout or cache location.
export ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
export HF_HOME="$ROOT/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$ROOT/.cache/hf/hub"
export UV_CACHE_DIR="$ROOT/.uv_cache"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
# BLAS threads = 1 per planner rules (we parallelize across processes/GPUs)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VIRTUAL_ENV="$ROOT/.venv"
export PATH="$ROOT/.venv/bin:$PATH"
# The mem27 package lives in src/; the repo root holds allmem/ and graphiti/.
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
