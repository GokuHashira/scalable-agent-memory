# Source this before any work: source env.sh
export ROOT=/fs/gamma-projects/audio/kajayaku/agents
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
