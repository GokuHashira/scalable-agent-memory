#!/bin/bash
# Source on Zaratan before any command (login node or job). Keeps every file the run touches inside SCRATCH_ROOT
# (constitution "Zaratan boundary") and refuses to run from anywhere else.
export SCRATCH_ROOT=/home/gsakthiv/scratch.cmsc848n
_ZENV_SELF="${BASH_SOURCE[0]:-$0}"  # bash: BASH_SOURCE; zsh: $0 is the sourced file
REPO_DIR=$(git -C "$(dirname "${_ZENV_SELF}")" rev-parse --show-toplevel 2>/dev/null) || REPO_DIR=$(pwd -P)
REPO_DIR=$(cd "${REPO_DIR}" && pwd -P)
# SCRATCH_ROOT is a symlink (to /scratch/zt1/project/cmsc848n/user/gsakthiv); compare physical paths.
SCRATCH_REAL=$(cd "${SCRATCH_ROOT}" 2>/dev/null && pwd -P) || SCRATCH_REAL="${SCRATCH_ROOT}"
case "${REPO_DIR}/" in
  "${SCRATCH_REAL}"/* | "${SCRATCH_ROOT}"/*) ;;
  *) echo "zaratan_env.sh: repo ${REPO_DIR} is outside ${SCRATCH_ROOT} (${SCRATCH_REAL}); refusing to run" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export REPO_DIR
_ZENV_CACHE="${SCRATCH_ROOT}/.cache"  # prefixed: this file is sourced into job scripts
export XDG_CACHE_HOME="${_ZENV_CACHE}" XDG_DATA_HOME="${SCRATCH_ROOT}/.local/share" XDG_CONFIG_HOME="${SCRATCH_ROOT}/.config"
export UV_CACHE_DIR="${_ZENV_CACHE}/uv" UV_PYTHON_INSTALL_DIR="${SCRATCH_ROOT}/.local/share/uv/python" UV_TOOL_DIR="${SCRATCH_ROOT}/.local/share/uv/tools"
export UV_PROJECT_ENVIRONMENT="${REPO_DIR}/.venv"
export PIP_CACHE_DIR="${_ZENV_CACHE}/pip" HF_HOME="${_ZENV_CACHE}/huggingface" TIKTOKEN_CACHE_DIR="${_ZENV_CACHE}/tiktoken"
export TORCH_HOME="${_ZENV_CACHE}/torch" MPLCONFIGDIR="${_ZENV_CACHE}/matplotlib" TMPDIR="${SCRATCH_ROOT}/tmp"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TIKTOKEN_CACHE_DIR}" "${TMPDIR}" "${REPO_DIR}/logs"
cd "${REPO_DIR}"
unset _ZENV_CACHE _ZENV_SELF SCRATCH_REAL
