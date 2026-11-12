# Sourced by every script in scripts/. Run the scripts from anywhere; they cd to the repository root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-python}
OUT=${OUT:-runs}                 # where new training/evaluation outputs go
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-quantum-doom-$USER}

# CKPT picks the checkpoints that evaluation scripts read: "runs" (your own, the default when
# present) or "pretrained" (the checkpoints behind the numbers in the report).
ckpt() {  # ckpt <relative run dir> -> path to its best.pt
  local root=${CKPT:-}
  if [ -z "$root" ]; then
    if [ -f "$OUT/$1/best.pt" ]; then root=$OUT; else root=pretrained; fi
  fi
  [ -f "$root/$1/best.pt" ] || { echo "missing $root/$1/best.pt" >&2; exit 1; }
  echo "$root/$1/best.pt"
}

run() {  # run <output dir> <python args...>: runs one job and keeps its log next to its outputs
  local dir=$1; shift
  mkdir -p "$dir"
  echo ">>> $PYTHON $*" | tee "$dir/train.log"
  "$PYTHON" "$@" 2>&1 | tee -a "$dir/train.log"
}
