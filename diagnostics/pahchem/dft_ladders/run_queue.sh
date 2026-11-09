#!/usr/bin/env bash
# Run the ladder jobs of JOBS (lines "rel z two_s diffuse", from make_ladder.py) NPAR at a time, THREADS threads each, at
# nice 19 (megatron, or inside a DIaL3 node job: make_slurm.py). LADDER_MAX_MEMORY [MB] caps each job, LADDER_MAXSTEPS its optimisation steps. Usage: run_queue.sh ROOT JOBS [NPAR] [THREADS]. Launch detached:
#   setsid nohup bash run_queue.sh ROOT JOBS 6 16 > ROOT/queue.log 2>&1 < /dev/null &
set -u
ROOT=$1; JOBS=$2; NPAR=${3:-6}; THREADS=${4:-16}
PY=${PY:-$HOME/opt/qc-venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
grep -v '^\s*$' "$JOBS" | xargs -P "$NPAR" -L 1 bash -c '
  rel=$0; z=$1; s=$2; dif=$3
  tag=q$(printf "%+d" "$z")_s$s
  echo "$(date +%F_%T) start $rel $tag"
  nice -n 19 "'"$PY"'" "'"$HERE"'/run_state.py" "'"$ROOT"'" "$rel" "$z" "$s" "$dif" --threads "'"$THREADS"'" \
      --max-memory "'"${LADDER_MAX_MEMORY:-12000}"'" --maxsteps "'"${LADDER_MAXSTEPS:-150}"'" \
      > "'"$ROOT"'/$rel/$tag.out" 2>&1
  echo "$(date +%F_%T) end   $rel $tag exit $?"
'
echo "$(date +%F_%T) queue done"
