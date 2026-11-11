#!/usr/bin/env bash
# Submit the whole pipeline with SLURM dependencies:
#   sample → generate[] → judge[] → aggregate ─┐
#                       └→ extract[] ──────────┴→ evaluate[] → report
# Every stage resumes by key, so re-submitting after a failure only redoes the missing work.
# Usage: slurm/submit_all.sh <config.yaml> [first-stage]      (WITH_PARITY=1 also submits the parity array;
#        CLUSTER_ENV=<file> reads cluster settings from <file> instead of slurm/cluster.env)
set -euo pipefail
CONFIG=$1; FIRST=${2:-sample}
cd "$(dirname "$0")/.."
export CLUSTER_ENV=${CLUSTER_ENV:-slurm/cluster.env}   # the sbatch scripts re-read the same file
[ -f "$CLUSTER_ENV" ] || { echo "create $CLUSTER_ENV from slurm/cluster.env.example" >&2; exit 1; }
set -a; source "$CLUSTER_ENV"; set +a
mkdir -p slurm/logs slurm/units

# Build the environment once, on the login node, so array tasks never sync concurrently.
uv sync --frozen

RUN=$(uv run --frozen python -c "from hbench.config import load_config; print(load_config('$CONFIG').run_id)")
common=(--parsable ${ACCOUNT:+--account=$ACCOUNT} ${QOS:+--qos=$QOS})
gpu=("${common[@]}" --partition="$GPU_PARTITION" --gres=gpu:"$GPUS_PER_TASK" --cpus-per-task="$CPUS_PER_GPU_TASK" --mem="$MEM_GPU_TASK")
cpu=("${common[@]}" --partition="$CPU_PARTITION" --cpus-per-task="$CPUS_PER_CPU_TASK" --mem="$MEM_CPU_TASK")
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }

units() {  # writes the units file of a stage and prints its line count
  local f=slurm/units/$RUN-$1.tsv
  uv run --frozen hbench units -c "$CONFIG" --stage "$1" > "$f"
  wc -l < "$f"
}
array() {  # stage, time, dependency, resources...
  local st=$1 t=$2 d=$3; shift 3
  local n; n=$(units "$st")
  sbatch "$@" --time="$t" --job-name="hb-$st-$RUN" --array=0-$((n - 1)) $(dep "$d") slurm/array.sbatch "$st" "$CONFIG" "slurm/units/$RUN-$st.tsv"
}
single() {  # stage, dependency
  sbatch "${cpu[@]}" --time="$TIME_SHORT" --job-name="hb-$1-$RUN" $(dep "$2") slurm/single.sbatch "$1" "$CONFIG"
}

stages=(sample generate judge aggregate extract evaluate report)
skip=1; for s in "${stages[@]}"; do [ "$s" = "$FIRST" ] && skip=0; [ $skip = 1 ] && eval "SKIP_$s=1"; done

J_SAMPLE=""; J_GEN=""; J_JUDGE=""; J_AGG=""; J_EXT=""; J_EVAL=""
[ -z "${SKIP_sample:-}" ]    && J_SAMPLE=$(single sample "")
[ -z "${SKIP_generate:-}" ]  && J_GEN=$(array generate "$TIME_GENERATE" "$J_SAMPLE" "${gpu[@]}")
[ -z "${SKIP_judge:-}" ]     && J_JUDGE=$(array judge "$TIME_JUDGE" "$J_GEN" "${gpu[@]}")
[ -z "${SKIP_aggregate:-}" ] && J_AGG=$(single aggregate "$J_JUDGE")
[ -z "${SKIP_extract:-}" ]   && J_EXT=$(array extract "$TIME_EXTRACT" "$J_GEN" "${gpu[@]}")
EVAL_DEP=$(IFS=:; echo "${J_AGG}${J_AGG:+${J_EXT:+:}}${J_EXT}")
[ -z "${SKIP_evaluate:-}" ]  && J_EVAL=$(array evaluate "$TIME_EVALUATE" "$EVAL_DEP" "${cpu[@]}")
J_REP=$(single report "$J_EVAL")
# Parity tests (D65): after sample (they compare against stored answers when generate is done).
J_PAR=""
[ -n "${WITH_PARITY:-}" ] && J_PAR=$(array parity "$TIME_PARITY" "$J_GEN" "${gpu[@]}")
echo "run $RUN: sample=$J_SAMPLE generate=$J_GEN judge=$J_JUDGE aggregate=$J_AGG extract=$J_EXT evaluate=$J_EVAL report=$J_REP parity=$J_PAR"
