#!/usr/bin/env bash
# One array task = one work unit of a stage (prompt §3.2):
#   generate / extract / evaluate : one (generator × dataset)
#   judge                         : one (judge × generator × dataset)
# Usage (from submit_all.sh): sbatch --array=0-(N-1) slurm/array.sbatch <stage> <config> <units-file>
#SBATCH --job-name=hbench
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
STAGE=$1; CONFIG=$2; UNITS=$3
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f slurm/cluster.env ] && set -a && source slurm/cluster.env && set +a
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$UNITS")
IFS=$'\t' read -r A B C <<<"$LINE"
echo "[$(date -Is)] $STAGE task $SLURM_ARRAY_TASK_ID on $(hostname): $LINE"
nvidia-smi -L 2>/dev/null || true
case "$STAGE" in
  judge)    exec ./bin/hbench-judge -c "$CONFIG" --judge "$A" --generator "$B" --dataset "$C" ;;
  generate|extract|evaluate)
            exec uv run --frozen hbench "$STAGE" -c "$CONFIG" --generator "$A" --dataset "$B" ;;
  *) echo "unknown array stage $STAGE" >&2; exit 2 ;;
esac
