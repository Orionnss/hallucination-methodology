#!/usr/bin/env bash
# A stage that runs once for the whole experiment (sample, aggregate, report).
# Usage: sbatch slurm/single.sbatch <stage> <config>
#SBATCH --job-name=hbench
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
STAGE=$1; CONFIG=$2
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -f slurm/cluster.env ] && set -a && source slurm/cluster.env && set +a
echo "[$(date -Is)] $STAGE on $(hostname)"
exec uv run --frozen hbench "$STAGE" -c "$CONFIG"
