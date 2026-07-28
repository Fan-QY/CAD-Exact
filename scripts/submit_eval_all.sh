#!/bin/bash
# Fill the results table: every model evaluated on every test set.
#
# Five models (the joint baseline plus the four continual stages) crossed with
# four operations is 20 inference+eval jobs. This release ships the final stage
# only, so set STAGES to match the checkpoints you actually have. They are independent, so they are
# submitted as 20 separate short jobs rather than one long one -- on a busy
# partition twenty 12-hour jobs schedule far better than one 10-day job, and a
# single failed cell can be resubmitted alone.
#
#   bash scripts/submit_eval_all.sh 1              # 1-GPU jobs, g1 runs
#   bash scripts/submit_eval_all.sh 2 g2           # 2-GPU jobs, g2 runs
#   STAGES=loft bash scripts/submit_eval_all.sh 1  # just the released final model
#
# Cells whose checkpoint is absent are skipped with a warning rather than
# submitted, since a missing fused directory would only fail at runtime.

set -euo pipefail

NGPU="${1:-1}"
GROUP="${2:-g${NGPU}}"

SCRATCH=/scratch/$USER
CODE=$SCRATCH/CAD-Exact
RUNS=$SCRATCH/runs/$GROUP

STAGES="${STAGES:-joint extrude loft revolve sweep}"
OPS="${OPS:-extrude loft revolve sweep}"

cd "$CODE"
mkdir -p logs

submitted=0
skipped=0
for stage in $STAGES; do
  if [[ "$stage" == "joint" ]]; then ckpt=187500; else ckpt=46875; fi
  fused="$RUNS/cad-exact-$stage/checkpoint-$ckpt-fused"
  if [[ ! -f "$fused/config.json" ]]; then
    echo "skip $stage: no fused checkpoint at $fused"
    skipped=$(( skipped + 4 ))
    continue
  fi
  for op in $OPS; do
    jid=$(sbatch --parsable \
      --gres="gpu:h200:${NGPU}" \
      --job-name="eval-${stage}-${op}" \
      --export=ALL,STAGE="$stage",OP="$op",RUNS="$RUNS",CKPT="$ckpt",NGPU="$NGPU" \
      scripts/infer_and_eval_slurm.sh)
    echo "  $stage on $op -> $jid"
    submitted=$(( submitted + 1 ))
  done
done

echo
echo "submitted $submitted cells, skipped $skipped."
echo "reports land in $SCRATCH/inference/<stage>-ckpt<N>/<op>/eval_report.json"
