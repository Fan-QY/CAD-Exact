#!/bin/bash
# Inference + Tier-1/2/3 evaluation for ONE (model, operation) pair.
#
# Every reportable number is a cell in a 5x4 table: five models (joint, and the
# four continual stages) crossed with the four test sets. Editing paths in a file
# per cell is how the old version of this script went stale against the training
# chain, so nothing is hardcoded here -- the cell is chosen entirely by
# environment variables and scripts/submit_eval_all.sh fills the whole table.
#
#   STAGE=sweep OP=loft sbatch scripts/infer_and_eval_slurm.sh
#
# Variables (all optional except STAGE and OP):
#   STAGE   joint | extrude | loft | revolve | sweep   which trained model
#   OP      extrude | loft | revolve | sweep           which test set
#   RUNS    run root, default /scratch/$USER/runs/g1
#   CKPT    checkpoint number, default 187500 for joint else 46875
#   NGPU    1 or 2, default 2 (2 shards the test set across both cards)
#
#SBATCH --job-name=cad-exact-eval
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
# --gres is passed by the submit script so one file serves 1- and 2-GPU runs.

set -euo pipefail

STAGE="${STAGE:?set STAGE=joint|extrude|loft|revolve|sweep}"
OP="${OP:?set OP=extrude|loft|revolve|sweep}"

SCRATCH=/scratch/$USER
CODE_ROOT=$SCRATCH/CAD-Exact
PARQUET_ROOT=$SCRATCH/GeomCAD
TEST_ROOT=$SCRATCH/test_extracted
CACHE_DIR=$SCRATCH/hf_cache
SIF="${SIF:-$SCRATCH/CADLLM.sif}"

RUNS="${RUNS:-$SCRATCH/runs/g1}"
# joint runs the staged chain's total budget in one pass, so its final
# checkpoint number differs from the per-stage one.
if [[ "$STAGE" == "joint" ]]; then
  CKPT="${CKPT:-187500}"
else
  CKPT="${CKPT:-46875}"
fi
NGPU="${NGPU:-2}"

MODEL_DIR="$RUNS/cad-exact-$STAGE"
FUSED_DIR="$MODEL_DIR/checkpoint-$CKPT-fused"
SAVEDIR="$SCRATCH/inference/${STAGE}-ckpt${CKPT}/${OP}"

BS=32
NP=256
MNT=1024

mkdir -p logs "$SAVEDIR"
cd "$CODE_ROOT"

# singularity is a module, not on the default PATH, and a batch shell is
# non-login so the module function may need sourcing first.
# Partitions do not share a module system. The h200 nodes expose `module` in a
# batch shell already; the A100 nodes use Lmod from a different prefix, so the
# two hardcoded init paths found nothing there and the next line died with exit
# 127 one second into the job -- "command not found" for `module` itself, which
# set -e then turned into an instant failure that looked like a permission or
# path problem. Probe the usual prefixes, and skip the module system entirely if
# the runtime is already reachable.
if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  if ! command -v module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
                /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash \
                /opt/apps/lmod/lmod/init/bash /usr/local/Modules/init/bash; do
      [[ -f "$init" ]] && source "$init" && break
    done
  fi
  if command -v module >/dev/null 2>&1; then
    module load singularity/4.4.0 2>/dev/null || module load apptainer 2>/dev/null || true
  fi
fi
if ! command -v singularity >/dev/null 2>&1; then
  if command -v apptainer >/dev/null 2>&1; then
    # Apptainer is the renamed successor and is CLI-compatible for what we use.
    singularity() { apptainer "$@"; }
    export -f singularity
  else
    echo "neither singularity nor apptainer on PATH after module probing" >&2
    echo "module system: $(command -v module || echo absent)" >&2
    exit 1
  fi
fi

# Probe the image for pyarrow and wire in an alternate torch if needed.
PYLIBS="$SCRATCH/pylibs"
if singularity exec "$SIF" python -c "import pyarrow" 2>/dev/null; then
  PYLIBS=""
elif [[ ! -d "$PYLIBS/pyarrow" ]]; then
  echo "image lacks pyarrow and none at $PYLIBS -- the test-split extraction needs it" >&2
  exit 1
fi

PYPATH="$PYLIBS"
if [[ -n "${TORCH_LIBS:-}" ]]; then
  PYPATH="${PYLIBS:+$TORCH_LIBS:$PYLIBS}"
  PYPATH="${PYPATH:-$TORCH_LIBS}"
  : "${CAD_ATTN:=sdpa}"
  : "${CAD_FPS_TORCH:=1}"
fi
ENVARGS=()
[[ -n "$PYPATH" ]] && ENVARGS+=(--env "PYTHONPATH=$PYPATH")
[[ -n "${CAD_ATTN:-}" ]]      && ENVARGS+=(--env "CAD_ATTN=$CAD_ATTN")
[[ -n "${CAD_FPS_TORCH:-}" ]] && ENVARGS+=(--env "CAD_FPS_TORCH=$CAD_FPS_TORCH")

SEXEC="singularity exec --nv -B /scratch:/scratch ${ENVARGS[*]} $SIF"

# A job submitted without --gres gets the node but no card, and then burns its
# whole wall clock discovering that. Check before doing any work.
VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [[ "$VISIBLE_GPUS" -lt 1 ]]; then
  echo "no GPU visible in this allocation -- was --gres passed?" >&2
  echo "  gres type is node-specific; check with: scontrol show node <gpu-node> | grep Gres=" >&2
  exit 1
fi
if [[ "$NGPU" == "2" && "$VISIBLE_GPUS" -lt 2 ]]; then
  echo "NGPU=2 but only $VISIBLE_GPUS GPU visible; falling back to single-GPU inference" >&2
  NGPU=1
fi

echo "=== cell: model=$STAGE ckpt=$CKPT  test=$OP  gpus=$NGPU"
echo "    model dir : $MODEL_DIR"
echo "    savedir   : $SAVEDIR"

# ---- Step 0: the fused weights must already exist -------------------------
# The released weights are already LoRA-merged, so this directory should exist
# as downloaded. If it does not, the layout is wrong -- stop and say so rather
# than silently running on the base model.
if [[ ! -f "$FUSED_DIR/config.json" ]]; then
  echo "fused checkpoint missing: $FUSED_DIR" >&2
  echo "expected layout: \$MODEL_DIR/checkpoint-\$CKPT-fused/{config.json,model.safetensors,...}" >&2
  exit 1
fi

# ---- Step 1: materialise the test split (once per op, shared by all models) --
# Several cells of the table run at the same time and would race on the same op
# directory, so the extraction is serialised on a per-op lock. Both shards are
# always written, which lets a 1-GPU and a 2-GPU run share one extraction.
mkdir -p "$TEST_ROOT"
(
  flock 9
  if [[ ! -f "$TEST_ROOT/$OP/shard1/test_mesh_paths.json" ]]; then
    echo "=== $(date) extracting test split for $OP ==="
    $SEXEC python "$CODE_ROOT/tools/extract_test_from_parquet.py" \
        --parquet_root "$PARQUET_ROOT" \
        --ops "$OP" \
        --out "$TEST_ROOT" \
        --shards 2
  else
    echo "test split for $OP already extracted"
  fi
) 9>"$TEST_ROOT/.extract-$OP.lock"

# ---- Step 2: inference ------------------------------------------------------
# --modeldir takes the directory *containing* checkpoint-<N>-fused, not the
# fused directory itself; infer_batched.py joins the two.
echo "=== $(date) START inference ==="
if [[ "$NGPU" == "2" ]]; then
  for S in 0 1; do
    CUDA_VISIBLE_DEVICES=$S $SEXEC python -u infer_batched.py \
        --datainfo "$TEST_ROOT/$OP/shard$S" \
        --modeldir "$MODEL_DIR" \
        --checkpoint "$CKPT" \
        --savedir "$SAVEDIR" \
        --batch_size $BS --n_points $NP --max_new_tokens $MNT \
        > "logs/infer_${STAGE}_${OP}_gpu${S}.log" 2>&1 &
  done
  wait
else
  $SEXEC python -u infer_batched.py \
      --datainfo "$TEST_ROOT/$OP" \
      --modeldir "$MODEL_DIR" \
      --checkpoint "$CKPT" \
      --savedir "$SAVEDIR" \
      --batch_size $BS --n_points $NP --max_new_tokens $MNT
fi
echo "=== $(date) inference done -> $SAVEDIR ==="

# ---- Step 3: Tier 1/2/3 evaluation -----------------------------------------
echo "=== $(date) START eval ==="
$SEXEC python "$CODE_ROOT/eval.py" \
    --gt_dir   "$TEST_ROOT/$OP" \
    --pred_dir "$SAVEDIR" \
    --out_json "$SAVEDIR/eval_report.json" \
    --per_sample_json "$SAVEDIR/eval_per_sample.json" \
    --name "${STAGE}-ckpt${CKPT}-on-${OP}"

echo "=== $(date) DONE: $SAVEDIR/eval_report.json ==="
