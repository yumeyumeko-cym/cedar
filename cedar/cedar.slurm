#!/bin/bash
#SBATCH --job-name=cedar_paircon_unsw
#SBATCH --account=rrg-benliang
#SBATCH --mail-user=ymk.chen@mail.utoronto.ca
#SBATCH --mail-type=FAIL
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=cedar_paircon_%A_%a.out
#SBATCH --error=cedar_paircon_%A_%a.err
#SBATCH --array=0-7%4
#SBATCH --mem=250G

set -euo pipefail

module purge
module load gcc arrow/17.0.0
source /scratch/yimingc/cedar/cedar_venvs/bin/activate

# ---------- paths ----------
CEDAR_PROJECT_ROOT="/scratch/yimingc/cedar"
DATASET="nf-unsw"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_GROUP="run_${JOB_TAG}_paircon"

TRAIN_ROOT="/scratch/yimingc/chipoccon_extension/cedar_checkpoint_unsw_0507"
EVAL_ROOT="/scratch/yimingc/chipoccon_extension/cedar_eval_unsw_0507"
TRAIN_OUT_DIR="${TRAIN_ROOT}/${RUN_GROUP}/config_${ARRAY_ID}"
EVAL_OUT_DIR="${EVAL_ROOT}/${RUN_GROUP}/config_${ARRAY_ID}"
mkdir -p "$TRAIN_OUT_DIR" "$EVAL_OUT_DIR"

# ---------- search grid ----------
SEEDS=(13 15 17 23 25 27 33 35 37 42)
CUT_OFF_LIST=(1)
EMBEDDING_DIM_LIST=(32)
BATCH_SIZE_LIST=(512)
TAU_WEAK_LIST=(0.07 0.3)
LAMBDA_WEAK_LIST=(3.0)
NUM_CLUSTERS_LIST=(30)
KAPPA_LIST=(20.0 50.0)
NUM_POSITIVES_LIST=(1 4)

# ---------- fixed hyperparameters ----------
HIDDEN_DIM=""
DROPOUT="0.1"
LEARNING_RATE="5e-4"
WEIGHT_DECAY="1e-4"
MAX_GRAD_NORM="1.0"
WARMUP_EPOCHS="50"
EM_ITERS="10"
ENCODER_EPOCHS="8"
INIT_RESTARTS="1"
PI_SMOOTHING="0.01"
MIN_COMPONENT_WEIGHT="1e-3"
NUM_WORKERS="4"
DEVICE="auto"
RESULTS_FILE="cedar_paircon_eval_results_unsw.csv"
F1_PERCENTILE="auto"
FIXED_TOP_PERCENT="5.0"

# ---------- index unrolling over the grid ----------
LIST_NAMES=(CUT_OFF_LIST EMBEDDING_DIM_LIST BATCH_SIZE_LIST TAU_WEAK_LIST \
            LAMBDA_WEAK_LIST NUM_CLUSTERS_LIST KAPPA_LIST NUM_POSITIVES_LIST)

TOTAL_CONFIGS=1
for name in "${LIST_NAMES[@]}"; do
  declare -n _arr="$name"
  TOTAL_CONFIGS=$(( TOTAL_CONFIGS * ${#_arr[@]} ))
done

if (( ARRAY_ID >= TOTAL_CONFIGS )); then
  echo "[WARN] ARRAY_ID ${ARRAY_ID} exceeds config space (${TOTAL_CONFIGS}); exiting."
  exit 0
fi

idx=$ARRAY_ID
declare -A PICK
for name in "${LIST_NAMES[@]}"; do
  declare -n _arr="$name"
  n=${#_arr[@]}
  PICK[$name]="${_arr[$(( idx % n ))]}"
  idx=$(( idx / n ))
done

CUT_OFF="${PICK[CUT_OFF_LIST]}"
EMB_DIM="${PICK[EMBEDDING_DIM_LIST]}"
BATCH_SIZE="${PICK[BATCH_SIZE_LIST]}"
TAU_WEAK="${PICK[TAU_WEAK_LIST]}"
LAMBDA_WEAK="${PICK[LAMBDA_WEAK_LIST]}"
NUM_CLUSTERS="${PICK[NUM_CLUSTERS_LIST]}"
KAPPA="${PICK[KAPPA_LIST]}"
NUM_POSITIVES="${PICK[NUM_POSITIVES_LIST]}"

# ---------- threading caps ----------
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ---------- optional flags ----------
HIDDEN_DIM_ARGS=()
if [[ -n "$HIDDEN_DIM" ]]; then
  HIDDEN_DIM_ARGS=(--hidden_dim "$HIDDEN_DIM")
fi

# ---------- launch ----------
printf '[INFO] Config %d/%d | cut=%s emb=%s bs=%s tau=%s lambda=%s clusters=%s kappa=%s C=%s\n' \
  "$ARRAY_ID" "$TOTAL_CONFIGS" \
  "$CUT_OFF" "$EMB_DIM" "$BATCH_SIZE" "$TAU_WEAK" "$LAMBDA_WEAK" "$NUM_CLUSTERS" "$KAPPA" "$NUM_POSITIVES"
printf '[INFO] Seeds: %s\n' "${SEEDS[*]}"
printf '[INFO] Checkpoints: %s\n' "$TRAIN_OUT_DIR"
printf '[INFO] Eval outputs: %s\n' "$EVAL_OUT_DIR"
printf '[INFO] Started at %s\n' "$(date)"

cd "$CEDAR_PROJECT_ROOT"
python -m cedar \
  --dataset "$DATASET" \
  --train_out_dir "$TRAIN_OUT_DIR" \
  --eval_out_dir "$EVAL_OUT_DIR" \
  --seeds "${SEEDS[@]}" \
  --cut_off_list "$CUT_OFF" \
  --embedding_dim_list "$EMB_DIM" \
  --batch_size_list "$BATCH_SIZE" \
  "${HIDDEN_DIM_ARGS[@]}" \
  --dropout "$DROPOUT" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --tau_weak_list "$TAU_WEAK" \
  --lambda_weak_list "$LAMBDA_WEAK" \
  --num_clusters_list "$NUM_CLUSTERS" \
  --em_iters "$EM_ITERS" \
  --encoder_epochs "$ENCODER_EPOCHS" \
  --kappa_list "$KAPPA" \
  --init_restarts "$INIT_RESTARTS" \
  --num_positives "$NUM_POSITIVES" \
  --pi_smoothing "$PI_SMOOTHING" \
  --min_component_weight "$MIN_COMPONENT_WEIGHT" \
  --num_workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --results_file "$RESULTS_FILE" \
  --f1_percentile "$F1_PERCENTILE" \
  --fixed_top_percent "$FIXED_TOP_PERCENT"

printf '[INFO] Finished at %s\n' "$(date)"
