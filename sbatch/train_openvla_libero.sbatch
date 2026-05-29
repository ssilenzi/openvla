#!/bin/bash
#SBATCH --job-name=openvla-ft-libero
#SBATCH --partition=boost_usr_prod
#SBATCH --account=pnrr_fit4medrob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu_L40S_45G|gpu_A40_45G"
#SBATCH --time=23:30:00                 # margine 30min sotto il limite 24h
#SBATCH --signal=B:USR1@1800            # USR1 30 min prima del timeout
#SBATCH --requeue
#SBATCH --output=logs/ft_%x_%j.out
#SBATCH --error=logs/ft_%x_%j.err

set -euo pipefail

# ============================================================================
# USER CONFIGURATION — edit before submitting.
# Launch: cd <repo-root> && sbatch sbatch/train_openvla_libero.sbatch [SUITE]
#   SUITE = libero_spatial | libero_object | libero_goal | libero_10
#           (default: libero_spatial)
# ============================================================================
# Conda environment with openvla dependencies installed.
CONDA_ENV="/work/pnrr_fit4medrob/VLA/venvs/env_openvla"

# Weights & Biases account.
WANDB_ENTITY="ssilenzi-university-of-modena-and-reggio-emilia"
WANDB_PROJECT="openvla-libero-ft"

# Cache redirects (use a writable location off shared /homes).
HF_HOME="/work/pnrr_fit4medrob/VLA/users/ssilenzi/hf_cache"
TORCH_HOME="/work/pnrr_fit4medrob/VLA/users/ssilenzi/torch_cache"
TMPDIR="/work/pnrr_fit4medrob/VLA/users/ssilenzi/tmp"

# RLDS dataset root (contains <suite>_no_noops/ subdirs).
DATA_ROOT_DIR="/work/pnrr_fit4medrob/VLA/datasets/modified_libero_rlds"

# Where fine-tuning checkpoints are written (one subdir per suite).
RUN_ROOT_BASE="/work/pnrr_fit4medrob/VLA/runs"
# ============================================================================

# ---------------- Param from CLI ----------------
SUITE=${1:-libero_spatial}
DATASET_NAME="${SUITE}_no_noops"

# ---------------- Repo anchor (logs go to $REPO_ROOT/logs) ----------------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "$REPO_ROOT/vla-scripts/finetune.py" ]]; then
    echo "[error] Submit from the openvla repo root:" >&2
    echo "        cd <repo-root> && sbatch sbatch/train_openvla_libero.sbatch [SUITE]" >&2
    echo "        SLURM_SUBMIT_DIR='$REPO_ROOT' is not the repo root." >&2
    exit 1
fi
mkdir -p "$REPO_ROOT/logs" "$TMPDIR"

# ---------------- Path ----------------
DATA_ROOT="$DATA_ROOT_DIR"
RUN_ROOT="$RUN_ROOT_BASE/$SUITE"
ADAPTER_DIR="$RUN_ROOT/adapter_tmp"
mkdir -p "$RUN_ROOT" "$ADAPTER_DIR"

# ---------------- Conda + env ----------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME TORCH_HOME TMPDIR
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# 45 GB cards (L40S/A40) — recover fragmented free memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 32 CPUs / 4 GPU procs = 8 threads per rank
export OMP_NUM_THREADS=8

# ---------------- Resume logic ----------------
# finetune.py owns the resume: if $ADAPTER_DIR/openvla-7b+*--image_aug/ already
# contains a saved adapter (LoRA + training_state.pt), it re-attaches it and
# loads the optimizer state. We always pass the HF base as --vla_path so the
# merge step rebuilds the merged checkpoint from a fixed base + cumulative LoRA.
#
# We still recover here from an interrupted atomic save: if save_latest_only
# was killed mid rename-swap, the previous good merged dir is at "*.prev".
PREV_CKPT=$(ls -d "$RUN_ROOT"/openvla-7b+*--image_aug.prev 2>/dev/null | head -1 || true)
LATEST_CKPT=$(ls -d "$RUN_ROOT"/openvla-7b+*--image_aug 2>/dev/null | grep -v '\.\(prev\|staging\)$' | head -1 || true)
if [[ -z "${LATEST_CKPT:-}" && -n "${PREV_CKPT:-}" && -f "$PREV_CKPT/model.safetensors.index.json" ]]; then
    TARGET="${PREV_CKPT%.prev}"
    echo "[recover] Restoring interrupted save: $PREV_CKPT -> $TARGET"
    mv "$PREV_CKPT" "$TARGET"
fi

VLA_PATH="openvla/openvla-7b"
echo "[start] VLA_PATH=$VLA_PATH; resume detection delegated to finetune.py via $ADAPTER_DIR"

# ---------------- Auto-resubmit on SIGUSR1 ----------------
# When SLURM sends USR1 (30 min before timeout), submit a follow-up job and
# let the current training continue until the wall-clock kill. Atomic-save
# in finetune.py protects the checkpoint from the eventual SIGTERM.
resubmit() {
    echo "[signal] Received SIGUSR1, submitting follow-up job"
    sbatch "$REPO_ROOT/sbatch/train_openvla_libero.sbatch" "$SUITE"
    echo "[signal] Follow-up submitted; current training continues until walltime"
}
trap resubmit USR1

# ---------------- Sanity ----------------
nvidia-smi
echo "suite=$SUITE  dataset=$DATASET_NAME  vla_path=$VLA_PATH"
cd "$REPO_ROOT"

# ---------------- Launch training ----------------
torchrun --standalone --nnodes 1 --nproc-per-node 4 \
    vla-scripts/finetune.py \
    --vla_path "$VLA_PATH" \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET_NAME" \
    --run_root_dir "$RUN_ROOT" \
    --adapter_tmp_dir "$ADAPTER_DIR" \
    --lora_rank 32 \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --image_aug True \
    --save_steps 2000 \
    --max_steps 50000 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --run_id_note "${SUITE}-lora32-bs8x2" \
    &
TRAIN_PID=$!
# wait may be interrupted by SIGUSR1; loop until child actually exits
set +e
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
    TRAIN_EXIT=$?
done
set -e

# If the training session ended normally (not due to a signal), I won't resubmit
echo "[done] Training completed with exit code $TRAIN_EXIT"
exit $TRAIN_EXIT
