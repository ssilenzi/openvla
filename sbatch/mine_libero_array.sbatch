#!/bin/bash
#SBATCH --job-name=openvla-libero-mine-eval
#SBATCH --partition=boost_usr_prod
#SBATCH --account=pnrr_fit4medrob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_L40S_45G|gpu_A40_45G"
#SBATCH --time=24:00:00
#SBATCH --output=logs/mine_libero_eval_%A_%a.out
#SBATCH --error=logs/mine_libero_eval_%A_%a.err
#SBATCH --array=0-1            # 0=spatial, 1=object

set -euo pipefail

# ============================================================================
# USER CONFIGURATION — edit before submitting.
# Submit from the repo root: `cd <repo> && sbatch sbatch/mine_libero_array.sbatch`
# ============================================================================
# Conda environment with openvla dependencies installed.
CONDA_ENV="/work/pnrr_fit4medrob/VLA/venvs/env_openvla"

# Weights & Biases account.
WANDB_ENTITY="ssilenzi-university-of-modena-and-reggio-emilia"
WANDB_PROJECT="openvla-libero-eval"

# Cache redirects (use a writable location off shared /homes).
HF_HOME="/work/pnrr_fit4medrob/VLA/users/ssilenzi/hf_cache"
TORCH_HOME="/work/pnrr_fit4medrob/VLA/users/ssilenzi/torch_cache"
XDG_CACHE_HOME="/work/pnrr_fit4medrob/VLA/users/ssilenzi/xdg_cache"
TMPDIR="/work/pnrr_fit4medrob/VLA/users/ssilenzi/tmp"
WANDB_DIR="/work/pnrr_fit4medrob/VLA/users/ssilenzi/wandb"

# Where local fine-tuned checkpoints live (one subdir per suite).
RUN_ROOT_BASE="/work/pnrr_fit4medrob/VLA/runs"
# ============================================================================

# ---------- Repo anchor (logs go to $REPO_ROOT/logs) ----------
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f "$REPO_ROOT/experiments/robot/libero/run_libero_eval.py" ]]; then
    echo "[error] Submit from the openvla repo root:" >&2
    echo "        cd <repo-root> && sbatch sbatch/mine_libero_array.sbatch" >&2
    echo "        SLURM_SUBMIT_DIR='$REPO_ROOT' is not the repo root." >&2
    exit 1
fi
mkdir -p "$REPO_ROOT/logs" "$TMPDIR" "$WANDB_DIR" "$WANDB_DIR/cache"

# ---------- Conda env ----------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ---------- Cache redirects (off /homes) ----------
export HF_HOME TORCH_HOME XDG_CACHE_HOME TMPDIR

# ---------- W&B ----------
export WANDB_MODE=online
export WANDB_DIR
export WANDB_CACHE_DIR="$WANDB_DIR/cache"

# ---------- LIBERO config (per-project, isolates from ~/.libero) ----------
export LIBERO_CONFIG_PATH="$REPO_ROOT/.libero"

# ---------- MuJoCo headless rendering (EGL) ----------
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=device
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# ---------- Array index -> (suite, local checkpoint) ----------
SUITES=(libero_spatial libero_object)
SUITE=${SUITES[$SLURM_ARRAY_TASK_ID]}

# Same glob as the training script: save_latest_checkpoint_only=True keeps a
# single checkpoint dir per suite. Skip .staging / .prev siblings.
RUN_ROOT="$RUN_ROOT_BASE/$SUITE"
CKPT=$(ls -d "$RUN_ROOT"/openvla-7b+*--image_aug 2>/dev/null \
       | grep -v '\.\(prev\|staging\)$' | head -1 || true)

if [[ -z "$CKPT" || ! -f "$CKPT/model.safetensors.index.json" ]]; then
    echo "[error] No valid checkpoint under $RUN_ROOT" >&2
    exit 1
fi

echo "=== Job $SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID on $(hostname) ==="
echo "Suite:      $SUITE"
echo "Checkpoint: $CKPT"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "from libero.libero import benchmark; print('libero OK')"
python -c "import robosuite; print('robosuite', robosuite.__version__)"

cd "$REPO_ROOT"

srun python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "$CKPT" \
  --task_suite_name "$SUITE" \
  --center_crop True \
  --run_id_note "mine_${SUITE}_seed7" \
  --seed 7 \
  --use_wandb True \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_entity "$WANDB_ENTITY"
