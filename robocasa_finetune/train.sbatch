#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --requeue
# slurm 래퍼. 제출 셸에서 DS, OUT(또는 MODEL_OUTPUT_DIR) export, 환경은 CONDA_ENV(기본 n16) 로 켠다.
#   DS=... OUT=... sbatch robocasa_finetune/train.sbatch
set -uo pipefail
if [ -n "${CONDA_ENV:-n16}" ] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "${CONDA_ENV:-n16}"
fi
bash "${SLURM_SUBMIT_DIR:-.}/robocasa_finetune/train.sh"
