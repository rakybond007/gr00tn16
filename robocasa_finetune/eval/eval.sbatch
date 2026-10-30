#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue
# 8 샤드 배열 평가 (샤드당 태스크 3개). 제출 셸에서 CKPT RUN POLICY_PY EVAL_PY 를 export.
#   CKPT=... RUN=... POLICY_PY=$CONDA_PREFIX/../n16/bin/python EVAL_PY=.../robocasa_eval/bin/python \
#   SHARDS="0 1 2 3 4 5 6 7" sbatch --array=0-7 robocasa_finetune/eval/eval.sbatch
# 재큐되면 이미 끝난 에피소드는 건너뛴다.
set -uo pipefail
bash "${SLURM_SUBMIT_DIR:-.}/robocasa_finetune/eval/run_eval.sh"
