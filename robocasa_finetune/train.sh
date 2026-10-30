#!/bin/bash
# GR00T N1.6 RoboCasa Kitchen(24 태스크, PandaOmron) 파인튜닝.
#   설정: NEW_EMBODIMENT, robocasa_finetune/modality_config.py, tune_visual/llm=False(기본값), lr 1e-4(기본값),
#         총 배치 64 (GPU 2 × 32), 60k step, 5k 마다 저장.
#
#   DS=/path/to/lerobot_dataset OUT=/path/to/output bash robocasa_finetune/train.sh
#
#   환경 변수: DS(필수)  OUT(필수, slurm 에선 MODEL_OUTPUT_DIR 도 받음)  NGPU(2)  BS(64)  MAXSTEPS(60000)  SAVESTEPS(5000)
#   slurm:  sbatch --gres=gpu:2 robocasa_finetune/train.sbatch   (재큐되면 마지막 체크포인트에서 이어 학습)
set -uo pipefail
: "${DS:?DS 필요 (LeRobot 형식 RoboCasa 데이터셋 경로)}"
OUT=${OUT:-${MODEL_OUTPUT_DIR:?OUT 필요}}; mkdir -p "$OUT"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" VIDEO_BACKEND=${VIDEO_BACKEND:-torchcodec}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NG=${NGPU:-2}
echo "[$(date -Is)] host=$(hostname) DS=$DS OUT=$OUT ngpu=$NG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
torchrun --nproc_per_node=$NG --master_port=$((29500 + ${SLURM_JOB_ID:-$$} % 1000)) gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B --dataset-path "$DS" \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path "$ROOT/robocasa_finetune/modality_config.py" \
  --num-gpus $NG --global-batch-size ${BS:-64} --gradient-accumulation-steps 1 \
  --max-steps ${MAXSTEPS:-60000} --save-steps ${SAVESTEPS:-5000} --save-total-limit 3 --dataloader-num-workers 8 \
  --output-dir "$OUT"; RC=$?
echo "[$(date -Is)] rc=$RC"; ls "$OUT"; exit $RC
