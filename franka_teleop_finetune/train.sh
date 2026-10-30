#!/bin/bash
# GR00T N1.6 Franka 텔레옵 데이터 파인튜닝.
#   설정: NEW_EMBODIMENT, franka_teleop_finetune/modality_config.py, tune_visual/llm=False(기본값), lr 1e-4(기본값).
#   기본값(GPU 1장, 배치 4, 20 step)은 스모크 테스트용이다. 실제 학습은 아래처럼 값을 준다.
#
#   DS=/path/to/lerobot/<TASK_ID> OUT=/path/to/output bash franka_teleop_finetune/train.sh
#   실제 학습 예:  NGPU=2 BS=64 MAXSTEPS=20000 SAVESTEPS=2000 DS=... OUT=... bash franka_teleop_finetune/train.sh
#   32GB GPU 스모크: OPTIM=paged_adamw_8bit DS=... OUT=... bash franka_teleop_finetune/train.sh
#
#   환경 변수: DS(필수)  OUT(필수)  NGPU(1)  BS(4)  GRAD_ACC(1)  MAXSTEPS(20)  SAVESTEPS(MAXSTEPS)  WORKERS(4)
#   메모리: OPTIM(adamw_torch)  TUNE_TOP_LLM_LAYERS(4)  GRAD_CKPT(0) — launch.py 참고
#   데이터셋에 meta/modality.json 이 없으면 prepare_dataset.py 로 먼저 만든다.
set -uo pipefail
: "${DS:?DS 필요 (rlwrld-franka-teleop 로 변환한 LeRobot 데이터셋 경로)}"
: "${OUT:?OUT 필요}"; mkdir -p "$OUT"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" VIDEO_BACKEND=${VIDEO_BACKEND:-torchcodec}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NG=${NGPU:-1}
MAXSTEPS=${MAXSTEPS:-20}

if [ ! -f "$DS/meta/modality.json" ]; then
  python franka_teleop_finetune/prepare_dataset.py --dataset-path "$DS" || exit 1
fi

echo "[$(date -Is)] host=$(hostname) DS=$DS OUT=$OUT ngpu=$NG bs=${BS:-4} steps=$MAXSTEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
torchrun --nproc_per_node=$NG --master_port=$((29500 + $$ % 1000)) franka_teleop_finetune/launch.py \
  --base-model-path nvidia/GR00T-N1.6-3B --dataset-path "$DS" \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path "$ROOT/franka_teleop_finetune/modality_config.py" \
  --num-gpus $NG --global-batch-size ${BS:-4} --gradient-accumulation-steps ${GRAD_ACC:-1} \
  --max-steps $MAXSTEPS --save-steps ${SAVESTEPS:-$MAXSTEPS} --save-total-limit 2 \
  --dataloader-num-workers ${WORKERS:-4} \
  --output-dir "$OUT"; RC=$?
echo "[$(date -Is)] rc=$RC"; ls "$OUT"; exit $RC
