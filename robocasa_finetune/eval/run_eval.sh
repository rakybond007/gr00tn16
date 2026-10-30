#!/bin/bash
# RoboCasa Kitchen 24 태스크 평가: N1.6 정책 서버 1개(GPU) + 태스크마다 클라이언트 1개(같은 GPU 에서 렌더링).
# 프로토콜은 GR00T N1.5 와 같다: seed 42, generative textures, 50 ep/태스크, 16스텝 청크 전체 실행(EXEC_H=16).
#
#   CKPT=/path/to/checkpoint-60000 RUN=my_run bash robocasa_finetune/eval/run_eval.sh
#
#   필수: CKPT(체크포인트 폴더, config.json 이 있는 곳)  RUN(결과 폴더 이름)
#   선택: POLICY_PY(서버 python, 기본 `python`)  EVAL_PY(클라이언트 python, 기본 `python`)
#         EXEC_H(16)  N_EP(50)  SEED(42)  NO_VIDEO(비우면 영상 저장)  EVAL_ROOT(./eval_out)
#         TASKS("CloseDrawer OpenDrawer ..." 공백/쉼표) 또는 SHARD=i/n (24 태스크를 n 등분한 i 번째)
#         slurm 배열이면 SHARDS="0 1 2 ..." 를 주면 배열 번호로 그중 하나를 고른다(n=8).
#   출력: $EVAL_ROOT/$RUN/<task>/prediction.txt, eval.log, (영상)  — 다시 돌리면 끝난 에피소드는 건너뛴다.
#   요약: python robocasa_finetune/eval/summarize.py $EVAL_ROOT/$RUN
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
POLICY_PY=${POLICY_PY:-python}
EVAL_PY=${EVAL_PY:-python}
ALL_TASKS="TurnSinkSpout TurnOnStove TurnOnSinkFaucet TurnOnMicrowave TurnOffStove TurnOffSinkFaucet TurnOffMicrowave PnPStoveToCounter PnPSinkToCounter PnPMicrowaveToCounter PnPCounterToStove PnPCounterToSink PnPCounterToMicrowave PnPCounterToCab PnPCabToCounter OpenSingleDoor OpenDrawer OpenDoubleDoor CoffeeSetupMug CoffeeServeMug CoffeePressButton CloseSingleDoor CloseDrawer CloseDoubleDoor"
if [ -z "${SHARD:-}" ] && [ -n "${SHARDS:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  a=($SHARDS); SHARD=${a[$SLURM_ARRAY_TASK_ID]}/8
fi
if [ -n "${SHARD:-}" ]; then
  i=${SHARD%/*}; n=${SHARD#*/}; arr=($ALL_TASKS); TASKS=""
  for ((j=i; j<${#arr[@]}; j+=n)); do TASKS="$TASKS ${arr[$j]}"; done
fi
TASKS="${TASKS:-$ALL_TASKS}"; TASKS="${TASKS//,/ }"
N_EP="${N_EP:-50}"; EXEC_H="${EXEC_H:-16}"; SEED="${SEED:-42}"
: "${RUN:?RUN 필요}"; : "${CKPT:?CKPT 필요}"
[ -f "$CKPT/config.json" ] || { echo "[ERR] $CKPT 에 config.json 이 없다 (checkpoint-XXXX 폴더를 주어야 한다)"; exit 1; }
JID=${SLURM_JOB_ID:-$$}
PORT="${PORT:-$((30000 + (JID * 7 + ${SLURM_ARRAY_TASK_ID:-0} * 13) % 9000))}"
OUT="${EVAL_ROOT:-$ROOT/eval_out}/$RUN"; mkdir -p "$OUT"
echo "node=$(hostname) run=$RUN ckpt=$CKPT tasks=[$TASKS] n_ep=$N_EP exec_h=$EXEC_H seed=$SEED port=$PORT"
cleanup(){ kill ${SPID:-} 2>/dev/null; sleep 2; kill -9 ${SPID:-} 2>/dev/null; }
trap cleanup EXIT
TAG=$(echo $TASKS | cksum | cut -d' ' -f1)
# 정책 서버 (N1.6 기본 스크립트). 체크포인트·베이스 모델은 미리 받아 둔 캐시만 쓴다.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 \
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1} \
  $POLICY_PY -u "$ROOT/gr00t/eval/run_gr00t_server.py" --model-path "$CKPT" --embodiment-tag NEW_EMBODIMENT \
  --use-sim-policy-wrapper --host 127.0.0.1 --port $PORT > "$OUT/server_$TAG.log" 2>&1 &
SPID=$!
for k in $(seq 1 240); do
  grep -q "Server is ready" "$OUT/server_$TAG.log" 2>/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "[ERR] 서버가 죽었다"; tail -40 "$OUT/server_$TAG.log"; exit 1; }
  sleep 5
done
echo "[i] server ready"
PIDS=()
for T in $TASKS; do
  TD="$OUT/$T"; mkdir -p "$TD"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
    $EVAL_PY -u "$ROOT/robocasa_finetune/eval/robocasa_client.py" --port $PORT --host 127.0.0.1 \
    --action_horizon $EXEC_H --env_name "$T" --video_dir "$TD" --seed $SEED --n_episodes $N_EP \
    --max_episode_steps 1500 --generative_textures ${NO_VIDEO:+--no_video} > "$TD/eval.log" 2>&1 &
  PIDS+=($!); sleep 2
done
for p in "${PIDS[@]}"; do wait $p; done
for T in $TASKS; do echo "$T episodes=$(grep -c episode $OUT/$T/prediction.txt 2>/dev/null) success=$(grep -c True $OUT/$T/prediction.txt 2>/dev/null)"; done
echo "[$(date -Is)] DONE"
