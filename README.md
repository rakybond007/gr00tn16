# GR00T N1.6 — RoboCasa Kitchen 파인튜닝·평가

NVIDIA [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 의 `n1.6.1-release` (커밋 `5dc80c4`) 위에
RoboCasa Kitchen 24 태스크(PandaOmron) 파인튜닝·평가 스크립트를 얹은 레포다.
GR00T 코드(`gr00t/` 등)는 **한 줄도 바꾸지 않았고**, 추가한 것은 전부 `robocasa_finetune/` 과 `franka_teleop_finetune/` 안에 있다.
원래 README 는 [`README_ISAAC_GR00T.md`](README_ISAAC_GR00T.md) 에 그대로 두었다.

결과 요약 (같은 데이터·설정, 각 1회 학습, 24 태스크 × 50 ep):

| 모델 | 성공률 |
|---|---|
| GR00T N1.5 | 70.3% |
| **GR00T N1.6 (이 레포)** | **68.8%** |
| GR00T N1.7 | 49.6% |

태스크별 결과와 실행 길이 비교는 [`robocasa_finetune/results/n16_rc_mg_noisy.md`](robocasa_finetune/results/n16_rc_mg_noisy.md).

## 구성

```
robocasa_finetune/
  modality_config.py          RoboCasa 모달리티 설정 (NEW_EMBODIMENT 로 자기 등록)
  train.sh / train.sbatch     파인튜닝 (gr00t/experiment/launch_finetune.py 호출)
  eval/
    run_eval.sh / eval.sbatch 정책 서버 1개 + 태스크별 클라이언트, 24 태스크 평가
    robocasa_client.py        폐루프 평가 클라이언트 (torch 불필요)
    wrappers/                 GR00T N1.5 의 RoboCasa gym 래퍼 (N1.5 와 같은 평가 프로토콜용, Apache-2.0)
    summarize.py              prediction.txt → 태스크별·전체 성공률
  env/
    requirements_n16_freeze.txt     학습·정책 서버 env 전체 버전 (결과를 낸 env 의 pip freeze)
    requirements_robocasa_eval.txt  평가 클라이언트 env
  results/n16_rc_mg_noisy.md
```

환경은 두 개다. 정책(학습·서버)은 torch 2.7 env, RoboCasa 시뮬레이션(클라이언트)은 robosuite/robocasa env 에서 돌고,
둘은 localhost ZMQ 로 통신한다. 버전 충돌 없이 N1.5 와 같은 시뮬레이터 버전을 쓰기 위해서다.

## 1. 환경 셋업

Linux x86_64, CUDA 12.8 을 돌릴 수 있는 NVIDIA 드라이버, python 3.10 기준.

### 1-1. 정책 env (`n16`): 학습·정책 서버

결과를 낸 env 를 그대로 재현하는 방법 (conda + pip):

```bash
git clone https://github.com/rakybond007/gr00tn16.git && cd gr00tn16
conda create -n n16 python=3.10 -y && conda activate n16
pip install -r robocasa_finetune/env/requirements_n16_freeze.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e . --no-deps
```

`requirements_n16_freeze.txt` 의 주요 버전: torch 2.7.1+cu128, torchvision 0.22.1, torchcodec 0.4.0, transformers 4.51.3,
flash-attn 2.7.4.post1 (공식 wheel URL), deepspeed 0.17.6, diffusers 0.35.1, peft 0.17.1, numpy 1.26.4.
`pyproject.toml` 의 핀과 같다(차이는 dm-tree 0.1.10 과, 쓰지 않는 tensorrt 미설치뿐).

공식 방법(uv)으로 깔아도 같은 버전이 된다:

```bash
uv sync && uv pip install -e .     # 또는 bash scripts/deployment/dgpu/install_deps.sh
```

확인:

```bash
python -c "import torch, flash_attn, gr00t; print(torch.__version__, torch.cuda.is_available())"
```

### 1-2. 평가 env (`robocasa_eval`): RoboCasa 시뮬레이션 클라이언트

```bash
conda create -n robocasa_eval python=3.10 -y && conda activate robocasa_eval
pip install -r robocasa_finetune/env/requirements_robocasa_eval.txt

git clone https://github.com/ARISE-Initiative/robosuite.git
(cd robosuite && git checkout e94119512f737635111ae651768c17ad1be72859 && pip install -e .)   # 1.5.2

git clone https://github.com/robocasa/robocasa.git
(cd robocasa && git checkout 756598a5be52e052339bb2d957426e39015c2afb && pip install -e . \
  && python robocasa/scripts/download_kitchen_assets.py \
  && python robocasa/scripts/setup_macros.py)                                                   # 0.2.0, 에셋 수 GB
```

렌더링은 EGL(`MUJOCO_GL=egl`)이라 GPU 노드에서 돌린다. 확인:

```bash
MUJOCO_GL=egl python -c "import robosuite, robocasa; print(robosuite.__version__)"
```

## 2. 데이터

LeRobot 형식 RoboCasa Kitchen 데이터 (24 태스크, PandaOmron, 3 카메라 left/right/wrist):

```bash
huggingface-cli download prehj/rc_mg_noisy --repo-type dataset --local-dir $DATA/rc_mg_noisy
```

액션 12차원 = `0:4 base_motion · 4 control_mode · 5:8 EE 위치 델타 · 8:11 EE 회전 델타 · 11 그리퍼`.
모달리티 키는 데이터셋 `meta/modality.json` 을 그대로 따른다. 액션은 robosuite 컨트롤러가 바로 쓰는 델타 명령이라
`action_configs` 를 비워(변환 없이 그대로) 쓴다. 자세한 이유는 `modality_config.py` 주석 참고.

## 3. 학습

```bash
conda activate n16
DS=$DATA/rc_mg_noisy OUT=$CKPT/n16_rc_mg_noisy bash robocasa_finetune/train.sh
# slurm:  DS=... OUT=... sbatch robocasa_finetune/train.sbatch     (CONDA_ENV 기본 n16)
```

기본값: 베이스 `nvidia/GR00T-N1.6-3B`, `NEW_EMBODIMENT`, GPU 2장 × 배치 32 = 64, 60k step, 5k 마다 저장(최근 3개 보관),
나머지(lr 1e-4, tune_visual/tune_llm=False 등)는 `launch_finetune.py` 기본값. `NGPU BS MAXSTEPS SAVESTEPS` 로 바꿀 수 있다.
첫 실행 때 베이스 모델을 HF 에서 받는다.

주의: `checkpoint-*/global_step*` 의 deepspeed 옵티마이저 상태가 커서, 5 step 스모크에서도 출력 폴더가 31 GB 였다.
평가에는 옵티마이저 상태가 필요 없다. 학습이 끝나면 `SAVESTEPS` 와 상관없이 마지막 체크포인트가 저장된다.

## 4. 평가

정책 서버(`n16` env)와 클라이언트(`robocasa_eval` env)를 한 GPU 에서 같이 띄운다. `CKPT` 는 `config.json` 이 든
`checkpoint-XXXX` 폴더여야 한다(processor 설정이 그 안에 있다).

```bash
CKPT=$CKPT/n16_rc_mg_noisy/checkpoint-60000 RUN=n16_noisy \
POLICY_PY=$(conda run -n n16 which python) EVAL_PY=$(conda run -n robocasa_eval which python) \
TASKS="CloseDrawer" N_EP=2 NO_VIDEO=1 bash robocasa_finetune/eval/run_eval.sh     # 스모크

# 전체 24 태스크를 8 샤드로 (slurm 배열)
CKPT=... RUN=n16_noisy POLICY_PY=... EVAL_PY=... SHARDS="0 1 2 3 4 5 6 7" \
  sbatch --array=0-7 robocasa_finetune/eval/eval.sbatch

python robocasa_finetune/eval/summarize.py eval_out/n16_noisy
```

- 프로토콜(N1.5 와 동일): seed 42, generative textures "100p", obj_instance_split A, styles [0..8, 11], 태스크당 50 ep,
  16스텝 청크를 전부 실행(`EXEC_H=16`). 최대 스텝은 env horizon.
- 결과는 `eval_out/<RUN>/<task>/prediction.txt` 에 `episode i is_success: [True] action_steps: N` 한 줄씩 쌓인다.
  다시 돌리면 끝난 에피소드는 건너뛴다(장면 순서는 유지).
- 서버는 기본으로 오프라인(`HF_HUB_OFFLINE=1`)이다. HF 캐시가 비어 있는 머신이면 `HF_HUB_OFFLINE=0` 으로 한 번 돌린다.
- 실행 길이를 바꾸려면 `EXEC_H` (1~16).

## Franka 텔레옵 데이터 파인튜닝

[rlwrld-franka-teleop](https://github.com/rakybond007/rlwrld-franka-teleop) 으로 수집·변환한 LeRobot v2.1 데이터(`--action-type eef`)를 학습한다.
환경은 위 `n16` 과 같다.

```bash
# 1. 데이터셋에 meta/modality.json 추가 (train.sh 가 없으면 자동으로 실행)
python franka_teleop_finetune/prepare_dataset.py --dataset-path $DATA/<TASK_ID>

# 2. 학습 (기본값은 GPU 1장, 배치 4, 20 step 스모크)
NGPU=2 BS=64 MAXSTEPS=20000 SAVESTEPS=2000 DS=$DATA/<TASK_ID> OUT=$CKPT/<RUN> bash franka_teleop_finetune/train.sh
```

- state 7차원: EE 위치 3 · EE 회전(euler) 3 · 그리퍼 1. action 7차원: EE 속도 명령 6(−1~1) · 그리퍼 목표 위치 1.
  액션은 `RobotEnv` 가 그대로 받는 값이라 `action_configs` 를 비워 변환 없이 쓴다. 16스텝 청크 = 10Hz 에서 1.6초.
- 카메라: `wrist_view`(ZED-M 11845904 left), `exo_view`(ZED 2i 37149196 left). 다르면 `prepare_dataset.py --wrist-key/--exo-key`.
- 환경 변수를 안 주면 `launch.py` 는 `launch_finetune.py` 와 똑같이 동작한다.
- RTX 5090(32GB) 1장은 배치 4 에서도 `adamw_torch` 로 첫 step 에 OOM 이 난다. 파이프라인만 확인하려면
  `pip install bitsandbytes` 후 `OPTIM=paged_adamw_8bit` 로 돌린다. 이 옵션은 스모크용이고 deepspeed 와 같이 쓰지 않는다.
- 이미지는 데이터로더 워커가 샤드 단위로 전 프레임을 처리한다(디코딩 → 짧은 변 256 리사이즈 → 95% 크롭).
  RAM 이 작은 머신에서는 `WORKERS` 를 줄인다.

검증 범위: 1 에피소드(728 프레임) 데이터로 RTX 5090, 배치 4, `paged_adamw_8bit` 20 step 스모크가 끝까지 돌고 체크포인트가 저장됐다
(GPU 메모리 최대 약 21GB). 기본값 `adamw_torch` 나 멀티 GPU/deepspeed 로는 이 데이터를 돌려 보지 않았고, 학습 성능도 확인하지 않았다.

## 원본 대비 바뀐 것

- 추가: `robocasa_finetune/`, `franka_teleop_finetune/` 전체.
- 삭제: git-lfs 로 관리되던 `media/*.gif` 등 그림과 `demo_data/` (이 레포에는 LFS 객체가 없어 포인터 파일만 남기 때문).
  원본 README 의 그림 일부가 안 보이는 이유다. 필요하면 원본 레포에서 받는다.
- 이동: `README.md` → `README_ISAAC_GR00T.md`.

첫 커밋(`1c0fcde`)이 원본 n1.6.1-release(5dc80c4)의 파일 그대로다. `git diff 1c0fcde HEAD` 로 원본과의 차이를 볼 수 있다.

## 검증 범위

- 학습·평가 스크립트는 위 결과(68.8%)를 낸 스크립트에서 서버 고유 경로만 걷어 낸 것이다. 이 레포 그대로
  기존 env 에서 스모크를 돌려 확인했다: 학습 5 step(GPU 1장, 배치 4) 정상 종료, 평가 CloseDrawer 1 ep 성공.
- `requirements_*.txt` 는 결과를 낸 env 에서 뽑은 버전이다. 새 머신에서 처음부터 env 를 만들어 끝까지 돌리는 것은
  아직 확인하지 않았다.

## 라이선스

GR00T 코드와 `robocasa_finetune/eval/wrappers/` 는 NVIDIA 의 Apache-2.0 ([LICENSE](LICENSE)).
