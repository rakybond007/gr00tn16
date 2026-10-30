# GR00T N1.6 — RoboCasa Kitchen 24 태스크 결과

- 데이터: `prehj/rc_mg_noisy` (HF, public). 라벨은 수집 때 실제 실행된 액션.
- 학습: 이 레포의 `robocasa_finetune/train.sh` 기본값 (NEW_EMBODIMENT, 총 배치 64 = 2 GPU × 32, 60k step), 1회.
- 평가: `robocasa_finetune/eval/run_eval.sh` 기본값 (seed 42, generative textures, 태스크당 50 ep, 실행 길이 16), `checkpoint-60000`.

## 전체

| 모델 | 성공률 (1200 ep) |
|---|---|
| GR00T N1.5 (같은 데이터·설정, N1.5 코드베이스) | 70.3% |
| **GR00T N1.6 (이 레포)** | **68.8%** (825/1200) |
| GR00T N1.7 (같은 데이터·설정, N1.7 코드베이스) | 49.6% |

N1.5·N1.6 은 사전학습에 RoboCasa(Panda 시뮬) 데이터가 들어 있고, N1.7 의 공개 사전학습 설정(`experiment_cfg/conf.yaml`)에는 없다.
세 모델 모두 1회 학습 결과다.

## 태스크별 (/50)

| 태스크 | 성공 | 태스크 | 성공 |
|---|---|---|---|
| CloseDrawer | 50 | PnPStoveToCounter | 33 |
| CloseSingleDoor | 49 | PnPCounterToCab | 30 |
| CloseDoubleDoor | 47 | PnPCounterToStove | 26 |
| TurnOffMicrowave | 48 | PnPSinkToCounter | 22 |
| CoffeePressButton | 48 | PnPCounterToMicrowave | 14 |
| CoffeeServeMug | 44 | PnPMicrowaveToCounter | 12 |
| TurnOnMicrowave | 44 | PnPCabToCounter | 12 |
| OpenSingleDoor | 44 | CoffeeSetupMug | 18 |
| OpenDoubleDoor | 42 | TurnOffStove | 12 |
| TurnOffSinkFaucet | 42 | TurnOnStove | 36 |
| TurnOnSinkFaucet | 41 | TurnSinkSpout | 38 |
| OpenDrawer | 37 | PnPCounterToSink | 36 |

## 실행 길이

같은 체크포인트, 같은 시드로 `EXEC_H=4` 를 돌리면 63.6% (1196 ep) 로, 16 (68.8%) 보다 낮다
(짝 McNemar p=5e-4).

## 학습 속도

같은 클러스터의 GPU 2장으로 60k step 에 11시간 58분. 같은 설정에서 N1.5 코드베이스는 23시간 19분, N1.7 은 13시간 2분 걸렸다.
