"""gr00t/experiment/launch_finetune.py 를 그대로 실행하되, 필요하면 메모리 관련 설정만 덮어쓴다.

환경 변수를 안 주면 launch_finetune.py 와 똑같이 동작한다 (실제 학습은 이대로 큰 서버에서 돌린다).

launch_finetune.py 는 옵티마이저를 adamw_torch 로 고정하고, GPU 1장이면 deepspeed 를 쓰지 않아서
학습 파라미터(DiT ≈1.1B + LLM 상위 4층, fp32)의 Adam 상태가 전부 GPU 에 올라간다 → RTX 5090(32GB)에서 OOM.
작은 GPU 에서 파이프라인만 확인할 때 OPTIM=paged_adamw_8bit 로 돌린다 (bitsandbytes 필요, deepspeed 와는 같이 쓰지 말 것).
gr00t/ 코드를 바꾸지 않기 위해, run() 직전에 config 를 고쳐 끼우는 방식으로 감싼다.
CLI 인자는 launch_finetune.py 와 같고, 덮어쓸 값은 환경 변수로 받는다:

  OPTIM                 옵티마이저 (기본 adamw_torch = 원래 값)
  TUNE_TOP_LLM_LAYERS   학습할 LLM 상위 층 수 (기본 4 = 원래 값. 0 이면 LLM 전부 고정)
  GRAD_CKPT             1 이면 gradient checkpointing (기본 0)
"""
import os
import runpy
from pathlib import Path

import gr00t.experiment.experiment as experiment

_orig_run = experiment.run


def _run_with_overrides(config):
    config.training.optim = os.environ.get("OPTIM", "adamw_torch")
    config.model.tune_top_llm_layers = int(os.environ.get("TUNE_TOP_LLM_LAYERS", "4"))
    config.training.gradient_checkpointing = os.environ.get("GRAD_CKPT", "0") == "1"
    print(
        f"[franka_teleop] optim={config.training.optim} "
        f"tune_top_llm_layers={config.model.tune_top_llm_layers} "
        f"gradient_checkpointing={config.training.gradient_checkpointing}"
    )
    return _orig_run(config)


experiment.run = _run_with_overrides

if __name__ == "__main__":
    launch = Path(experiment.__file__).with_name("launch_finetune.py")
    runpy.run_path(str(launch), run_name="__main__")
