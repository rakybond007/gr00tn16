"""Franka 텔레옵(rlwrld-franka-teleop) 데이터 모달리티 설정 (n1.6 --modality-config-path 용).

임포트되면 스스로 등록한다 — launch_finetune 이 모듈을 임포트만 하기 때문이다.

키 이름은 prepare_dataset.py 가 쓰는 meta/modality.json 과 같다.
데이터는 rlwrld-franka-teleop 의 convert_h5_to_v2.py --action-type eef 로 변환한 LeRobot v2.1 이다.

  state  7차원:  0:3 EE 위치(xyz, m) · 3:6 EE 회전(euler rpy, rad) · 6:7 그리퍼 위치(0=열림, 1=닫힘)
  action 7차원:  0:3 EE 선속도 명령 · 3:6 EE 각속도 명령 (둘 다 −1~1 로 정규화된 cartesian_velocity)
                 · 6:7 그리퍼 목표 위치
  video:         wrist_view (ZED-M 11845904 left) · exo_view (ZED 2i 37149196 left)

action_configs 는 일부러 비워 둔다. 비우면 전 키가 ABSOLUTE / NON_EEF / DEFAULT 로
채워지는데, 이는 액션 값을 변환 없이 그대로 쓴다는 뜻이다. 우리 액션은 RobotEnv 가
그대로 받는 속도 명령(cartesian_velocity)이라 이게 맞다. RELATIVE 로 두면 현재 상태 기준
델타로 다시 계산하려 들어 이중 변환이 된다 (robocasa_finetune 과 같은 이유).
"""
import n15_image_transforms  # noqa: F401  임포트하면 이미지 변환이 N1.5 방식으로 교체된다
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig

franka_teleop_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist_view", "exo_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "eef_position",
            "eef_rotation",
            "gripper_position",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),          # 청크 지평 16 (10Hz 수집 → 1.6초)
        modality_keys=[
            "eef_linear_velocity",
            "eef_angular_velocity",
            "gripper_position",
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(franka_teleop_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
