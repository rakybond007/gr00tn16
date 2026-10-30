"""rlwrld-franka-teleop 로 변환한 LeRobot v2.1 데이터셋에 GR00T 용 meta/modality.json 을 추가한다.

    python franka_teleop_finetune/prepare_dataset.py --dataset-path /path/to/lerobot/<TASK_ID>

GR00T LeRobot 은 표준 LeRobot v2 에 meta/modality.json 하나만 더 요구한다
(getting_started/data_preparation.md). 여기서 state/action 배열을 의미 단위로 자르고,
카메라 이름을 wrist_view / exo_view 로 붙이고, 언어 지시문은 task_index 를 그대로 쓴다.
키 이름은 modality_config.py 와 맞춰야 한다.

stats.json / relative_stats.json 은 학습 시작 때 GR00T 가 알아서 계산한다.
"""
import argparse
import json
from pathlib import Path

# 카메라 시리얼 (rlwrld-franka-teleop droid/misc/parameters.py 기준). 왼쪽 렌즈만 쓴다.
DEFAULT_VIDEO = {
    "wrist_view": "observation.images.11845904_left",  # ZED-M (손목)
    "exo_view": "observation.images.37149196_left",  # ZED 2i (외부)
}

MODALITY = {
    "state": {
        "eef_position": {"start": 0, "end": 3},
        "eef_rotation": {"start": 3, "end": 6},
        "gripper_position": {"start": 6, "end": 7},
    },
    "action": {
        "eef_linear_velocity": {"start": 0, "end": 3},
        "eef_angular_velocity": {"start": 3, "end": 6},
        "gripper_position": {"start": 6, "end": 7},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--wrist-key", default=DEFAULT_VIDEO["wrist_view"])
    p.add_argument("--exo-key", default=DEFAULT_VIDEO["exo_view"])
    p.add_argument("--overwrite", action="store_true", help="이미 있는 modality.json 을 덮어쓴다")
    args = p.parse_args()

    meta = args.dataset_path / "meta"
    info = json.loads((meta / "info.json").read_text())
    features = info["features"]

    # 변환 결과가 이 설정과 맞는지 확인 (eef 모드 = state/action 7차원)
    for key in ("observation.state", "action"):
        shape = features[key]["shape"]
        assert shape == [7], f"{key} shape {shape} != [7] — convert_h5_to_v2.py --action-type eef 로 변환했는지 확인"
    video = {"wrist_view": args.wrist_key, "exo_view": args.exo_key}
    for name, key in video.items():
        assert key in features, f"{name}: {key} 가 데이터셋에 없음. 있는 카메라: " + ", ".join(
            k for k in features if k.startswith("observation.images.")
        )

    out = meta / "modality.json"
    if out.exists() and not args.overwrite:
        print(f"[skip] {out} 이미 있음 (--overwrite 로 덮어쓰기)")
        return
    modality = {**MODALITY, "video": {name: {"original_key": key} for name, key in video.items()}}
    out.write_text(json.dumps(modality, indent=2) + "\n")
    print(f"[ok] {out}")
    print(f"     episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")


if __name__ == "__main__":
    main()
