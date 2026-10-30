"""gr00t/eval/run_gr00t_server.py 를 N1.5 이미지 전처리로 실행한다. CLI 인자는 원본과 같다.

이 레포로 학습한 체크포인트는 반드시 이 스크립트로 띄운다 (n15_image_transforms.py 주의 참고).
입력 영상은 학습과 같은 원본 비율로 보낸다.
"""
import runpy
from pathlib import Path

import n15_image_transforms  # noqa: F401  임포트하면 변환이 교체된다

if __name__ == "__main__":
    server = Path(n15_image_transforms.processing.__file__).parents[2] / "eval" / "run_gr00t_server.py"
    runpy.run_path(str(server), run_name="__main__")
