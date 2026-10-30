"""이미지 전처리를 GR00T-action-quantization(N1.5) 의 droid_franka_eef_real 설정과 맞춘다.

N1.5 순서:   95% 크롭(학습 랜덤 / 평가 중앙) → 224x224 리사이즈(비율 무시) → 색상 변형(학습만)
N1.6 기본:   검은 여백 패딩 → 짧은 변 256 → 95% 크롭 → 짧은 변 256

16:9 영상을 N1.6 기본으로 처리하면 검은 여백이 화면의 약 44% 를 차지한다. 그래서 N1.6 의 변환 생성
함수를 임포트 시점에 바꿔 끼운다. 패딩이 없어지고, 크롭·색상 변형은 N1.5 처럼 모든 카메라에 같은
값으로 들어간다. gr00t/ 코드는 바꾸지 않는다.

주의: 변환은 체크포인트에 저장되지 않고 프로세서를 만들 때마다 다시 생성된다. 이 모듈을 임포트하지
않고 체크포인트를 띄우면 오류 없이 패딩이 들어간다. 추론은 반드시 serve.py 로 띄운다.

크기·크롭·색상 값은 인자로 받지 않고 여기 상수로 고정한다. 프로세서의 from_pretrained 가
shortest_image_edge 덮어쓰기를 무시해서, 설정으로 넘기면 베이스 모델의 256 이 그대로 쓰이기 때문이다.
그래서 체크포인트의 processor_config.json 에는 256 으로 적혀 있어도 실제로는 224 다.
"""
import albumentations as A
import cv2

from gr00t.model.gr00t_n1d6 import processing_gr00t_n1d6 as processing
from gr00t.model.gr00t_n1d6.image_augmentations import FractionalCenterCrop, FractionalRandomCrop

N15_IMAGE_SIZE = 224
N15_CROP_FRACTION = 0.95
N15_COLOR_JITTER = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}


def build_n15_image_transforms(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
    extra_augmentation_config=None,
):
    """build_image_transformations_albumentations 와 같은 시그니처로 N1.5 변환을 만든다. 인자는 쓰지 않는다.

    INTER_AREA 는 torchvision Resize(antialias=True) 에 가장 가까운 축소 방식이다.
    """
    size = N15_IMAGE_SIZE
    train_transform = A.ReplayCompose(
        [
            FractionalRandomCrop(crop_fraction=N15_CROP_FRACTION),
            A.Resize(height=size, width=size, interpolation=cv2.INTER_AREA),
            A.ColorJitter(**N15_COLOR_JITTER, p=1.0),
        ],
        p=1.0,
    )
    eval_transform = A.Compose(
        [
            FractionalCenterCrop(crop_fraction=N15_CROP_FRACTION),
            A.Resize(height=size, width=size, interpolation=cv2.INTER_AREA),
        ]
    )
    return train_transform, eval_transform


processing.build_image_transformations_albumentations = build_n15_image_transforms
