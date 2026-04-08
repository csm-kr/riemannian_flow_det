# Riemannian Flow for Object Detection

객체 검출에서 기하학적 연속 박스 궤적 모델링을 위한 연구 코드베이스.

## 개요

기존 검출 방식은 박스를 단일 스텝 회귀로 예측한다. 이 프로젝트는 박스 정제를 기하학적 box state space 위에서의 **연속 흐름(continuous flow)** 으로 모델링하며, flow matching으로 벡터 필드를 정의하고 학습한다.

핵심 기여:
- 기하학적 보간을 갖춘 box state space 정의
- box 궤적 학습을 위한 flow matching objective
- 벡터 필드 예측을 위한 DiT-style transformer
- 반복적 추론 정제(iterative inference refinement)

## 데이터셋

COCO, VOC

## 하드웨어

GPU 1장 · 96GB VRAM

## 구조

```
dataset/    # COCO/VOC 래퍼, box 연산, transforms
model/      # backbone, DiT blocks, flow matching, trajectory, loss
script/     # train, eval, infer, visualize, analyze
configs/    # 실험별 YAML 설정
utils/      # config, logger, seed, checkpoint, metrics
docs/       # 문제 정의, 계획, 논문 아웃라인
outputs/    # 체크포인트, 로그, 그림 (git 미추적)
```

## 실행

```bash
python script/train.py --config configs/coco.yaml
python script/eval.py  --config configs/coco.yaml --checkpoint outputs/checkpoints/best.pth
```
