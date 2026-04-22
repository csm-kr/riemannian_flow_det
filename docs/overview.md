# Overview

## 프로젝트 개요

**Geometry-aware continuous box trajectory modeling for object detection.**

기존 detection의 one-step box regression을 넘어, box state space(`ℝ² × ℝ₊²`) 위에서 flow matching으로 연속 궤적을 학습하는 연구 코드베이스.
3개 core 실험 → boosting 순서로 개발.

---

## 기술 스택

- **Python 3.10+**, PyTorch 2.x, torchvision
- **ConfigArgParse** (추후 Hydra 마이그레이션 가능 — config 접근은 `utils/config.py`로 추상화)
- **TensorBoard** (`torch.utils.tensorboard`) — 학습 tracking
- pycocotools — COCO annotation 로드
- GPU: 1장, 96GB VRAM

---

## 프로젝트 구조

```
dataset/    # COCO/VOC 래퍼, box 포맷 변환, transforms, collate
model/      # backbone, DiT blocks, flow matching, trajectory, loss
script/     # train.py, eval.py, infer.py, visualize_trajectory.py
configs/    # 실험별 YAML (base / coco / voc / exp1~3 / boosting)
utils/      # config, logger, seed, checkpoint, metrics, viz
docs/       # problem_statement, plan_todo, paper_outline, overview, conventions
plans/      # 개별 작업/실험 플랜 문서 (주제별 .md, 하위 폴더로 그룹화)
outputs/    # checkpoints, logs, figures (git 미추적)
TODO.md     # 현재 작업 항목 체크리스트
```

> 구조는 아직 확정이 아니며 실험 진행에 따라 변경될 수 있음.

---

## Docker 실행

자세한 실행 명령은 `README.md` 참조.

```bash
docker compose up --build -d   # 처음 or Dockerfile 변경 시
docker compose up -d           # 이후 재실행
docker compose exec rflow bash  # 실행 중인 컨테이너 접속
```
