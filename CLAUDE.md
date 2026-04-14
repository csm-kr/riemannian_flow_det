# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Read This First

항상 먼저 읽기:
- `docs/plan_todo.md`

model, loss, trajectory 등 내부 설계를 다룰 때만 읽기:
- `docs/problem_statement.md`

이슈가 해결되지 않을 때만 읽기:
- `docs/issues.md`

---

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
- Detectron2 — 구현 도구 (datasets, transforms, evaluators 한정, 논문 기여 아님)
- GPU: 1장, 96GB VRAM

---

## 프로젝트 구조

```
dataset/    # COCO/VOC 래퍼, box 포맷 변환, transforms, collate
model/      # backbone, DiT blocks, flow matching, trajectory, loss
script/     # train.py, eval.py, infer.py, visualize_trajectory.py
configs/    # 실험별 YAML (base / coco / voc / exp1~3 / boosting)
utils/      # config, logger, seed, checkpoint, metrics, viz
docs/       # problem_statement, plan_todo, paper_outline
outputs/    # checkpoints, logs, figures (git 미추적)
```

> 구조는 아직 확정이 아니며 실험 진행에 따라 변경될 수 있음.

---

## 박스 포맷 규약

- **기준 포맷**: `cxcywh` normalized — `[cx, cy, w, h]` ∈ (0, 1)
- **State space** (모델 내부): `[cx, cy, log_w, log_h]` — `dataset/box_ops.py`의 `cxcywh_to_state` / `state_to_cxcywh` 사용
- 파이프라인: `xyxy pixel` (로드) → `cxcywh pixel` → `normalized cxcywh` (입력) → `log-scale state` (모델 내부)

---

## 자주 사용하는 명령어

실행 명령어 전체는 `README.md` 참조.

```bash
# Docker (개발용 — 코드 마운트, 수정 즉시 반영)
docker compose up --build -d   # 처음 or Dockerfile 변경 시
docker compose up -d           # 이후 재실행
docker compose exec rflow bash  # 실행 중인 컨테이너 접속
```

---

## 코딩 규칙

**함수 docstring** — 모든 주요 함수에 필수:
```python
def forward(self, images, boxes, t):
    """
    Purpose: Predict vector field for box states at time t.
    Inputs:
        images: [B, 3, H, W], float32 — normalized
        boxes:  [B, N, 4],    float32 — normalized cxcywh
        t:      [B],          float32 — time in [0, 1]
    Outputs:
        v:      [B, N, 4],    float32 — vector field in box state space
    """
```

**모듈 테스트** — 각 파일 하단에 `if __name__ == "__main__":` 블록 포함.
최소 shape assert + sanity check. 추후 복잡해지면 `pytest` + `tests/` 디렉토리로 이동.

---

## Trigger Keywords

| 키워드 | 동작 |
|--------|------|
| `/test [파일]` | 해당 파일의 `__main__` 블록 실행으로 검증 |
| `/ship [파일]` | 테스트 통과 → `git add` → `git commit` → `git push` 순서 실행. 테스트 실패 시 중단 |
| `/done [항목ID]` | `plan_todo.md`에서 해당 항목 `[x]` 처리 후 다음 TODO 출력 |

---

## Git 관리 규약

- 작업 단위마다 `feature/한일` 브랜치 생성 후 push
- 예시:
  ```bash
  git checkout -b feature/add-voc-dataset
  # ... 작업 ...
  git add <파일>
  git commit -m "feat: VOC dataset 래퍼 구현"
  git push origin feature/add-voc-dataset
  ```
- 브랜치 네이밍: `feature/` 접두사 + 작업 내용 (영문 소문자, `-` 구분)
- main 브랜치 직접 push 금지 — PR을 통해 병합

---

## 주의사항

- `problem_statement.md`에 정의된 box state space(`ℝ² × ℝ₊²`)와 구현이 어긋나지 않도록 주의
- Detectron2 API는 편의 도구 — 논문 기여와 혼동하지 말 것
- `outputs/` 는 git 미추적. 체크포인트/로그 경로 하드코딩 금지, config에서 관리
- config 시스템 변경(ConfigArgParse → Hydra) 시 `utils/config.py`만 수정하면 되도록 추상화 유지
