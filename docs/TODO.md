# TODO

[`ROADMAP.md`](ROADMAP.md)의 Phase 구조를 그대로 따라간다. **Phase 단위 체크박스 + 하위 task 체크박스**.
세부 설계는 [`plans/<주제>.md`](plans/), 막힌 이슈는 [`ISSUES.md`](ISSUES.md).

> **표기**: ✅ 완료 · 🔄 진행 중 · ⬜ 예정
> **규칙**
> - Phase 하위 task가 전부 완료되면 Phase 표시도 ✅
> - Phase 목록/순서가 바뀌면 ROADMAP과 **동시에** 업데이트

---

## Phase 0 — 설계 결정

- ✅ **Phase 0** 설계 결정
  - ✅ D0-1 박스 포맷 = `cxcywh` normalized, state = `[cx, cy, log_w, log_h]`
  - ✅ D0-2 로딩 = `pycocotools` + `torchvision.VOCDetection`
  - ✅ D0-3 Augmentation = resize / flip / normalize 최소 구성

---

## Phase 1 — `dataset/`

- ✅ **Phase 1** `dataset/`
  - ✅ D1 `box_ops.py` — 포맷 변환, state space, 유틸 (`box_iou` 등)
  - ✅ D2 `transforms.py` — `ResizeShortestEdge`, `RandomHorizontalFlip`, `Normalize`, `Compose`
  - ✅ D3 `voc.py` / `coco.py` — detection dataset 래퍼
  - ✅ D4 `collate.py` — `detection_collate_fn`, `build_dataloader`
  - ✅ D5 `__init__.py` — `build_dataset`, `build_dataloader`

---

## Phase 2 — `model/`

- ✅ **Phase 2** `model/`
  - ✅ M1 `modules.py` — SinusoidalEmbedding, BoxEmbedding, AdaLN, MLP, 2D RoPE
  - ✅ M2 `backbone.py` — FPNBackbone, DINOv2Backbone (vits14~vitg14)
  - ✅ M3 `dit.py` — MultiHeadAttentionRoPE, DiTBlock, FlowDiT
  - ✅ M4 `head.py` — BoxHead
  - ✅ M5 `trajectory.py` — LinearTrajectory, RiemannianTrajectory
  - ✅ M6 `loss.py` — FlowMatchingLoss
  - ✅ M7 `flow_matching.py` — RiemannianFlowDet (`forward_train` / `forward_inference`)
  - ✅ M8 `__init__.py` — `build_model(config)`
  - ✅ M9~M11 문서 — `model/CLAUDE.md`, `docs/model_plan.md`, `docs/diagrams/RiemannianFlowDet.md`

---

## Phase 2.5 — Toy Example Dataset

- ✅ **Phase 2.5** MNIST Box dataset ([plan](plans/mnist_box_plan.md))
  - ✅ MB1 `dataset/mnist_box.py` — 클래스 + cv2 기반 `__main__`
  - ✅ MB2 `dataset/__init__.py` — `build_dataset`에 `mnist_box` 분기 + docstring
  - ✅ MB3 smoke test — `python -m dataset` batch shape / collate 호환 / 1-to-1 검증
  - ✅ MB4 `configs/mnist_box.yaml` — 데이터셋/모델/학습 기본 하이퍼파라미터 (PyYAML 추가)
  - ✅ MB5 single-image overfit — `script/overfit_mnist_box.py`, loss 4.63→0.026, max_err 0.11 / `outputs/mb5_overfit/`
  - ✅ (bonus) FlowDiT에 learnable query positional embedding 추가 — class-indexed 매칭 버그 해결 ([ISSUES](ISSUES.md))
  - ✅ (bonus) e0 ablation 실험 설계 — variants A/B/C/D, winner=B (5000 step + cosine), max_err 34→22px [`experiments/e0_mb5_overfit/report.md`](../experiments/e0_mb5_overfit/report.md)
  - ✅ (bonus) `trajectory.init_noise()` 훅 — Riemannian vs Euclidean 공정 비교 (ODE 50: Rm tail 0.026/max_err 10px vs Eu tail 0.41/max_err 196px). README에 GIF 임베드, report에 모델 다이어그램 + 공간 전이 분석 섹션 추가.

---

