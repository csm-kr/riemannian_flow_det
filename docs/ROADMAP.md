# Roadmap

**큰 그림 — 각 단계의 목적·이유·방향.**
세부 작업 항목과 진행 상태는 [`docs/TODO.md`](TODO.md), 미해결 블로커는 [`docs/ISSUES.md`](ISSUES.md) 참조.

---

## 📍 North Star

Detection을 **endpoint regression**이 아니라 **box state space 위에서의 continuous trajectory 학습** 문제로 재정의.
박스가 `ℝ² × ℝ₊²` 구조(center translation × positive scale)를 갖는 점을 반영하여 geometry-aware vector field로 refinement한다.

→ 배경/동기: [`docs/problem_statement.md`](problem_statement.md)

---

## Phase 0 — 설계 결정 ✅

**목적**: 이후 구현이 어긋나지 않도록 박스 포맷·state space·로딩 방식·augmentation 범위를 먼저 못박음.
**이유**: 박스 표현(`cxcywh` vs state space)과 scale 변수 취급 방식이 이후 trajectory/loss 전부를 결정하기 때문.
**방향**: normalized `cxcywh` 기준 + 모델 내부 log-scale state — 전 컴포넌트가 이 규약만 따른다.

---

## Phase 1 — `dataset/` ✅

**목적**: COCO/VOC detection 파이프라인과 박스 포맷 변환 유틸을 갖춘다.

**이유**: 모델/학습 로직이 어떤 dataset이든 동일 인터페이스(normalized cxcywh + collate)로 받도록 통합해야 재사용 가능.

**방향**: Detectron2/torchvision은 로드·augmentation 도구로만 쓰고, 내부 state 변환과 collate 규약은 직접 관리.

---

## Phase 2 — `model/` ✅

**목적**: image-conditioned vector field 예측기(`RiemannianFlowDet`)를 완성.

**이유**: flow matching은 `(b_t, t, image feature) → v̂_t`를 예측하는 함수를 필요로 하며, 이 함수의 표현력이 궤적 품질 상한을 결정.

**방향**: DiT 스타일(cross-attn으로 img→box) + 2D RoPE + AdaLN(t 조건) + box/state 전용 head — backbone은 FPN/DINOv2 교체 가능하게.

---

## Phase 2.5 — Toy Example Dataset ✅

**목적**: Phase 3 학습 루프 착수 **이전에** 제어된 합성 데이터로 모델·궤적·loss가 실제로 수렴하는지 격리 검증.

**이유**: COCO/VOC은 feature 품질·scale 다양성·매칭 모호성 같은 노이즈가 많아 **궤적 학습 자체**의 성공/실패를 진단하기 어려움. 1장 overfit이 되는지조차 불투명.

**방향**: MNIST Box dataset — 256×256 canvas에 0~9 전체 10개 숫자를 scale [14, 56] 랜덤·non-overlap 배치. 각 class query가 해당 숫자로 흐르는 궤적을 시각화할 수 있는 최소 bench.

**주요 발견**: 1장 overfit이 **수렴 실패** → FlowDiT에 per-query positional embedding 부재가 원인 (query들이 대칭적이라 class-indexed 매칭 불가). DETR 표준 방식으로 `nn.Embedding(num_queries, dim)` 추가 후 loss 4.63 → 0.026 수렴 확인. Toy bench가 본연의 역할 수행.

→ 세부 설계: [`plans/mnist_box_plan.md`](plans/mnist_box_plan.md)

---

## Phase 3 — `script/` + `utils/` 🔲

**목적**: end-to-end 학습·평가 파이프라인(train.py / eval.py) + 주변 인프라(config, logger, checkpoint, metrics).

**이유**: 모델과 데이터만으로는 실험 불가능 — 재현 가능한 학습 루프와 mAP 지표가 있어야 Phase 4 ablation이 의미 있음.

**방향**: config는 `utils/config.py`로 추상화(추후 Hydra 마이그레이션 여지), 로깅은 TensorBoard 기본. 토이 bench(Phase 2.5)에서 먼저 1장 overfit → 수 백 장 → 전체로 스케일업.

---

## Phase 4 — 실험 🔲

**목적**: 제안 방법(Riemannian trajectory)의 기여도를 ablation으로 입증 + backbone/NFE 등 하이퍼파라미터의 영향을 맵핑.

**이유**: 논문 기여는 "geometry-aware trajectory가 실제로 localization을 개선한다"는 검증에 달려 있음. 단일 설정 성능보다 **차이를 만드는 요인**을 분리하는 게 핵심.

**방향**: Linear vs Riemannian (핵심), FPN vs DINOv2, RoPE on/off, depth·NFE scaling — toy → VOC → COCO 순으로 신뢰도 쌓기.

---


