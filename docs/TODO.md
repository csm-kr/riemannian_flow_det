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
  - ✅ (bonus) `trajectory.init_noise()` 훅 — Riemannian vs Euclidean 공정 비교. README에 GIF 임베드, report에 모델 다이어그램 + 공간 전이 분석 섹션 추가.
  - ✅ (bonus) **e1 unified-prior fair comparison** — 두 trajectory가 동일한 `N(0,I) state prior`에서 출발(같은 seed → 같은 init box)하도록 재설계. 차이는 interpolation 공간(state vs cxcywh) 하나로 국한. Rm tail 0.021/max_err 11px vs Eu tail 0.11/max_err 27px — Riemannian 우위가 **이론적 구조 차이**(constant vs time-dependent field)에서 옴을 확증 [`experiments/e1_unified_prior_fair_compare/report.md`](../experiments/e1_unified_prior_fair_compare/report.md)
  - ✅ (bonus) **e2 2×2 ablation: prior × interp space + multi-seed 재검증** — `state N(0,I)` vs `arb clip(N(0.5,1/6²),0.02,1) cxcywh` prior × `state` vs `cxcywh` interp. **초기 single-seed 결론 (arb_prior winner) 은 철회** — 3 seed sweep 결과 `riemannian_arb_prior` 는 [35, 32, **2.6**] px 로 13× 흔들림. **Multi-seed winner = `riemannian` baseline** (mean_err 6.6 ± 4.0 px, 유일하게 low variance). state interp > cxcywh interp 는 tail loss 40× 차이로 확인. Methodological take-away: 1-image overfit 은 반드시 3+ seed 로 평가. `run_multiseed.sh` + `aggregate_multiseed.py` 인프라 도입. 추가 artefact: `RiemannianTrajectoryArbPrior`, 4-panel GIF, raw training image (노란 박스 해명). [`experiments/e2_arbitrary_euclidean_prior/report.md`](../experiments/e2_arbitrary_euclidean_prior/report.md)
  - ✅ (bonus) **e2 root cause 규명 — independent coupling ill-posedness** — arb_prior 실패가 prior 자체가 아니라 **flow matching 의 independent coupling + prior/target support overlap** 조합에서 **marginal vector field 가 0 으로 붕괴** 하기 때문임을 이론적으로 확립. baseline `riemannian` 이 잘 되는 이유도 "support 가 우연히 분리돼 있어 이 문제를 회피" 로 재해석. Fix plan: [`docs/plans/ot_coupling_plan.md`](plans/ot_coupling_plan.md) (mini-batch OT coupling, Tong 2023 / Pooladian 2023 계열) 을 Phase 3 `train.py` 에 탑재 예정. ISSUES.md 재작성 ([`docs/ISSUES.md`](ISSUES.md) "arb_prior 가 학습 안 됨" 섹션).
  - ✅ (bonus) **e3 OT coupling verification — 가설 기각** — `OTCoupledTrajectory` (per-image Q×Q Hungarian) 구현 후 4×3 multi-seed sweep. arb_prior variance 복원 실패 (riemannian_arb_prior 23→32 px, std 14.65→13.36 거의 불변). Baseline 은 오히려 2.5-4× 악화 (riemannian 6.6→16.1 px). 해석: batch=1, Q=10 setting 에서 per-image Hungarian 은 query 간 local reassignment 만 수행 → OT 로 좁혀진 학습 manifold 는 inference 의 random b₀ 분포와 mismatch. OT 단독은 정답 아님. 다음 후보: (a) μ=-2.0 shift (support 분리 독립 검증), (b) loss re-weighting [5,5,1,1], (c) batch>1 에서 OT 재평가. [`experiments/e3_ot_coupling/report.md`](../experiments/e3_ot_coupling/report.md)

---

## Phase 3 — `script/` + `utils/` 🔲

- 🔄 **Phase 3** 학습 파이프라인
  - ✅ P3-OT **mini-batch OT coupling** — `OTCoupledTrajectory` wrapper 구현 완료, e3 에서 검증 (가설 기각, toy 에서 해로움 확인). Phase 3 `train.py` 에서는 default **off** 유지, 옵션으로 제공. 설계: [`plans/ot_coupling_plan.md`](plans/ot_coupling_plan.md)
  - ⬜ P3-CFG config schema 정리 (`ot_coupling: false` default, seed 3+ 기본 실행 통합)
  - ⬜ P3-TRAIN `script/train.py` — config 기반 end-to-end 학습 루프 + TensorBoard
  - ⬜ P3-EVAL `script/eval.py` — mAP / per-dim err 자동 집계, `aggregate_multiseed.py` 재활용
  - ⬜ P3-AUX position vs size 이슈 완화 — `L1 + GIoU(cxcywh)` auxiliary loss (ISSUES.md "position 오차 >> size 오차")

---

