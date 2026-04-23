# Issues

해결되지 않은 블로커, 디자인 dilemma, 재현 안 되는 버그를 기록. Claude는 막혔을 때만 읽는다.

> **새 이슈 발생 시**: `Open` 섹션에 템플릿을 채워 추가. 해결되면 `Resolved`로 이동하고 **해결 방식**과 **원인**을 남긴다.

---

## 템플릿

```markdown
### [제목 — 한 줄 요약]
- **Date**: YYYY-MM-DD
- **Area**: dataset / model / train / eval / infra
- **Files**: `path/to/file.py`, ...
- **Severity**: blocker / major / minor
- **Symptom**: 관찰된 현상 (error message, wrong output, flaky test 등)
- **Repro**: 재현 방법 (command / input)
- **Tried**: 시도한 해결책과 결과
- **Hypothesis**: 의심되는 원인
- **Status**: open / investigating / resolved
```

---

## 🔴 Open

### e2 single-seed run 결과가 **variance 극심** — "winner" 단정 불가 (특히 arb_prior)
- **Date**: 2026-04-23
- **Area**: train / eval
- **Files**: `script/overfit_mnist_box.py`, `experiments/e2_arbitrary_euclidean_prior/*`
- **Severity**: major (e2 report 결론을 뒤집음)
- **Symptom**: 같은 config·같은 seed 로 실행해도 결과가 크게 흔들림.
  - `run.sh` 4-variant 실행: `riemannian_arb_prior` mean_err **2.6 px** (최고).
  - `trajectory_gif.py` (같은 scripts, 다른 process, 같은 seed): `Rm | arb prior` panel 에서 박스 10개 모두 **캔버스 중앙에 클러스터** — digit 위치를 못 찾음.
  - `analyze_per_dim_err.py` (또 다른 reproduction): `riemannian_arb_prior` pos L2 mean **86 px** (run.sh 대비 30× 악화).
  - `euclidean` baseline 도 run 사이 tail loss 0.078 ↔ 0.96 로 **10× variance**.
- **User observation (GIF 판독)**: e2 의 GIF 에서 4 panel 중 `Rm | state prior` 만 digit 위치를 찾고, `Eu | state prior` 는 절반만 맞추며, `Rm | arb prior` / `Eu | arb prior` 는 **박스가 전부 중앙에 뭉친 채로 끝남**. 수치 report 의 "arb_prior 가 이긴다" 와 정면 충돌.
- **Root cause (가설)**:
  1. **B=1, N=1 overfit 은 매우 sensitive** — uniform `t ~ U[0,1]` 샘플링 noise 가 step 단위 loss variance 를 그대로 누적.
  2. **cuDNN non-determinism**: `torch.manual_seed` 만으로는 GPU 커널 결과 고정 안 됨. `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` 부재.
  3. **Model 초기값 민감도**: 5000 step · 1-image overfit 은 basin-of-attraction 이 작을 때 일부 seed 에서 local minima 로 빠짐.
- **Fix**: **모든 결론은 최소 3 seed mean±std 로 보고**. 구현:
  - `experiments/e2_arbitrary_euclidean_prior/run_multiseed.sh` — 4 variant × 3 seed = 12 run (각각 독립 python process 로 RNG 고립).
  - `aggregate_multiseed.py` — `report.json` 에서 `tail100_loss / mean_err_px / max_err_px` mean±std 집계.
  - e2 report 의 "winner" 섹션 제거, mean±std 표로 교체.
- **Verification (완료)**: 3 seed × 4 variant multi-seed sweep 실행 → aggregate:

  | variant | n | tail100_loss mean±std | mean_err_px mean±std | max_err_px mean±std |
  |---|---|---|---|---|
  | **riemannian** | 3 | **0.026 ± 0.004** | **6.56 ± 4.03** | 34.47 ± 30.04 |
  | euclidean | 3 | 1.13 ± 0.84 | 6.77 ± 2.70 | 50.40 ± 32.05 |
  | riemannian_arb_prior | 3 | 0.081 ± 0.076 | 23.27 ± 14.65 | 107.10 ± 78.23 |
  | euclidean_arb_prior | 3 | 0.257 ± 0.094 | 38.77 ± 6.24 | 149.71 ± 54.15 |

  per-seed 상세 (mean_err_px): `riemannian_arb_prior` = [35.1, 32.1, **2.6**] — 13.4× range. 기존 "2.6 px winner" 는 seed=0 lucky run. Multi-seed 기준 winner 는 `riemannian` baseline.

- **방향성 fix**: e2 report §4.2a 에 multi-seed 표 추가, §6 결론 재작성 (lucky single-seed 결론 철회). Runner 도구: `run_multiseed.sh` + `aggregate_multiseed.py`. Phase 3 `train.py` 는 seed 3+ 로 default 실행 예정.
- **Yellow box 의심 해소**: GIF 에서 각 digit 에 보이는 노란 박스는 `script/trajectory_gif.py::make_frame` 의 **viz-only overlay** (`cv2.rectangle(canvas, ..., _GT_COLOR, 1)` — mnist_bgr 를 display canvas 에 붙인 **후** GT box 를 위에 그림). `dataset/mnist_box.py:107` 에서 학습 tensor 는 digit 픽셀만 paste 된 **깨끗한 256×256 흑백**. 증거: `outputs/e2_arbitrary_euclidean_prior/raw_training_image.png` — 실제 학습 이미지 (노란 박스 없음).
- **Status**: investigating → resolved (이 이슈 fix 완료 후 `✅ Resolved` 로 이동).

### Riemannian 결과의 **position 오차 >> size 오차** (log-scale artifact)
- **Date**: 2026-04-22
- **Area**: model / loss
- **Files**: `model/loss.py`, `model/trajectory.py`, `model/flow_matching.py`
- **Severity**: minor (결과는 여전히 good, 그러나 localization 품질 한계)
- **Symptom**: e2 의 `riemannian` 결과 시각(`overfit_gt_vs_pred.png`, `docs/assets/e2_trajectory_compare.gif`)에서 **박스 크기(w, h)는 GT에 타이트하게 맞으나 중심 위치(cx, cy)가 수 픽셀 어긋남**. per-dim 측정:

  | variant | cx mean (px) | cy mean (px) | w mean (px) | h mean (px) | **pos L2 / size L2** |
  |---|---|---|---|---|---|
  | riemannian           | 2.66 | 5.79 | 0.89 | 1.12 | **4.37×** |
  | riemannian_arb_prior | 67.6* | 46.9* | 11.4* | 16.7* | **4.02×** |

  *arb_prior 절댓값은 해당 run 에서 variance 로 튄 값이나 **비율** 패턴은 일관 (run.sh 측정에서는 절댓값 훨씬 작음).

- **Root cause (가설)**: state space `[cx, cy, log_w, log_h]` 에서 MSE loss 는 **4 dim 균등 가중**. 하지만 cxcywh 로 변환 시 **size 차원은 `w = exp(log_w)` 의 Jacobian ≈ w (≈ 0.14 for MNIST Box)** 로 축소되는 반면 **position 차원은 identity 변환**.

  결과적으로 **같은 state-space 오차** `ε` 가 cxcywh 공간에서:
  - `err_cx ≈ ε`  (변환 identity)
  - `err_w  ≈ w · ε ≈ 0.14 · ε`  (Jacobian 축소)

  → 예상 비율 `err_cx / err_w ≈ 1 / w ≈ 7×`. 측정값 ~4× 는 이론값과 같은 order, 차이는 per-dim 수렴 속도 (w/h dim 이 더 큰 target magnitude 라 먼저 수렴).

- **Why it's a real issue**: Flow matching 은 state-space loss 를 optimize 하지만, downstream detection mAP 는 **cxcywh/pixel-space IoU** 로 평가. log-scale representation 은 size 에 유리하고 position 에 불리한 **암묵적 weighting** 을 introduces. 이는 Riemannian 방법의 **설계 tradeoff** — Phase 4 VOC/COCO 에서 mAP 상 손해 가능.

- **Tried**: 없음 (이번에 발견). Target field 분석 (`e2/analyze_target_field.py`) 에서 `||u_t||` per-dim 분포는 cx/cy (p99 2.9) vs log_w/log_h (p99 4.55) 로 이미 비대칭 — 이론적 근거.

- **Mitigation 후보**:
  1. **Loss re-weighting**: state-space MSE 에 `diag(1, 1, 1/w_t², 1/h_t²)` 혹은 간단히 position dim 에 boosted weight (예: 5×). `b_t_cx = state_to_cxcywh(b_t)` 참조해 dynamic.
  2. **Auxiliary cxcywh loss**: DETR 관행 따라 `L1 + GIoU(cxcywh)` 를 state MSE 에 섞는다. Phase 3 train.py 구현 시 기본 채택 후보.
  3. **Pixel-space target inference refinement**: 마지막 K ODE step 에서 state 대신 cxcywh 공간에서 correction. 복잡.
  4. **보지 않음**: state 자체를 `[cx, cy, w, h]` 로 되돌리기 — ℝ² × ℝ₊² geometry 의 이점(e1/e2 공통 결론) 상실.

- **Status**: open — Phase 3 train.py 에 auxiliary L1/GIoU loss 를 기본 설계로 포함해 mAP 기준 재검증 예정. 현재 Phase 2.5 toy 에서는 size 기준 성능이 우수하므로 blocker 아님.

---

## 🟡 Investigating

_(없음)_

---

## ✅ Resolved

### EuclideanTrajectory train/infer prior 불일치 → 공정 비교 불가
- **Date**: 2026-04-22 (open) → 2026-04-22 (resolved)
- **Area**: model
- **Files**: `model/trajectory.py`, `model/flow_matching.py`, `script/trajectory_gif.py`
- **Severity**: major (Phase 4 E1 ablation 전에 해결 필요)
- **Symptom**: Euclidean trajectory가 overfit 실패 — 학습 loss는 낮아져도 추론 궤적이 GT에 수렴 안 함.
- **Root cause**: `EuclideanTrajectory.sample`은 `b₀ ~ U([0.05, 0.95])` in cxcywh로 학습. 그러나 `RiemannianFlowDet.forward_inference`는 trajectory 구분 없이 `torch.randn` in state space로 `b`를 초기화 → Euclidean 추론은 학습에서 본 적 없는 분포로 시작.
- **Fix**: `RiemannianTrajectory`/`EuclideanTrajectory`에 `init_noise(B, Q, device)` 메서드 추가. `forward_inference`가 이 훅을 호출하게 수정.
- **Verification**: 공정 비교 (5000 step, cosine, ODE 50, 1-image overfit):
  - Riemannian: tail₁₀₀ 0.026, max_err **10 px**
  - Euclidean:  tail₁₀₀ 0.412, max_err **196 px**
  - Euclidean은 공정 비교에서도 Riemannian 대비 대폭 뒤처짐 — time-dependent 벡터장의 본질적 어려움으로 확인. Riemannian 우위는 bug가 아닌 **이론적 차이**.
- **Follow-up**: `experiments/e0_mb5_overfit/report.md` section 9 참고.

### FlowDiT에 per-query positional embedding 부재 → class-indexed 1-to-1 매칭 불가
- **Date**: 2026-04-22 (open) → 2026-04-22 (resolved)
- **Area**: model
- **Files**: `model/dit.py`, `model/flow_matching.py`
- **Severity**: major
- **Symptom**: MNIST Box 1장 overfit run에서 loss 진동 (~0.5, 2000 step), `max_box_err ≈ 0.6`.
- **Root cause**: FlowDiT의 box token이 `BoxEmbedding(b_t)`만 사용 → 추론 시 `b_0 ~ N(0,I)` 대칭이라 query들이 서로 구분되지 않음.
- **Fix**: `FlowDiT`에 `nn.Embedding(num_queries, dim)` 추가, `box_tokens = BoxEmbedding(b_t) + query_embed(arange(N))`. `RiemannianFlowDet`에서 `num_queries`를 FlowDiT로 전달.
- **Verification**: MB5 재실행 → loss 4.63 → 0.026, `mean_box_err` 0.17 → 0.022, `max_box_err` 0.62 → 0.11. GT vs pred 시각적으로 모든 digit 매칭.
