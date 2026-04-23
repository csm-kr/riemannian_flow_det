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

### arb_prior 가 학습 안 됨 — **independent coupling + prior/target support overlap → marginal vector field 붕괴**
- **Date**: 2026-04-23 (open), 2026-04-23 (root cause 확정)
- **Area**: model / train (flow matching 구조적 이슈)
- **Files**: `model/trajectory.py:166-210`, `model/flow_matching.py:86-120`
- **Severity**: **major** — flow matching 이 prior 변경에 robust 하지 않다는 걸 드러냄. Phase 4 이전에 구조적 해결 필요.

- **Symptom**:
  - multi-seed sweep (3 seed × 4 variant): `riemannian_arb_prior` mean_err `[35.1, 32.1, 2.6]` px — **13.4× 흔들림**.
  - GIF (`docs/assets/e2_frame_t_1.00.png` 3·4번째 panel): 10개 query box 가 **캔버스 중앙에 뭉친 채** digit 위치로 도달 못 함.
  - `euclidean` baseline 은 tail loss variance 극심 (0.11 / 1.11 / 2.16) 이나 box 위치는 그래도 수렴.
  - `riemannian` baseline 만 안정적 (tail 0.026 ± 0.004, mean_err 6.6 ± 4.0).

- **핵심 질문 (사용자 지적)**: riemannian/euclidean 는 잘 되는데 **"b₀ prior 만 바꾸었다"** 고 inference 가 무너지는 건 단순 tuning 이 아니라 구조적 문제가 있다는 뜻. → 맞다. 이하 원인.

- **Root cause**: flow matching 의 **independent coupling** 와 arb_prior 의 **support overlap** 이 서로 곱해져 **marginal vector field 가 0 으로 붕괴**.

  네트워크가 학습하는 대상:

  `v*(x, t) = E[u_t | γ(t; b₀, b₁) = x]`

  즉 같은 (x, t) 에 도달하는 모든 `(b₀, b₁)` 쌍의 평균. `trajectory.sample()` 은 매 step `(b₀, b₁)` 을 **독립 샘플** (= independent coupling).

  **state N(0, I) prior**: `b₀.cx ∈ [-3, 3]` (이미지 밖), target `b₁.cx ∈ [0.1, 0.9]` → 두 support **거의 분리**. 임의 `x_t` 를 지나는 `(b₀, b₁)` 쌍은 대부분 "밖→안" 방향으로 일관 → `v*` 크기 유의미.

  **arb_prior**: `b₀.cx ∈ [0.02, 1]`, `b₁.cx ∈ [0.1, 0.9]` → support **거의 완전히 겹침**. 예시:
  - query i: b₀.cx=0.3, b₁.cx=0.7 → u_cx = +0.4
  - query j: b₀.cx=0.7, b₁.cx=0.3 → u_cx = -0.4
  - 두 trajectory 가 t=0.5 에서 같은 `x.cx ≈ 0.5` 통과.
  
  네트워크는 같은 `(x, t)` 에 `+0.4` 와 `-0.4` 두 값을 regression. **MSE 최적해 = 평균 = 0**. 결과:

  1. `v*(center, t) ≈ 0` → ODE 가 **중앙 부근에서 정지** → GIF 의 "박스 10 개 중앙 클러스터" 그대로.
  2. null field 에서 query 구분은 `query_embed` symmetry breaking 에 전적으로 의존 → seed 에 따라 성패 갈림 → 13× variance.
  3. 학습 가능한 신호는 w/h dim 에만 남아 "크기는 맞추는데 위치는 못 맞춤" 의 실제 관측과 부합.

  이건 **알려진 flow matching limitation** — independent coupling 의 ill-posedness. 해결 주제가 Tong et al. (2023) "Conditional Flow Matching" 과 Pooladian et al. (2023) "Multisample Flow Matching" 계열 — **mini-batch OT coupling** 으로 `(b₀, b₁)` 을 "가까운 것끼리" 짝지어 marginal field 의 평균화 효과를 제거.

- **Amplifiers (부차 요인)**:
  1. cx/cy target magnitude 붕괴 — arb_prior 의 `|u_cx|` p99 0.66 vs `|u_log_w|` p99 2.46, per-dim equal MSE → gradient 가 w/h 에 dominated. 이 이슈 단독으로는 "위치 학습 느림" 수준이지만 위의 null field 와 결합해 수렴 불가.
  2. `model/dit.py:182-184` 의 RoPE `.clamp(0, 1)` — state prior 는 t=0 에서 clamp saturated 지만 `BoxEmbedding(b_t)` 의 raw 4D 값이 크게 달라 query 구분 OK. arb_prior 는 clamp 영향 없지만 b_t span 이 좁아 RoPE/BoxEmbedding 둘 다 약한 신호 → 학습이 `query_embed` 하나에 과의존.
  3. B=1·5000 step overfit 민감도 — 위 구조적 null field 가 있으면 basin-of-attraction 이 좁아져 seed 간 variance 폭증.

- **Fix 계획** (phase 별):
  - **Phase 2.5 단기 해결 (불필요)**: arb_prior 실험은 이미 "coupling 결함의 증거" 로 역할 끝. `riemannian` baseline (support 분리) 을 winner 로 확정, arb prior 재시도는 후속 탐구 (아래 e3 결과 참고) 이후로 연기.
  - **Phase 3 구조적 해결** (권장): `docs/plans/ot_coupling_plan.md` — `trajectory.sample()` 에 mini-batch OT `(b₀, b₁)` assignment 탑재. `train.py` 에 통합. **단 e3 결과를 보면 OT 단독은 부족, 배치+loss-weight 조합 시도 필요.**
  - **Phase 4 large-scale 검증**: VOC/COCO 에서 coupling 있을 때/없을 때 mAP 비교. Prior-agnostic robustness 확인.

- **Mitigations (검토만, 채택 안 함)**:
  - Per-dim loss weight `[5, 5, 1, 1]` — amplifier 2 만 완화, 구조적 null field 는 그대로. **e3 결과 이후 재검토 — OT 단독이 실패했으니 이것도 함께 시도할 가치 있음.**
  - `μ = -2.0` 로 arb_prior 평균 이동해 support 분리 — "arb prior 의 원래 의도 (target 근접)" 를 해침. 그러나 **support overlap 가설의 독립 검증** 으로는 가치 있음 (V3).
  - OT coupling 을 정답으로 가정했으나 **e3 에서 반증됨** (아래 참조).

- **Verification (e2 multi-seed, 완료)**:
  - Multi-seed sweep 3×4: `experiments/e2_arbitrary_euclidean_prior/run_multiseed.sh` + `aggregate_multiseed.py`.

    | variant | n | tail100 mean±std | mean_err_px mean±std | max_err_px mean±std |
    |---|---|---|---|---|
    | **riemannian** | 3 | **0.026 ± 0.004** | **6.56 ± 4.03** | 34.47 ± 30.04 |
    | euclidean | 3 | 1.13 ± 0.84 | 6.77 ± 2.70 | 50.40 ± 32.05 |
    | riemannian_arb_prior | 3 | 0.081 ± 0.076 | 23.27 ± 14.65 | 107.10 ± 78.23 |
    | euclidean_arb_prior | 3 | 0.257 ± 0.094 | 38.77 ± 6.24 | 149.71 ± 54.15 |

- **e3 OT coupling verification (완료, 가설 기각)**: `experiments/e3_ot_coupling/`
  - `OTCoupledTrajectory` (per-image Q×Q Hungarian) 구현 후 4×3 multi-seed sweep.

    | variant | e2 (no OT) | e3 (OT) | Δ mean | Δ std |
    |---|---|---|---|---|
    | riemannian | 6.56 ± 4.03 | **16.14 ± 10.20** | +9.6 (❌) | +6.2 (❌) |
    | euclidean | 6.77 ± 2.70 | **29.06 ± 6.36** | +22.3 (❌) | +3.7 (❌) |
    | riemannian_arb_prior | 23.27 ± 14.65 | **32.11 ± 13.36** | +8.8 (❌) | −1.3 (=) |
    | euclidean_arb_prior | 38.77 ± 6.24 | **39.31 ± 5.89** | +0.5 (=) | −0.4 (=) |

  - **결과**: OT coupling 만으로 arb_prior variance 복원 **실패**. Baseline (support 분리) 에는 오히려 2.5-4× **악화**.
  - **해석**: (1) batch=1, Q=10 setting 에서 per-image Hungarian 은 query 간 local reassignment 만 수행 — batch-level OT 의 본래 효과 제한적. (2) OT 로 좁혀진 학습 manifold 는 inference 의 random b₀ 분포와 mismatch → 외삽 실패. (3) marginal field collapse 가설은 유효하되, 해결책이 단순 coupling 개선으로는 불충분.
  - Full report: `experiments/e3_ot_coupling/report.md`.

- **Yellow box 의심 해소**: GIF 의 노란 사각형은 `script/trajectory_gif.py::make_frame` 의 **viz-only overlay**. `dataset/mnist_box.py:107` 학습 tensor 는 digit 픽셀만 붙인 깨끗한 256×256 흑백. 증거: `docs/assets/e2_raw_training_image.png`.

- **Status**: **open — 추가 탐구 필요**. OT 단독 해법 기각. 다음 후보: (a) μ=-2.0 shift (V3) 로 support 분리 독립 검증, (b) loss re-weighting [5,5,1,1] 결합, (c) batch>1 setting 에서 OT 재평가 (Phase 3). Phase 2.5 결론 (`riemannian` winner) 은 유지.

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
