# e3 — Mini-batch OT Coupling Verification

> **Goal**: e2 가 드러낸 독립 coupling 결함 (prior/target support overlap → marginal field 붕괴, 13× variance) 가 **mini-batch OT** 로 사라지는지 검증.

---

## 1. 목표

한 문장: "prior 만 바꾸면 inference 가 망한다" 는 구조적 결함이 OT coupling 으로 해결됨을 증명.

- 가설 H: `OTCoupledTrajectory` (Hungarian per-image (b₀, b₁) 매칭) 를 켜면
  - `*_arb_prior` 의 mean_err variance 가 `riemannian` baseline 수준 (3-4 px std) 으로 떨어진다.
  - 평균 mean_err 도 baseline 수준 혹은 그 이하로 회복.
  - `riemannian` baseline 은 거의 변화 없음 (이미 support 분리라 OT 효과 미미).

---

## 2. 설계

- e2 와 **완전 동일 config** (데이터 1 image, 5000 step, cosine lr, ODE 50) — `ot_coupling: true` 만 추가.
- 4 variant × 3 seed (0,1,2) = 12 run, 각각 별도 python process.
- `OTCoupledTrajectory` 는 `model/trajectory.py` 에 추가, base trajectory 를 wrap.
- Cost: L2 in base 의 native space (state for Riemannian, cxcywh for Linear).
- Hungarian (`scipy.optimize.linear_sum_assignment`), per-image `Q=10`, 매 step 호출.

구현: `model/trajectory.py::OTCoupledTrajectory`, 설계: [`docs/plans/ot_coupling_plan.md`](../../docs/plans/ot_coupling_plan.md).

---

## 3. 공통 학습 설정

| 항목 | 값 |
|---|---|
| dataset | MNIST Box · 1 sample · 10 digit · 14~56 px · non-overlap |
| 모델 | FPN · hidden 192 · depth 4 · 6 head · 10 query |
| optim | AdamW · lr 3e-4 · cosine → 1.5e-5 · grad clip 1.0 |
| 학습 | 5000 step, batch 1 |
| 추론 | Euler ODE 50 step |
| **변동** | `ot_coupling: true` (e2 는 false) |

실행: `bash experiments/e3_ot_coupling/run_multiseed.sh`

---

## 4. 결과 (3 seed × 4 variant)

Raw aggregate: `outputs/e3_ot_coupling/multiseed/aggregate.md`

| variant | n | tail100_loss mean±std | mean_err_px mean±std | max_err_px mean±std | final_loss mean±std |
| --- | --- | --- | --- | --- | --- |
| riemannian_ot | 3 | 0.032 ± 0.002 | **16.14 ± 10.20** | 103.38 ± 83.32 | 0.037 ± 0.034 |
| euclidean_ot | 3 | 1.730 ± 1.747 | **29.06 ± 6.36** | 108.90 ± 36.44 | 0.022 ± 0.019 |
| riemannian_arb_prior_ot | 3 | 0.087 ± 0.056 | **32.11 ± 13.36** | 161.02 ± 46.58 | 0.066 ± 0.043 |
| euclidean_arb_prior_ot | 3 | 0.237 ± 0.037 | **39.31 ± 5.89** | 152.60 ± 56.61 | 0.130 ± 0.116 |

비교 (e2 vs e3, mean_err_px mean ± std):

| variant | e2 (no OT) | e3 (OT) | Δ mean | Δ std |
|---|---|---|---|---|
| riemannian | 6.56 ± 4.03 | **16.14 ± 10.20** | +9.58 (❌ 악화) | +6.17 (❌ 악화) |
| euclidean | 6.77 ± 2.70 | **29.06 ± 6.36** | +22.29 (❌ 악화) | +3.66 (❌ 악화) |
| riemannian_arb_prior | 23.27 ± 14.65 | **32.11 ± 13.36** | +8.84 (❌ 악화) | −1.29 (= 유사) |
| euclidean_arb_prior | 38.77 ± 6.24 | **39.31 ± 5.89** | +0.54 (= 유사) | −0.35 (= 유사) |

---

## 5. 관찰

1. **가설 H 기각** — 모든 arb_prior variant 에서 OT coupling 이 variance 를 baseline 수준으로 복원하지 **못함**.
   - `riemannian_arb_prior`: std 14.65 → 13.36 (감소 없음). mean 은 오히려 악화.
   - `euclidean_arb_prior`: mean/std 모두 거의 불변 — OT 가 intercept 하지 못함.

2. **baseline (support 분리) 은 OT 로 대폭 악화**.
   - `riemannian`: 6.56 → 16.14 (2.5×), std 4.03 → 10.20 (2.5×).
   - `euclidean`: 6.77 → 29.06 (4.3×).
   - Support 가 이미 분리된 경우 OT 는 **훈련 분포를 매 step Hungarian-matched b₀ 좁은 경로로 축소**. 추론 때 random b₀ 는 학습분포 밖이라 외삽 실패.

3. **B=1 · Q=10 setting 에서 OT 의 유효성 제한**.
   - 구현은 **per-image Q×Q Hungarian** — 즉 한 이미지의 query 10 개를 GT 10 개에 1:1 재배치. batch 가 1 이라 batch-level OT 는 trivial.
   - Per-step 매 iteration 마다 random b₀ 샘플과 고정 b₁ 사이에 Hungarian 을 돌리면 **매 step 학습 target 이 달라지는 moving target** 이 되어 학습 수렴이 어려워질 수 있음.

4. **arb_prior 의 근본 원인은 단순 coupling reassignment 로 해결 안 됨**.
   - X-crossing trajectory 가 제거되어도, inference 에서 random (non-matched) b₀ 를 받으면 OT 로 좁혀진 학습 manifold 밖에서 예측해야 함.
   - 즉 "marginal field 붕괴" 가설 자체는 맞더라도, 그 해법이 per-query OT 만으로 충분하지 않음.

---

## 6. 결론

- **H 기각**: OT coupling 만으로 arb_prior 의 ill-posedness 가 해결되지 않음 (이 1-image toy 세팅에서). 단지 OT 를 켜는 것이 **자동 fix 가 아니다**.
- **OT 의 해악 측면 확인**: support 분리된 baseline (riemannian/euclidean) 에서 OT 는 명백히 해롭다 (mean 2.5-4×, std 2.5× 악화).
- **일반화**: 이 결과는 "OT coupling 은 toy batch-of-1 에서는 해보다 실이 큼. Phase 3 에서 **실제 batch size (예: 4-16)** 과 함께 재검증 필요". Phase 2.5 결론에 영향 없음 (`riemannian` baseline 유지).
- **후속 실험 가설 후보** (OT 대체/보완):
  - `μ = -2.0` 로 arb_prior 를 target 밖으로 밀어 support 분리 (V3, plan). — support overlap 가설의 대조군.
  - Loss re-weighting `diag(5, 5, 1, 1)` 로 cx/cy gradient 강화 (amplifier 2 완화).
  - Batch-level OT (B>1 + Q 많은 setting) — toy 에서 구조적으로 시험 불가.

---

## 7. 다음 단계

- **ISSUES.md 업데이트**: "OT 단독으론 unresolved. root cause 분석은 유효, 단 fix 는 추가 탐구 필요" 로 Status 갱신.
- **Phase 3 `train.py` 조심스러운 탑재**: OT coupling 을 default 로 켜지 **않는다**. 옵션으로 제공, batch size 4+ 에서 재평가.
- **V3 구현 후보** (μ=-2.0 shift): support overlap 가설의 독립 검증. 1 variant × 3 seed = 3 run.
- Sinkhorn soft matching 은 gradient 흐르지만 이 세팅의 근본 원인과 무관 — 우선순위 낮음.
