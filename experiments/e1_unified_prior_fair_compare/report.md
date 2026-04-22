# e1 — Unified-Prior Fair Trajectory Comparison

> **이전 실험(e0)의 후속**: e0의 bonus 분석에서 Riemannian vs Euclidean 공정 비교를 시도했으나, 두 trajectory가 서로 다른 prior(`randn state` vs `uniform cxcywh`)를 사용해서 **init 박스가 시각적으로 달랐다**. 성능 차이가 (a) 이론적 구조 vs (b) prior mismatch 중 어느 원인에서 오는지 분리할 수 없었음.
>
> 본 실험은 **두 trajectory가 완전히 동일한 init `b₀`를 공유**하게 만들고, 차이를 오직 **interpolation 공간**(Riemannian = state, Euclidean = cxcywh) 하나로 국한시켜 측정한다.

---

## 1. 목표

같은 입력(init box)에서 출발해 두 trajectory가 각자의 공간에서 궤적을 그릴 때, 성능 격차가 얼마나 나는지 **깨끗하게** 측정.

**가설**: Riemannian이 여전히 Euclidean 대비 유의미하게 우위 — `u_t*`가 constant인 것이 time-dependent보다 학습 신호로 유리하기 때문.

---

## 2. 설계 — "박스는 함께 만들고, 공간은 따로"

```
           b₀ ~ N(0, I) in state space         ← 공통 init (같은 seed면 완전히 동일)
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
 Riemannian:                Euclidean:
 state에서 선형보간           state → cxcywh 변환
                              cxcywh에서 선형보간
                              다시 state로
                              (DiT 입력용)
         │                        │
         └───────────┬────────────┘
                     ▼
              FlowDiT + BoxHead
              (state space 입력/출력, 공통)
                     │
                     ▼
              state → cxcywh (출력)
```

- **DiT 모듈은 둘 다 똑같이 state space `[cx, cy, log_w, log_h]`만** 입출력
- `b₀ ~ N(0, I) in state` 통일 (`trajectory.init_noise`)
- 차이는 **"중간 interpolation이 Riemannian이냐 Euclidean이냐"** 오직 하나

### 구현 요약 (`model/trajectory.py`)

**Riemannian** — state에서 직접 보간, target vector 상수
```python
b0  = randn_like(b1)                      # state
b_t = (1-t)·b0 + t·b1                     # linear in state (= geodesic in ℝ²×ℝ₊²)
u_t = b1 - b0                             # constant vector field
```

**Euclidean** — 같은 state b₀에서 cxcywh 경유 보간, target 시간의존
```python
b0     = randn_like(b1_cx)                # state (Riemannian과 동일!)
b0_cx  = state_to_cxcywh(b0)              # → cxcywh (w,h = exp(log_w,log_h))
b_t_cx = (1-t)·b0_cx + t·b1_cx            # linear in cxcywh (≠ geodesic)
b_t    = cxcywh_to_state(b_t_cx)          # DiT 입력용으로 state 복귀
u_t    = [Δcx, Δcy, (w1-w0)/w_t, (h1-h0)/h_t]   # time-dependent
```

---

## 3. 공통 학습 설정

변경되는 건 `trajectory` 필드 **하나뿐**. 나머지 모두 동일.

| 항목 | 값 |
|------|-----|
| dataset | `mnist_box` · 1 sample · 10 digits · 14~56px · non-overlap |
| 모델 | FPN backbone · hidden 192 · depth 4 · 6 heads · 10 queries (1-to-1 class) |
| optimizer | AdamW · lr 3e-4 · cosine → 1.5e-5 · grad clip 1.0 |
| 학습 | 5000 step, batch 1 |
| 추론 | Euler ODE 50 step |
| seed | 0 (양쪽 동일) |

Variant 파일: `variants/riemannian.yaml`, `variants/euclidean.yaml` (trajectory 필드만 다름)

실행: `bash experiments/e1_unified_prior_fair_compare/run.sh`

---

## 4. 결과

`bash experiments/e1_unified_prior_fair_compare/run.sh` 출력 기준 (재현 가능한 수치):

| Trajectory | tail₁₀₀ loss | mean err (norm) | max err (norm) | mean err (px) | max err (px) | wall time |
|---|---|---|---|---|---|---|
| **Riemannian** (ours) | **0.028** | **0.021** | **0.066** | **5.3** | **16.9** | ~57 s |
| Euclidean (baseline)  | 0.056 | 0.024 | 0.192 | 6.1 | **49.1** | ~60 s |

> Note: flow matching은 `t ~ U[0,1]` 샘플링 때문에 step 단위 loss가 본질적으로 noisy.
> 여러 run 사이 variance가 있어 정확한 숫자는 ±30% 오차 구간 내 (run 재현 권장).

### 관찰

1. **Mean err는 둘 다 낮음** (5.3 / 6.1 px) — 대부분의 박스는 잘 맞춤. "10개 중 대부분" 기준 성공.
2. **Max err에서 2.9배 차이** (17 vs 49 px) — worst-case 박스에서 Euclidean이 뚜렷히 뒤처짐.
3. **Tail loss 2배 차이** (0.028 vs 0.056) — 학습 수렴 자체가 Euclidean은 더 어려움.
4. **Euclidean은 variance가 더 큼** — 여러 run 비교 시 Rm은 max_err ~11~17 범위 내, Eu는 ~27~49 범위. Time-dependent field 학습의 **불안정성** 시사.
5. 이전(e0, 서로 다른 prior) 비교에서 Euclidean max err가 **196 px**였던 것과 대조 → 통일 prior에서는 49 px까지 회복. 이전 Eu의 대실패는 **prior mismatch가 주 원인**이었음.
6. 그럼에도 **Riemannian의 theoretical 우위는 실존** — prior mismatch 제거 후에도 2~3배 차이 유지.

---

## 5. 시각 결과

### 5.1 Trajectory GIF

두 경로에 동일한 GIF가 저장됨:
- **Canonical (git-tracked)**: [`docs/assets/trajectory_compare.gif`](../../docs/assets/trajectory_compare.gif) — README.md 임베드용, 최신 run 결과 고정.
- **Run 아티팩트**: `outputs/e1_unified_prior_fair_compare/gif/trajectory_compare.gif` — `run.sh` 실행 시 갱신. git 미추적.

51 frame · 12 fps · 768×768 확장 canvas side-by-side (Rm / Eu).

### 5.2 Loss curve — robustness 비교

![Loss compare](../../docs/assets/loss_compare.png)

좌: 전체 학습 궤적(log y-scale), 각 raw loss(faded) + EMA smoothed(solid).
우: tail 40% 확대. 음영은 robust 분산 지표.

**Tail 10% loss 통계** (마지막 500 step)

| variant | mean | median | **std** | p90 | **p99** |
|---|---|---|---|---|---|
| Riemannian | 0.029 | 0.013 | **0.072** | 0.061 | **0.231** |
| Euclidean  | 0.504 | 0.002 | **5.014** | 0.037 | **11.15** |

**해석** — 두 방법 모두 `t ~ U[0,1]` 샘플링으로 step 단위 loss는 noisy하지만:

1. **Riemannian은 spike가 거의 없음** — p99 0.23, std 0.07. 학습이 **전체 t 구간에서 균일하게 안정적**.
2. **Euclidean은 간헐적 거대 spike** — p99 **11.15** (Rm 대비 **48×**), std **5.01** (Rm 대비 **70×**). Median은 낮지만(0.002) **worst-case가 수십~수백 배**로 튄다.
3. 원인: Euclidean의 target `u_t ∝ 1/w_t` — 작은 박스(w_t → 0)에서 field가 발산. 특정 `t` 값에서 극단적 gradient 발생.

즉 Euclidean은 "대부분의 step은 낮은 loss지만 때때로 폭발" → **학습 안정성(robustness) 측면에서 Riemannian이 우위**. 이 차이는 최종 max_err 격차(17 vs 49 px)와 직결.

**생성**: `bash experiments/e1_unified_prior_fair_compare/run.sh` 실행 시 자동. 또는 직접:
```bash
python script/plot_loss_compare.py \
    --variants riemannian:outputs/e1_unified_prior_fair_compare/riemannian/loss_log.txt \
               euclidean:outputs/e1_unified_prior_fair_compare/euclidean/loss_log.txt \
    --out outputs/e1_unified_prior_fair_compare/loss_compare.png
```

- **t=0.00**: 좌/우 **완전히 동일한 init 박스**. 일부 박스는 이미지 밖 (state Gaussian의 자연스러운 분포).
- **t=0.5**: 양쪽 다 박스를 MNIST 영역으로 끌어옴. Riemannian이 약간 빠름.
- **t=1.0**: Riemannian은 10개 GT에 타이트하게 덮음, Euclidean은 전체적으로 근접하나 일부(주로 작은 digit)에서 size/position 어긋남.

---

## 6. 결론

- **Riemannian > Euclidean은 이론적 차이에서 온다** (prior mismatch가 아님).
- 원인: `u_t*`가 constant이면 단일 방향 벡터만 예측하면 됨. time-dependent면 t별로 다른 필드 학습 필요 → capacity 소모·numerical instability.
- Phase 4 E1 ablation(VOC/COCO)에서 이 결과가 large-scale에서도 재현되는지 확인 필요.

## 7. 다음 단계

- **E2**(추론 시 ODE step 수 영향) — 이번 실험은 ODE 50 고정. Euclidean은 time-dependent field라 step 수에 민감할 가능성.
- **E3**(학습 중 `t` 샘플링 분포) — uniform 대신 logit-normal 등으로 early/late step weight 조절.
- **다중 샘플 일반화** — num_samples 1 → 4 → 16 → 100.
