# e2 — 2×2 Ablation: Prior × Interp Space

> **e1 후속**. e1 은 prior 를 통일(`state N(0,I)`)해서 차이를 "interp 공간" 하나로
> 국한시켰다. 이 실험은 한 걸음 더 나아가 **prior × interp** 를 독립 축으로 분리
> 하는 2×2 실험. 특히 **arbitrary Euclidean prior**(`clip(N(0.5, 1/6²), 0.02, 1)`
> in cxcywh)를 새 prior 축으로 추가해 "prior 가 entropy 낮거나 support 이상
> 해도 interp 가 state 면 괜찮은가?" 를 정량화한다.
>
> **주요 발견 (스포일러)**: 원래 가설 "arb prior → 학습 실패" 은 **틀렸다**.
> 진짜 지배 요인은 **interp 공간** 하나이며, arb prior 는 오히려
> *state interp* 와 결합하면 baseline 보다 더 좋은 수렴을 낸다.

---

## 1. 목표

"유클리드 공간에서 임의 box prior 가 왜 나쁜가?" 를 2×2 로 분해:

|           | **state interp** (Rm)            | **cxcywh interp** (Eu)              |
|-----------|----------------------------------|-------------------------------------|
| state N(0,I) prior         | `riemannian` (e1 기준)        | `euclidean` (e1 baseline)           |
| arb cxcywh prior           | `riemannian_arb_prior` (NEW)  | `euclidean_arb_prior` (NEW)         |

- **행**을 비교 → **prior** 의 영향
- **열**을 비교 → **interp 공간** 의 영향
- 대각선 비교 → 상호작용

가설:
- (H1) prior entropy 가 중요하다면 arb prior 행의 두 cell 모두 나쁠 것
- (H2) interp 공간이 중요하다면 cxcywh interp 열의 두 cell 모두 나쁠 것
- (H3) 둘 다 중요하다면 `euclidean_arb_prior` 가 최악, `riemannian` 이 최고

---

## 2. 설계

네 trajectory 를 같은 모델·같은 GT 에 붙여 비교.

| variant | b₀ 공간 | b₀ 분포 | interp 공간 | 이론 u_t |
|---|---|---|---|---|
| `riemannian` | state | `N(0, I)` ∈ ℝ⁴ | state | `b₁−b₀` (const in x_t) |
| `euclidean` | state | `N(0, I)` → cxcywh (exp ⇒ log-normal) | cxcywh | `[Δcx, Δcy, Δw/w_t, Δh/h_t]` |
| `riemannian_arb_prior` | **cxcywh** | `clip(N(0.5, 1/6²), 0.02, 1)` | **state** | `b₁−b₀` (const, but bounded b₀) |
| `euclidean_arb_prior` | cxcywh | 동일 arb prior | cxcywh | 동일 time-dependent 형태 |

구현: [`model/trajectory.py`](../../model/trajectory.py) — `LinearTrajectoryArbPrior`, `RiemannianTrajectoryArbPrior`.

---

## 3. 공통 학습 설정

e1 동일. 변하는 것은 `trajectory` 필드 하나.

| 항목 | 값 |
|------|-----|
| dataset | `mnist_box` · 1 sample · 10 digit · 14~56 px · non-overlap |
| 모델 | FPN · hidden 192 · depth 4 · 6 head · 10 query |
| optim | AdamW · lr 3e-4 · cosine → 1.5e-5 · grad clip 1.0 |
| 학습 | 5000 step, batch 1 |
| 추론 | Euler ODE 50 step |
| seed | 0 |

실행: `bash experiments/e2_arbitrary_euclidean_prior/run.sh`

---

## 4. 결과

### 4.1 Target field 분석 (학습 전, 200 k 샘플)

**||u_t||₂** (state 공간)

| variant | mean | std | p99 | max |
|---|---|---|---|---|
| riemannian | 3.52 | 0.98 | 5.87 | 8.6 |
| euclidean  | 3.65 | **3.77** | **18.5** | **229.6** |
| riemannian_arb_prior | **1.98** | **0.50** | **3.10** | **3.9** |
| euclidean_arb_prior  | 1.99 | 1.16 | 6.53 | 15.0 |

**Conditional Lipschitz** `L̂(x_t) ≈ |u_t_{log_w}|`

| variant | p99 | max |
|---|---|---|
| riemannian | **0** | **0** |
| riemannian_arb_prior | **0** | **0** |
| euclidean | 16.80 | **229.5** |
| euclidean_arb_prior | 5.84 | 14.9 |

Key: **state interp 를 쓰면 prior 와 무관하게 Lipschitz 0** (constant field). cxcywh interp 일 때만 1/w 항이 생겨 L̂ 유한값. `riemannian_arb_prior` 는 `||u_t||` 자체가 가장 작고 std 도 가장 작음 — **target 자체가 가장 깨끗**한 regression 문제.

![Target field stats](../../outputs/e2_arbitrary_euclidean_prior/analysis/target_field_stats.png)

### 4.2 5000-step 학습 결과

| variant | prior | interp | tail100 | **mean err (px)** | **max err (px)** | wall |
|---|---|---|---|---|---|---|
| riemannian | state | state | 0.029 | 5.6 | 17.2 | ~58 s |
| euclidean  | state | cxcywh | 0.957 | 30.5 | 93.5 | ~58 s |
| **riemannian_arb_prior** | **arb** | **state** | **0.007** | **2.6** | 19.0 | ~58 s |
| euclidean_arb_prior | arb | cxcywh | 0.321 | 35.8 | 100.3 | ~58 s |

![Loss compare](../../outputs/e2_arbitrary_euclidean_prior/loss_compare.png)

**2×2 시각화** (mean_err px)

|           | state interp | cxcywh interp |
|-----------|--------------|----------------|
| state prior | **5.6**    | 30.5           |
| arb prior  | **2.6**    | 35.8           |

- **열 차이 (interp 공간)**: 5.6 → 30.5 (5.5×), 2.6 → 35.8 (14×)
- **행 차이 (prior)**: 5.6 → 2.6 (2.2× 개선), 30.5 → 35.8 (비슷)
- **상호작용**: state interp 에서는 arb prior 가 도움, cxcywh interp 에서는 무관

### 4.3 Trajectory GIF — 4-panel 비교

![Trajectory 4-panel](../../docs/assets/e2_trajectory_compare.gif)

좌→우: `riemannian` / `euclidean` / `riemannian_arb_prior` / `euclidean_arb_prior`.

- **t=0** ([frame](../../docs/assets/e2_frame_t_0.00.png)): state-prior 두 panel 은 b₀ 가 이미지 밖까지 분산 (state Gaussian 의 표준편차가 크다). arb-prior 두 panel 은 canvas 중앙에 모인 mid-size 박스로 시작.
- **t=0.5** ([frame](../../docs/assets/e2_frame_t_0.50.png)): Rm 계열 (1, 3번 panel) 이 GT 근처로 빠르게 모이는 중. Eu 계열은 크게 흔들림.
- **t=1** ([frame](../../docs/assets/e2_frame_t_1.00.png)): **`riemannian_arb_prior` 이 가장 깔끔**하게 10 GT 위에 타이트. Rm baseline 도 우수. Eu 계열 둘 다 시각적으로 어긋난 박스 다수.

---

## 5. 관찰 — 가설 재검토

### 5.1 원래 가설 3개 중 H2 만 맞다

- **H1 (prior entropy 가 주원인)**: ❌ 반증됨. `riemannian_arb_prior` 가 모든 지표에서 최고 (tail 0.007, mean_err 2.6 px). 같은 prior 로 `euclidean_arb_prior` 는 최악 — prior 만으로는 결과가 갈리지 않는다.
- **H2 (interp 공간 이 주원인)**: ✅ 검증됨. prior 축 고정할 때 cxcywh interp 가 state interp 보다 항상 5~14× 더 나쁨.
- **H3 (둘 다 중요)**: 부분만 맞음. interp 공간이 지배적이고, prior 는 interp = state 일 때만 (그것도 좋은 방향으로) 영향.

### 5.2 왜 `arb_prior + state interp` 가 baseline 을 이기는가

- `arb prior` 는 cxcywh 에서 `N(0.5, 1/6²)` 중앙 집중, `[0.02, 1]` bounded → b₀ 중심이 **이미지 중앙 근처**, 크기도 GT 수준 (mean w=0.5, GT w≈0.14 보다 크지만 같은 order).
- 반면 baseline state N(0,I) 는 cx ~ N(0,1) 즉 정규화된 좌표 `[-3, 3]` → b₀ 중심 대부분 이미지 밖. Transport 거리가 훨씬 큼.
- `||u_t||` p99 가 5.87 (baseline) → 3.10 (arb+state) 으로 **2× 작아짐**. 즉 학습해야 할 "거리" 자체가 반으로 줄었고 target 분산도 줄었음 (std 0.98 → 0.50).
- 요약: **prior 가 target 에 가까울수록 flow matching 은 더 쉽다**, 단 interp 공간이 올바를 때만. cxcywh interp 에서는 1/w 특이점이 여전히 문제.

### 5.3 왜 `euclidean_arb_prior` 는 Lipschitz 가 작은데도 가장 나쁜가

- arb prior 의 `|u_cx|` p99 = 0.66 (baseline 2.9 의 1/4) — **중심 이동 신호가 너무 작다**.
- cxcywh interp 에서 u_t_{log_w} = Δw/w_t 는 그래도 p99 ≈ 5 — **w,h 신호는 상대적으로 크다**.
- 결과: loss 가 w,h 차원에 dominate 되고, cx/cy 차원의 signal 이 noise 에 묻힘. 쿼리 간 중심 위치 구분을 못 함.
- 이 pathology 는 state interp 에선 안 나타남 — `u_t.cx` 도 `u_t.log_w` 도 같은 스케일 (state N(0,I) 아닌 arb 기반에서도 p99 ≈ 0.66 vs 2.5, 4× 차이에 그침, 학습 가능한 범위).

### 5.4 Lipschitz 와 학습 성능의 관계

| 축 | Lipschitz (L̂ p99) | 학습 tail loss |
|---|---|---|
| state interp | 0 (두 prior 모두) | 0.029 / 0.007 |
| cxcywh interp | 5.8 / 16.8 | 0.32 / 0.96 |

**state interp** 는 L̂=0 으로 이론상 regression 가장 쉬움. 실제도 그러함. **cxcywh interp** 는 L̂ bounded 여부와 관계없이 둘 다 실패. 즉 **Lipschitz 0 (constant field) 이 결정적 조건** — 비-0 L̂ 은 크기 상관없이 학습 방해.

---

## 6. 결론

1. **진짜 축은 interp 공간**. e1 에서 "constant vs time-dependent u_t" 차이로 얘기했던 것이 여기서 더 명확히 확인된다.
2. **prior 는 부차적** — support 만 reasonable 하면 된다. 오히려 target 에 가까운 arb prior 는 state interp 와 결합해 baseline 을 2× 이상 개선한다.
3. **`euclidean_arb_prior` 의 실패 원인은 "bounded prior → 작은 signal"** — cxcywh interp 의 time-dependent u_t 가 균일하지 않은 per-dim 스케일을 만들어 loss 가 w,h 에 dominate 됨.
4. Phase 4 VOC/COCO ablation 에 반영:
   - state interp (Riemannian) 을 기본으로 채택 확정
   - prior 를 `arb cxcywh` 로 바꾸는 ablation 이 추가 개선 가능성 있음 (요조사)

**Winner**: `riemannian_arb_prior` (tail 0.007, mean_err 2.6 px, max_err 19 px). baseline `riemannian` 대비 mean_err 2× 개선.

---

## 7. 다음 단계

- **E3** — prior 를 더 공격적으로 target align. `b₀_cx ~ dataset 통계에 맞춘 mixture` 로 훈련. MNIST Box 가 아닌 real detection (VOC/COCO) 에서도 이 효과가 지속되는지.
- **E4** — arb_prior 의 `μ, σ, ε` hyperparam sweep. `ε` 을 극단적으로 작게 (0.005) 하면 `euclidean_arb_prior` 의 Lipschitz 가 다시 커지는지.
- **interp 공간 hybrid** — state 보간 + 마지막에 cxcywh 보정, 같은 것을 혼합해 각 장점 취함.
- 본 실험의 reverse engineering: **"prior = target 에 가까운 간단한 Gaussian + state interp"** 가 일반 flow matching 설계의 legit recipe 인지 문헌 조사.
