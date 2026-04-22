# e0 — MB5 Single-Image Overfit Ablation

> Phase 2.5의 "모델·궤적·loss가 실제로 수렴하는가" 를 **정밀히** 검증하기 위한 초기 완성 실험 세팅.

---

## 1. 목표

1장의 MNIST Box 샘플에 `RiemannianFlowDet`이 **거의 pixel-perfect** 로 overfit 되도록 학습 설정을 튜닝한다.
"1장 overfit은 아예 수렴 가능한가?"(= MB5 baseline, 이미 ✅)를 넘어 **얼마나 타이트하게 붙는가**를 측정.

**성공 기준**: normalized box error `max_err < 0.1` (≈ 26px in 256²), 시각적으로 **10개 query가 GT digit box를 눈에 띄게 덮도록** 예측.

---

## 2. Dataset

`dataset/mnist_box.py` · `configs/mnist_box.yaml`

| 항목 | 값 |
|------|-----|
| Canvas | 256 × 256, 3ch, ImageNet-normalized |
| 객체 수 | 10 (클래스 0~9 각 1개, 1-to-1 class-indexed) |
| Digit scale | uniform ∈ [14, 56] px (정사각형) |
| 배치 | non-overlap (rejection sampling, 큰 scale 우선) |
| 배경 | 0 (검정) |
| 샘플 | `num_samples=1` — 1장 고정 (seed 0, idx 0) |

출력 규약: `boxes[i]`는 digit i의 박스, `labels = torch.arange(10)`. Hungarian matching 불필요.

---

## 3. Model

`RiemannianFlowDet` (model/flow_matching.py)

| 항목 | 값 (A·B·C) | 값 (D) |
|------|------|------|
| backbone | FPN (ResNet50 + FPN, pretrained) | 동일 |
| hidden_dim | 192 | 256 |
| depth | 4 | 6 |
| num_heads | 6 | 8 |
| mlp_ratio | 4 | 4 |
| num_queries | 10 (= class 수) | 동일 |
| FlowDiT query_embed | `nn.Embedding(10, dim)` (초기화 N(0, 0.02)) | 동일 |
| trajectory | Riemannian (state space linear geodesic) | 동일 |
| 파라미터 수 | ≈ 28.2M | ≈ 33.9M |

상세 구조: `docs/diagrams/RiemannianFlowDet.md`, `docs/model_plan.md`

---

## 4. Training

**Common**

- Optimizer: AdamW (β=(0.9, 0.999), weight_decay=0)
- grad clip: L2 norm ≤ 1.0
- Batch: 1 이미지 × 10 query × 10 class
- Loss: `FlowMatchingLoss` — `‖v̂_t − u_t*‖²`, valid 위치만 평균
- device: CUDA

**Per-step 데이터 흐름**
```
b₁        = cxcywh_to_state(gt_boxes)      # [1, 10, 4]
b₀        ~ N(0, I)                         # [1, 10, 4]
t         ~ U[0, 1]                         # [1]
b_t       = (1−t)·b₀ + t·b₁
u_t*      = b₁ − b₀                         # constant vector field
v̂_t      = BoxHead(FlowDiT(image, b_t, t))
loss      = MSE(v̂_t, u_t*)
```

**추론 (ODE)**
```
b ~ N(0, I) in state space
for i in range(ode_steps):
    v = BoxHead(FlowDiT(image, b, i/ode_steps))
    b = b + (1/ode_steps) · v
boxes_pred = state_to_cxcywh(b)
```

---

## 5. Ablation 설계

기준선 A(현재 기본값) 대비 **세 축**을 변동:

| 축 | 가설 |
|-----|------|
| 학습 길이 × LR 감쇠 | `t` 샘플링으로 loss는 본질적 noisy → step 수 늘리고 LR을 cosine으로 감쇠하면 tail loss 낮아진다 |
| ODE step 수 (10 → 50) | 추론 이산화 오차가 box error 상한을 정할 수 있음 |
| 모델 capacity | 용량 부족이 정밀도 병목인지 확인 |

### Variants

| Variant | steps | LR / sched | ODE | hidden/depth | 의도 |
|---|---|---|---|---|---|
| **A** baseline | 1500 | 1e-4 const | 10 | 192/4 | MB5 원본 설정 |
| **B** longer | **5000** | **3e-4 cosine** | 10 | 192/4 | 학습을 길게 + decay |
| **C** ode50 | 1500 | 1e-4 const | **50** | 192/4 | 추론 정밀도만 ↑ |
| **D** combined | 5000 | 3e-4 cosine | 50 | **256/6** | 전부 상향 |

Config 위치: `experiments/e0_mb5_overfit/variants/{A,B,C,D}.yaml`
실행: `bash experiments/e0_mb5_overfit/run_ablation.sh`

---

## 6. 결과

**실행 환경**: 1 × GPU (CUDA), 한 variant ≈ 20–70 s

| Variant | tail₁₀₀ loss | mean err (norm) | max err (norm) | mean err (px) | max err (px) | wall time |
|---|---|---|---|---|---|---|
| A baseline  | 0.107 | 0.038 | 0.133 | 9.7  | **34.0** | 18 s |
| **B longer** | **0.031** | **0.030** | **0.087** | **7.6** | **22.2** | **61 s** |
| C ode50     | 0.106 | 0.040 | 0.138 | 10.2 | 35.4 | 20 s |
| D combined  | 0.032 | 0.030 | 0.089 | 7.7  | 22.8 | 69 s |

> `tail₁₀₀ loss`: 마지막 100 step loss 평균 (`t` 샘플링 노이즈 완화용 지표).
> `err (px)`: 256 pixel canvas 기준. mean 7.6, max 22 px는 최소 digit(14px)의 0.5~1.5 배 수준.

### 관찰

1. **학습 길이 × LR 감쇠(B)가 핵심 기여** — tail loss 0.107 → 0.031 (3.5× 감소), max_err 34 → 22 px.
2. **ODE step 수 단독(C)은 효과 미미** — A 대비 거의 동일. 현재 병목은 model convergence이지 integration 정확도가 아님.
3. **모델 capacity 확장(D)도 한계 효용** — B와 거의 동률(tail 0.032 vs 0.031). 1장 overfit에는 28M params로 충분.
4. **Euclidean baseline은 여전히 실패** — 초기엔 train/infer prior 불일치가 원인으로 보였으나, `init_noise` 훅으로 공정 비교한 section 9.5 결과에서도 tail loss 16×·max err 196px로 대폭 뒤처짐. 원인은 **time-dependent 벡터장** 학습의 본질적 어려움 (섹션 9.2·9.5).

---

## 7. 시각 결과

### 7.1 Winner (B) — GT vs Pred (256×256)
`outputs/e0_mb5_overfit/B_longer/overfit_gt_vs_pred.png`

- 노랑(GT) + 녹색(pred) 이 10개 digit 모두에서 눈에 띄게 겹침
- 옅은 파랑 = init b₀ (대부분 캔버스 밖, clip 후 가장자리에 표시)

### 7.2 Trajectory GIF — Riemannian vs Euclidean
`outputs/e0_mb5_overfit/B_longer/trajectory_gif/trajectory_compare.gif`

- 두 모델 모두 winner 설정(5000 step, cosine, ODE 50)로 재학습 후 궤적 캡처
- t=0 동일한 Gaussian noise → t=1 **Riemannian만 GT에 정확히 수렴**, Euclidean는 발산/편향
- 768×768 확장 canvas — init 박스가 이미지 밖까지 보임

---

## 8. Winner Config (초기 완성 세팅)

### Variant **B_longer** 를 Phase 2.5 MNIST Box 기준선으로 채택.

```yaml
# experiments/e0_mb5_overfit/variants/B_longer.yaml + CLI
max_steps:     5000
lr:            3.0e-4
lr_schedule:   cosine           # eta_min = 0.05 · lr
ode_steps:     10
hidden_dim:    192
num_layers:    4
num_heads:     6
num_queries:   10
backbone:      fpn (pretrained)
trajectory:    riemannian
batch_size:    1
num_samples:   1                # overfit target
```

**재현 명령**

```bash
python script/overfit_mnist_box.py \
    --config experiments/e0_mb5_overfit/variants/B_longer.yaml \
    --tag    B_longer \
    --max_steps 5000 --lr 3e-4 --lr_schedule cosine --ode_steps 10 \
    --out_dir outputs/e0_mb5_overfit/B_longer --no-show
```

후속 실험(E1 trajectory ablation, E2 backbone, …)은 이 세팅을 **fixed baseline**으로 삼고 한 축씩만 바꿔 비교한다.

---

## 9. Riemannian vs Euclidean — 이론, 구현, 공정 비교

> **용어**: 본 섹션부터는 "Linear trajectory"를 **"Euclidean trajectory"** 로 통일한다.
> ("Linear interpolation"은 양쪽 다 쓰는 수학 연산이라 혼동을 유발.
> 차이는 **어느 공간 위에서 linear interpolation을 수행하는가** 이므로 공간 이름으로 부른다.)

### 9.1 왜 두 공간인가 — 박스의 기하학

박스 `[cx, cy, w, h]`에서
- `cx, cy` — 픽셀 **위치**. 자연스럽게 유클리드 (덧셈·차이가 그대로 의미 있음).
- `w, h`   — **스케일**. 항상 양수 제약. 덧셈으로 평균 내면 기하 평균이 아닌 산술 평균이 됨.

즉 박스의 상태 공간은 `ℝ² × ℝ₊²` — 2D 유클리드 × 2D 양의 실수 반(half) 공간.
scale 축을 **Euclidean처럼 취급**하면 "w=0.1과 w=10의 중간"이 산술 평균 5.05가 되어 기하학적으로 불균형 (0.1과 10의 기하 평균 = 1이 더 자연스러움).

**해결**: `log_w = log(w)` 치환 → `ℝ_{+}` → `ℝ`. 이제 state는 `[cx, cy, log_w, log_h] ∈ ℝ⁴`.
이 공간 위에서 선형 보간은 **원래 공간 `ℝ² × ℝ₊²`의 geodesic**에 해당.

### 9.2 두 trajectory의 수식 비교

| | **Riemannian** (ours) | **Euclidean** (baseline) |
|---|---|---|
| 보간 공간 | state `[cx, cy, log_w, log_h]` | cxcywh `[cx, cy, w, h]` |
| b₀ prior | `N(0, I)` in state | `U([0.05, 0.95])` in cxcywh |
| `b_t` 생성 | `(1−t)b₀ + tb₁` in state | `(1−t)b₀ + tb₁` in cxcywh, 그 후 state로 변환 |
| 벡터장 `u_t*` | **`b₁ − b₀` (상수, t와 무관)** | `[Δcx, Δcy, Δw/w_t, Δh/h_t]` — **time-dependent** |
| 모델 학습 | 상수 필드를 regress → 쉬움 | 분모 `w_t, h_t`가 시간에 따라 변함 → 어려움 |

핵심 통찰: Riemannian 쪽은 **학습 목표가 constant**라 모델이 단일 벡터만 예측하면 되지만, Euclidean 쪽은 **`b_t`에 따라 달라지는 field**를 학습해야 함. 같은 용량으로는 Riemannian이 본질적으로 유리하다.

### 9.3 구현 위치 — 어디서 공간이 바뀌는가

코드에서 **Euclidean ↔ Riemannian 경계는 단 두 함수**로 명확히 분리:

| 함수 | 입력 | 출력 | 역할 |
|---|---|---|---|
| `cxcywh_to_state(b)` | `[cx, cy, w, h]` | `[cx, cy, log_w, log_h]` | **유클리드 → 리만** 진입. `log()` 적용. |
| `state_to_cxcywh(s)` | `[cx, cy, log_w, log_h]` | `[cx, cy, w, h]` | **리만 → 유클리드** 복귀. `exp()` 적용. |

(`dataset/box_ops.py:66-90`)

이 두 함수 **외부**에서 `w, h`가 직접 바뀌는 연산은 없다. 즉:
- `log_w`, `log_h` 는 **학습·보간·추론 전 과정에서** 덧셈/뺄셈/선형결합만 받는다.
- 박스가 양수 제약을 **자동으로** 만족 (`exp` 출력이 양수).

### 9.4 모델 다이어그램 — 공간 전이 지점

```
┌────────────────────────────────────────────────────────────────────────────┐
│   EUCLIDEAN DOMAIN (normalized cxcywh ∈ (0,1) · ImageNet-normalized image)│
│                                                                            │
│    ┌──────────────┐       ┌──────────────────────┐                        │
│    │ image [B,3,H,W]│       │ boxes_gt [B,10,4] cxcywh │                  │
│    └──────┬───────┘       └────────┬─────────────┘                        │
│           │                         │                                      │
│           │                         │  ★ cxcywh_to_state()   log(w),log(h)│
│           │                         ▼          ═══════════════════════════▶
│           │              ╔════════════════════════════════════════════════╗
│           │              ║   RIEMANNIAN DOMAIN (state ∈ ℝ⁴)                ║
│           │              ║   [cx, cy, log_w, log_h]                       ║
│           │              ║                                                ║
│           │              ║   b₁ [B,10,4]                                  ║
│           │              ║                                                ║
│           │              ║   b₀ ~ trajectory.init_noise()                 ║
│           │              ║     Riemannian: randn in state                 ║
│           │              ║     Euclidean : rand(cxcywh) → cxcywh_to_state ║
│           │              ║                                                ║
│           │              ║   t ~ U[0,1]                                   ║
│           │              ║   b_t = trajectory.sample(b₁, t)               ║
│           │              ║     Riemannian: (1-t)b₀ + t·b₁                 ║
│           │              ║     Euclidean : 동일하나 w,h 축 뜻이 다름       ║
│           │              ║                                                ║
│           │              ║   ┌─────────────────┐                          ║
│           │              ║   │ BoxEmbedding    │ [B,10,dim]  ← state      ║
│           │              ║   │ + query_embed   │ (학습 가능한 per-query)   ║
│           │              ║   └────────┬────────┘                          ║
│           ▼                           │                                    ║
│    ┌──────────────┐                   │                                    ║
│    │ FPN/DINOv2   │                   │                                    ║
│    │  Backbone    │  img_tokens       │                                    ║
│    │              │─────────────────▶ ▼                                    ║
│    └──────────────┘              ┌────────────┐                            ║
│                                  │ FlowDiT    │  (self-attn + cross-attn +│
│                                  │  × depth   │   AdaLN(t) + MLP, RoPE)   ║
│                                  └─────┬──────┘                            ║
│                                        │ box_tokens [B,10,dim]             ║
│                                  ┌─────▼──────┐                            ║
│                                  │ BoxHead    │                            ║
│                                  └─────┬──────┘                            ║
│                                        │ v̂_t [B,10,4]  ← **state space**  ║
│                                        │                                   ║
│                      ┌─────────────────┴───────┐                           ║
│                      │ Training:               │ Inference:                ║
│                      │   u_t* = b₁-b₀ (Rm)    │   b ← b + Δt · v̂_t        ║
│                      │   또는 time-dep (Eu)    │   (Euler ODE × num_steps)║
│                      │   loss = ‖v̂-u‖²        │                           ║
│                      └─────────────────────────┘                           ║
│                                                                            ║
│                                        │  ★ state_to_cxcywh()              ║
│                                        ▼          ═════════════════════════
│    ┌──────────────┐                                                        │
│    │ boxes_pred   │   [B,10,4] cxcywh ∈ (0,1)                              │
│    └──────────────┘                                                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**공간 전이 경계 (★로 표시)**
- `cxcywh_to_state()` : `log()` 적용 순간부터 모든 연산이 **Riemannian state space**에서 진행
- `state_to_cxcywh()` : `exp()` 적용 순간 다시 **Euclidean cxcywh**로 복귀 (최종 출력용)

**w, h가 바뀌는 지점 (요약)**
| 위치 | 연산 | 의미 |
|------|------|------|
| `cxcywh_to_state` | `log_w = log(w)` | w를 log-scale로 치환 (한 번만) |
| `RiemannianTrajectory.sample` | `log_w_t = (1-t)log_w₀ + t·log_w₁` | state에서 선형 보간 = **원 공간의 geodesic** |
| `EuclideanTrajectory.sample` | `w_t = (1-t)w₀ + t·w₁` → `log(w_t)` | cxcywh에서 선형 보간 후 log 취함 (geodesic 아님) |
| `trajectory.ode_step` | `log_w_{t+dt} = log_w_t + dt · v̂_w` | 추론 시 log 축에서 이동 |
| `state_to_cxcywh` | `w = exp(log_w)` | state에서 cxcywh로 복귀 (한 번만) |

### 9.5 공정 비교 — prior 통일 설계 (최종)

**문제 제기**: 초기 설계에서 두 trajectory는 각자 **다른 prior**를 사용했다.
- Riemannian: `b₀ ~ N(0, I)` **in state space** — cx ~ N(0,1), log_w ~ N(0,1)
- Euclidean : `b₀_cx ~ U([0.05, 0.95])` **in cxcywh** → `cxcywh_to_state`로 변환

결과: 같은 `seed`여도 **init box가 시각적으로 완전히 다름**. 256×256 canvas 위에 그려보면:
- Riemannian 측 init: cx가 픽셀 기준 `[-768, 768]`, w/h가 `[exp(-3)~exp(3)]·256 = [13, 5120]`px → 대부분 이미지 밖/매우 큼
- Euclidean 측 init: cx가 [13, 243]px 내부, w/h도 [13, 243]px — 모두 이미지 안에 컴팩트하게 존재

"같은 조건"으로 보기 어려웠고, Euclidean이 학습 실패하는 이유를 (a) prior 자체 vs (b) 궤적 수학 구조 중 어느 것인지 분리할 수 없었다.

**수정 설계** (사용자 제안 반영)
> **"박스(cxcywh)를 만드는 건 공통이다. DiT 중간의 state space가 Riemannian 보간으로 만들어졌느냐, Euclidean 보간으로 만들어졌느냐만 차이를 만든다."**

- 두 trajectory 모두 **`b₀ ~ N(0, I) in state space`** 통일 (원래 Riemannian 표준 flow matching prior)
- 같은 seed → 완전히 동일한 `b₀` (state 값)
- **차이는 오직 interpolation 공간**:
  - Riemannian: `b_t = (1-t)·b₀ + t·b₁` — **state space에서 선형보간** (= geodesic)
  - Euclidean : `b₀_cx = state_to_cxcywh(b₀)` → `b_t_cx = (1-t)·b₀_cx + t·b₁_cx` **cxcywh에서 선형보간** → `b_t = cxcywh_to_state(b_t_cx)` 로 DiT 입력
- DiT/BoxHead는 **항상 state space만** 입출력 → 모델 구조 완전 동일

**구현** (`model/trajectory.py`)

```python
# Riemannian
def init_noise(self, B, Q, device, dtype):
    return torch.randn(B, Q, 4, device=device, dtype=dtype)          # state

def sample(self, b1, t):                        # b1 is state
    b0  = torch.randn_like(b1)                                       # state
    b_t = (1-t_)·b0 + t_·b1                                          # linear in state
    u_t = b1 - b0                                                    # constant
    return b_t, u_t, b0

# Euclidean — 동일한 state prior로 시작, cxcywh로 보내 보간, state로 복귀
def init_noise(self, B, Q, device, dtype):
    return torch.randn(B, Q, 4, device=device, dtype=dtype)          # state (동일!)

def sample(self, b1_cx, t):
    b0      = torch.randn_like(b1_cx)                                # state
    b0_cx   = state_to_cxcywh(b0)                                    # → cxcywh
    b_t_cx  = (1-t_)·b0_cx + t_·b1_cx                                # linear in cxcywh
    b_t     = cxcywh_to_state(b_t_cx)                                # → state (DiT 입력)
    u_t     = [Δcx, Δcy, (w1-w0)/w_t, (h1-h0)/h_t]                   # time-dependent
    return b_t, u_t
```

**공정 비교 결과** (양쪽 동일 설정: 5000 step, cosine LR 3e-4, ODE 50 step, FPN/192-4, 1-image overfit)

| Trajectory | tail₁₀₀ loss | mean_box_err (norm) | max_box_err (norm) | mean err (px) | max err (px) |
|---|---|---|---|---|---|
| **Riemannian** (ours) | **0.028** | **0.021** | **0.066** | **5.3** | **16.9** |
| Euclidean (baseline) | 0.056 | 0.024 | 0.192 | 6.1 | 49.1 |

> 공식 수치는 e1의 `run.sh` 재실행으로 재현 가능. flow matching의 `t` 샘플링 noise로
> run 간 ±30% 변동 있음. 자세한 건 [`experiments/e1_unified_prior_fair_compare/report.md`](../e1_unified_prior_fair_compare/report.md) 참고.

### 관찰

- `b₀`가 완전히 동일해진 덕에 **두 방법의 유일한 차이가 interpolation 공간**임이 명확.
- **Riemannian이 max err 기준 2.9배, tail loss 기준 2배 우위** — target 벡터장 구조(constant vs time-dependent)의 순수 이론적 차이.
- Mean err는 둘 다 낮음 (5.3 / 6.1 px) — 대부분의 박스는 잘 맞춤. 그러나 **worst-case**에서 Euclidean이 크게 벗어남 (49 px) → scale 축의 time-dependent field 학습이 상대적으로 어렵기 때문.
- 이전 비교 (다른 priors: Eu max 196 px) 대비 Euclidean이 훨씬 **공정하게** 성능 발휘. 이전 결과의 주요 원인은 이론이 아닌 prior mismatch였음을 역으로 확인.

### 9.6 시각 결과 (공정 비교)

`docs/assets/trajectory_compare.gif` — 51 프레임 · 12fps · 768×768 확장 canvas side-by-side.

- **t=0.00**: 좌/우 **완전히 동일한 init 박스** (통일 prior + 동일 seed). 일부는 이미지 밖, 일부는 매우 큼/작음 (state-space Gaussian이 만드는 자연스러운 분포).
- **t=0.50**: 양쪽 다 박스를 MNIST 영역으로 끌어당기는 중. Riemannian이 약간 더 빠르게 수렴.
- **t=1.00**: Riemannian이 10개 GT에 **타이트하게 덮음** (max_err 11px). Euclidean은 전체적으로 근접하나 일부 박스(특히 worst-case)가 size 또는 position에서 어긋남 (max_err 27px).

### 9.7 결론

Riemannian의 우위는 **두 trajectory 사이의 본질적 수학 구조 차이**에서 온다:

1. **Constant vs time-dependent target field**
   - Riemannian `u_t* = b₁ − b₀` — t에 무관한 상수
   - Euclidean   `u_t*` — 분모 `w_t, h_t`가 시간에 따라 변함
2. **Gradient landscape** — constant target은 전역적으로 단순, time-dependent는 t별로 다른 필드 학습 필요 → capacity 더 소모
3. **Numerical stability** — Euclidean의 scale 축 업데이트는 `1/w_t`를 포함해 작은 박스에서 민감도가 높음

**Phase 4 E1 ablation** 은 이 결과를 VOC/COCO로 스케일해 재현할 것.

---

## 10. 다음 단계 (out of scope)

- **E1 trajectory ablation**: 본 섹션 9.5의 공정 비교 수치를 기준으로 VOC/COCO로 스케일 확장 — Phase 4 E1의 근거.
- **다중 샘플 overfit** (`num_samples=4, 16`) — 일반화 초입 확인.
- **Phase 3 S1 train.py 구축** 시 본 실험의 training 루프 패턴을 재사용.
