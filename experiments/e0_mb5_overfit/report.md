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
4. **Linear baseline은 여전히 실패** — train 분포(`b_0 ~ U(0.05, 0.95)` in cxcywh)와 infer 분포(`randn` in state space) 불일치로 궤적이 수렴하지 않음 ([ISSUES.md의 별도 항목 예정](../../docs/ISSUES.md)).

---

## 7. 시각 결과

### 7.1 Winner (B) — GT vs Pred (256×256)
`outputs/e0_mb5_overfit/B_longer/overfit_gt_vs_pred.png`

- 노랑(GT) + 녹색(pred) 이 10개 digit 모두에서 눈에 띄게 겹침
- 옅은 파랑 = init b₀ (대부분 캔버스 밖, clip 후 가장자리에 표시)

### 7.2 Trajectory GIF — Riemannian vs Linear
`outputs/e0_mb5_overfit/B_longer/trajectory_gif/trajectory_compare.gif`

- 두 모델 모두 winner 설정(5000 step, cosine, ODE 50)로 재학습 후 궤적 캡처
- t=0 동일한 Gaussian noise → t=1 **Riemannian만 GT에 정확히 수렴**, Linear는 발산/편향
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

## 9. 다음 단계 (out of scope)

- **E1 trajectory ablation**: 본 실험의 Linear baseline 실패가 **train/infer prior 불일치**에 기인함을 확인 → `LinearTrajectory`에 init-noise 훅 추가 후 공정 비교 필요. 별도 이슈로 기록 예정.
- **다중 샘플 overfit** (`num_samples=4, 16`) — 일반화 초입 확인.
- **Phase 3 S1 train.py 구축** 시 본 실험의 training 루프 패턴을 재사용.
