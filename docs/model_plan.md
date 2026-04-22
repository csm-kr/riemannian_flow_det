# Model Architecture Plan & Implementation Analysis

## 1. 전체 파이프라인 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRAINING                                                                   │
│                                                                             │
│  images [B,3,H,W]           boxes_gt_list: List[ [Ni,4] cxcywh ]          │
│       │                              │                                      │
│       │                     cxcywh_to_state()                              │
│       │                              │                                      │
│       │                     b1 [B,N,4]  ←── padded, masked                 │
│       │                              │                                      │
│       │                    ┌─────────┴──────────┐                           │
│       │                    │ RiemannianTrajectory│                           │
│       │                    │  b0 ~ N(0,I)       │                           │
│       │                    │  t  ~ U[0,1]  [B]  │                           │
│       │                    │  b_t=(1-t)b0+t·b1  │                           │
│       │                    │  u_t = b1 - b0     │                           │
│       │                    └────────┬───────────┘                           │
│       │                            │                                        │
│       │                    b_t [B,N,4]   t [B]                             │
│       │                            │                                        │
│       └──────────┬─────────────────┘                                        │
│                  ▼                                                          │
│          ┌──────────────┐                                                   │
│          │  FlowDiT     │   (see Section 3)                                │
│          └──────┬───────┘                                                   │
│                 │  box_tokens [B,N,D]                                       │
│                 ▼                                                           │
│          ┌──────────────┐                                                   │
│          │   BoxHead    │  LayerNorm → Linear(D→D) → SiLU → Linear(D→4)   │
│          └──────┬───────┘                                                   │
│                 │  v̂_t [B,N,4]                                             │
│                 ▼                                                           │
│          ┌──────────────────────────┐                                       │
│          │  FlowMatchingLoss        │                                       │
│          │  MSE( v̂_t[mask],        │                                       │
│          │       u_t[mask] )        │                                       │
│          └──────────────────────────┘                                       │
│                 │  loss: scalar                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  INFERENCE                                                                  │
│                                                                             │
│  images [B,3,H,W]                                                           │
│       │                                                                     │
│       │     b ~ N(0,I) [B,Q,4]  ← Q = num_queries (300)                   │
│       │           │                                                         │
│       │    ┌──────┴────────────────────────────────────────┐               │
│       │    │  Euler ODE  (num_steps = 10)                  │               │
│       │    │  for i in range(num_steps):                   │               │
│       │    │      t = i / num_steps  [B]                   │               │
│       │    │      v = FlowDiT+BoxHead(images, b, t)        │               │
│       │    │      b = b + dt * v      (dt = 1/num_steps)   │               │
│       │    └──────────────────────────────────────────────-┘               │
│       │           │  b [B,Q,4]  (state space)                               │
│       │           ▼                                                         │
│       │    state_to_cxcywh()                                                │
│       │           │                                                         │
│       └───────────┘  boxes [B,Q,4]  normalized cxcywh                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Box State Space — Riemannian Manifold

```
 Euclidean center      Log-scale size
   (ℝ²)                 (ℝ₊² → ℝ²)
┌────────────┐        ┌──────────────────┐
│  cx ∈ ℝ   │        │  log_w = ln(w)   │
│  cy ∈ ℝ   │        │  log_h = ln(h)   │
└────────────┘        └──────────────────┘
      ↕ cxcywh_to_state / state_to_cxcywh (dataset/box_ops.py)

 Manifold: ℝ² × ℝ₊²
 Metric:   g = diag(1, 1, 1/w², 1/h²)   (log-Euclidean)
 Geodesic: straight line in [cx, cy, log_w, log_h]  ✓

 Flow path:
   b₀ = noise       ~ N(0, I)   in state space
   b₁ = target box  = cxcywh_to_state(boxes_gt)
   b_t = (1-t)·b₀ + t·b₁       ← geodesic  ✓
   u_t* = b₁ − b₀               ← constant vector field  ✓

 Why log-scale?
   Euclidean: (1-t)·w₀ + t·w₁   → path depends on absolute scale
   Log-space: exp((1-t)·log_w₀ + t·log_w₁)  → geometric mean = geodesic
```

---

## 3. FlowDiT 상세 다이어그램

```
                images [B,3,H,W]
                     │
          ┌──────────▼───────────┐
          │     FPNBackbone      │
          │                      │
          │  ResNet50:           │
          │  stem → layer1 (×)   │
          │         layer2 [512] │── P3: [B,D,H/8, W/8]   ─┐
          │         layer3[1024] │── P4: [B,D,H/16,W/16]   │ FPN
          │         layer4[2048] │── P5: [B,D,H/32,W/32]   │
          │                      │                          ↓
          │  torchvision FPN     │  flatten + concat
          │  in:[512,1024,2048]  │
          │  out: dim (256)      │
          └──────────┬───────────┘
                     │
          img_tokens [B, L_img, D]     hw_list [(H3,W3),(H4,W4),(H5,W5)]
          L_img = H/8·W/8 + H/16·W/16 + H/32·W/32
          (e.g. 800×800 → 10000+2500+625 = 13125 tokens)

    b_t [B,N,4]           t [B]
         │                  │
  ┌──────▼──────┐    ┌──────▼──────────────┐
  │ BoxEmbedding│    │ SinusoidalEmbedding  │
  │ Linear(4→D) │    │ sincos → MLP(D→4D→D)│
  │ LayerNorm   │    └──────────┬───────────┘
  └──────┬──────┘               │
         │ [B,N,D]          t_emb [B,D]
         │
  box_freqs = build_2d_rope_freqs(head_dim, cx_mean, cy_mean)  [N, D_h//2]
  img_freqs = build_2d_grid_rope_freqs per FPN scale → cat     [L, D_h//2]
         │
  ┌──────┴────────────────────────────────────────────────────┐
  │  DiTBlock × depth (default: 6)                            │
  │                                                           │
  │  ┌─────────────────────────────────────────────────────┐  │
  │  │ 1. Self-Attention  (box ↔ box)                      │  │
  │  │    norm1(box_tokens)                                │  │
  │  │    MultiHeadAttnRoPE(Q=box, K=box, V=box,          │  │
  │  │                       q_freqs=box_freqs,           │  │
  │  │                       k_freqs=box_freqs)           │  │
  │  │    residual add → [B, N, D]                        │  │
  │  │                                                     │  │
  │  │ 2. Cross-Attention  (box → image)                  │  │
  │  │    norm2(box_tokens)                                │  │
  │  │    MultiHeadAttnRoPE(Q=box, K=img, V=img,          │  │
  │  │                       q_freqs=box_freqs,           │  │
  │  │                       k_freqs=img_freqs)           │  │
  │  │    residual add → [B, N, D]                        │  │
  │  │                                                     │  │
  │  │ 3. AdaLN + MLP  (timestep conditioning)            │  │
  │  │    AdaLN(box_tokens, t_emb):                       │  │
  │  │      LayerNorm(elementwise_affine=False)            │  │
  │  │      scale,shift = Linear(D→2D)(t_emb)            │  │
  │  │      out = norm(x)·(1+scale) + shift               │  │
  │  │    MLP: Linear(D→4D) → GELU → Linear(4D→D)        │  │
  │  │    residual add → [B, N, D]                        │  │
  │  └─────────────────────────────────────────────────────┘  │
  └──────┬────────────────────────────────────────────────────┘
         │
  LayerNorm(D)
         │
  box_tokens [B, N, D]
```

---

## 4. MultiHeadAttentionRoPE 내부

```
  q_in [B,Nq,D]   k_in [B,Nk,D]   v_in [B,Nk,D]
       │                │                │
  q_proj(D→D)    k_proj(D→D)    v_proj(D→D)    (no bias on qkv)
       │                │
  view [B,N,H,D_head]   view [B,N,H,D_head]
       │                │
  apply_rope(q, k, q_freqs, k_freqs):
    cos/sin from freqs → x·cos + rotate_half(x)·sin
       │                │
  transpose [B,H,N,D_head]
       │
  F.scaled_dot_product_attention   (Flash Attention 가능)
       │
  out [B,H,Nq,D_head] → reshape [B,Nq,D]
       │
  out_proj(D→D)
       │
  [B, Nq, D]
```

---

## 5. RoPE 좌표 체계

```
  Box Self-Attention:
    position = (cx, cy) from b_t   ← box center position
    cx_mean, cy_mean = b_t[:,:,0/1].mean(0)  [N]  (batch-mean 근사)
    freqs = build_2d_rope_freqs(head_dim, cx_mean, cy_mean)   [N, D_h//2]
      head_dim//4 dims ← cx frequency
      head_dim//4 dims ← cy frequency

  Image Cross-Attention (K):
    position = regular spatial grid (i/H, j/W)
    freqs = build_2d_grid_rope_freqs(head_dim, H_k, W_k)   [H_k*W_k, D_h//2]
    per-scale → cat → [L_img, D_h//2]

  Constraint: head_dim % 4 == 0  (2D RoPE 필수 조건)
```

---

## 6. 파일별 모듈 구조 & 텐서 흐름

```
model/
│
├── modules.py          (primitives — 의존성 없음)
│   ├── SinusoidalEmbedding  t:[B] → [B,D]
│   ├── BoxEmbedding         [B,N,4] → [B,N,D]
│   ├── AdaLN                [B,N,D] + ctx[B,D] → [B,N,D]
│   ├── MLP                  [B,N,D] → [B,N,D]
│   ├── build_2d_rope_freqs  (cx,cy)[N] → [N, D_h//2]
│   ├── build_2d_grid_rope_freqs  (H,W) → [H*W, D_h//2]
│   └── apply_rope           q,k [B,N,H,Dh] → rotated q,k
│
├── backbone.py         (modules.py 불필요)
│   └── FPNBackbone
│       ResNet50 (layer2/3/4) + torchvision FPN
│       [B,3,H,W] → img_tokens[B,L,D], hw_list[(H,W)×3]
│
├── dit.py              (← modules.py, backbone.py)
│   ├── MultiHeadAttentionRoPE   MHA + optional RoPE
│   ├── DiTBlock         self-attn + cross-attn + adaLN-MLP
│   └── FlowDiT          backbone + box_embed + time_embed + blocks
│       [B,3,H,W] + [B,N,4] + [B] → [B,N,D]
│
├── head.py             (의존성 없음)
│   └── BoxHead          LN → Linear(D→D) → SiLU → Linear(D→4)
│       [B,N,D] → [B,N,4]
│
├── trajectory.py       (← dataset/box_ops.py)
│   ├── RiemannianTrajectory   state space에서 geodesic
│   │   sample: b1[B,N,4] + t[B] → b_t, u_t, b0
│   │   ode_step: b_t + dt·v_t
│   └── LinearTrajectory       cxcywh space에서 선형
│       sample: b1_cx[B,N,4] + t[B] → b_t(state), u_t(state)
│
├── loss.py             (의존성 없음)
│   └── FlowMatchingLoss   MSE(v̂_t, u_t)  → scalar
│
├── flow_matching.py    (← backbone, dit, head, trajectory, loss)
│   └── RiemannianFlowDet
│       forward_train   → {"loss": scalar}
│       forward_inference → boxes [B,Q,4] cxcywh
│
└── __init__.py
    └── build_model(config) → RiemannianFlowDet
```

---

## 7. 하이퍼파라미터 표

| 파라미터 | 기본값 | 근거 |
|---------|--------|------|
| `dim` (hidden) | 256 | DiffusionDet |
| `depth` (layers) | 6 | FlowDet |
| `num_heads` | 8 | LayoutFlow |
| `mlp_ratio` | 4 | 표준 Transformer |
| `num_queries` | 300 | DiffusionDet |
| `backbone` | ResNet50 | DiffusionDet |
| `inference_steps` | 10 | FlowDet (dynamic NFE) |
| `trajectory` | riemannian | 본 연구 제안 |

---

## 8. 구현 현황

| 파일 | 상태 | 검증 |
|------|------|------|
| `modules.py` | ✅ 완료 | `python model/modules.py` ✓ |
| `backbone.py` | ✅ 완료 | `python model/backbone.py` ✓ |
| `dit.py` | ✅ 완료 | `python model/dit.py` ✓ |
| `head.py` | ✅ 완료 | `python model/head.py` ✓ |
| `trajectory.py` | ✅ 완료 | `python model/trajectory.py` ✓ |
| `loss.py` | ✅ 완료 | `python model/loss.py` ✓ |
| `flow_matching.py` | ✅ 완료 | `python model/flow_matching.py` ✓ |
| `__init__.py` | ✅ 완료 | `build_model({})` |

---

## 9. 다음 구현 계획

### Phase A — 학습 스크립트 (`script/train.py`)

```
config 로드 (configs/base.yaml)
     │
build_dataset + build_dataloader  (dataset/)
     │
build_model()  → RiemannianFlowDet
     │
optimizer (AdamW, lr=1e-4, wd=1e-4)
lr_scheduler (cosine)
     │
train loop:
  batch → model.forward_train(images, boxes_gt_list)
        → loss["loss"].backward()
        → optimizer.step()
     │
checkpoint save (utils/checkpoint.py)
TensorBoard logging (utils/logger.py)
```

**주요 TODO:**
- [ ] `script/train.py` — 학습 루프
- [ ] `utils/config.py` — ConfigArgParse 기반 config 빌더
- [ ] `utils/checkpoint.py` — save/load
- [ ] `utils/logger.py` — TensorBoard 래퍼
- [ ] `configs/base.yaml` — 기본 하이퍼파라미터

### Phase B — 평가 (`script/eval.py`)

```
forward_inference(images, num_steps=10)
     │  boxes [B,Q,4] cxcywh
     │
NMS (torchvision.ops.nms)  or  top-K
     │
COCO mAP (pycocotools)
VOC mAP  (torchvision.ops.box_map)
```

**주요 TODO:**
- [ ] `script/eval.py` — evaluation 루프
- [ ] `utils/metrics.py` — mAP 계산 래퍼
- [ ] NMS / confidence head 결정 (BoxHead에 confidence branch 추가 검토)

### Phase C — 실험 변형

| 실험 | 변경 내용 | 목적 |
|------|----------|------|
| Exp 1 | `LinearTrajectory` vs `RiemannianTrajectory` | 제안 방법 ablation |
| Exp 2 | RoPE off / on | RoPE 기여도 |
| Exp 3 | depth 4 / 6 / 8 | capacity scaling |
| Boosting | num_steps 1→5→10 | dynamic NFE 효과 |

---

## 10. 핵심 설계 원칙 (불변)

```
1. 모델 내부 박스 연산은 항상 state space [cx, cy, log_w, log_h]
2. boxes_gt 입력 직후 cxcywh_to_state() 호출
3. 추론 출력 직후 state_to_cxcywh() 호출
4. head_dim % 4 == 0  (2D RoPE 제약)
5. Loss는 masked positions만 계산 (padding 제외)
6. RiemannianTrajectory의 b₀, b₁ 모두 state space 기준
```
