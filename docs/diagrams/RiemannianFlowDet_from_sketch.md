# RiemannianFlowDet — Model Diagram (from hand-drawn sketch)

> Source: `model/image.png` (hand-drawn architecture sketch, cross-validated with code)  
> Analysis mode: image → text + code 교차검증  
> Parameters: 33.89 M (trainable: 33.89 M)

---

## Summary

| 항목 | 값 |
|------|-----|
| 소스 | model/image.png (hand-drawn) |
| 모델명 | RiemannianFlowDet |
| 총 파라미터 | 33.89 M |
| Trainable | 33.89 M |
| 분석 모드 | image → text (코드 교차검증) |
| Conv 계층 수 | 33 |
| Linear 계층 수 | 10 per DiTBlock (×6 = 48 total) |
| Activation | ReLU×16, GELU×6, SiLU×7 |
| Norm 계층 수 | 26 |
| 사용자 정의 모듈 | 39 |
| State space | [cx, cy, log_w, log_h] ∈ ℝ² × ℝ₊² |
| Trajectory | Riemannian geodesic (linear in log-space) |
| 최대 깊이 | 8 |

---

## Training Flow

```
image [B,3,H,W]              boxes_gt  list[[Ni,4]] cxcywh
     │                              │
     │                    cxcywh_to_state
     │                              │
     │                         b₁ [B,N,4]  ← state space [cx,cy,log_w,log_h]
     │                         b₀ ~ N(0,I) ← noise (state space)
     │                         t  ~ U[0,1]
     │                              │
     │                    trajectory.sample(b₁, t)
     │                    b_t = (1-t)·b₀ + t·b₁  [B,N,4]  ← interpolated state
     │                    u_t* = b₁ - b₀          [B,N,4]  ← constant vector field
     │                              │
     ↓                              ↓
FlowDiT(image, b_t, t) → box_tokens [B,N,D]
BoxHead(box_tokens)    → v̂_t        [B,N,4]
Loss = MSE(v̂_t[mask], u_t*[mask])  ← valid boxes only
```

---

## Model Tree

```
[M] RiemannianFlowDet                              [B,3,H,W] + list[[Ni,4]] → {"loss": scalar}
│                                                                (train) / [B,Q,4] (inference)
│
├── [C] FlowDiT                                    [B,3,H,W]+[B,N,4]+[B] → [B,N,D]
│   │
│   ├── [C] FPNBackbone  (Extractor, 파란 박스)    [B,3,H,W] → [B,L,D]  (L=1344 @ 256px)
│   │   ├── [C] ResNet50 stem
│   │   │   ├── [K] Conv2d(3→64, 7×7, s=2, p=3)   [B,3,H,W]      → [B,64,H/2,W/2]
│   │   │   ├── [N] BatchNorm2d(64)
│   │   │   ├── [A] ReLU
│   │   │   └── [P] MaxPool2d(3×3, s=2)            [B,64,H/2,W/2] → [B,64,H/4,W/4]
│   │   ├── [C] layer1: Bottleneck ×3              [B,64,H/4,W/4]     → [B,256,H/4,W/4]
│   │   ├── [C] layer2: Bottleneck ×4              [B,256,H/4,W/4]    → [B,512,H/8,W/8]
│   │   ├── [C] layer3: Bottleneck ×6              [B,512,H/8,W/8]    → [B,1024,H/16,W/16]
│   │   ├── [C] layer4: Bottleneck ×3              [B,1024,H/16,W/16] → [B,2048,H/32,W/32]
│   │   └── [C] FPN (3 scales → D=256)             multi-scale → [B,L,256]  (flatten+cat)
│   │       ├── [K] inner_block×3: Conv2d(→256, 1×1)
│   │       └── [K] layer_block×3: Conv2d(256→256, 3×3)
│   │
│   ├── [C] BoxEmbedding  (box_s → token)          [B,N,4] → [B,N,D]
│   │   ├── [L] Linear(4→D)
│   │   └── [N] LayerNorm(D)
│   │
│   ├── [C] SinusoidalEmbedding  (time block)      [B] → [B,D]
│   │   ├── sinusoidal(t): [B] → [B,D]
│   │   └── [C] MLP: Linear(D→4D) → SiLU → Linear(4D→D)
│   │
│   └── [C] DiTBlock  (×6, 빨간 박스)             [B,N,D] → [B,N,D]
│       ├── [N] LayerNorm(D)
│       ├── [C] Self-Attn (box ↔ box) + 2D-RoPE   [B,N,D] → [B,N,D]
│       │   ├── [L] q_proj: Linear(D→D)
│       │   ├── [L] k_proj: Linear(D→D)  + RoPE(cx,cy)
│       │   ├── [L] v_proj: Linear(D→D)
│       │   └── [L] out_proj: Linear(D→D)
│       ├── [N] LayerNorm(D)
│       ├── [C] Cross-Attn (box → img) + 2D-RoPE  [B,N,D]+[B,L,D] → [B,N,D]
│       │   ├── [L] q_proj: Linear(D→D)  + RoPE(box cx,cy)
│       │   ├── [L] k_proj: Linear(D→D)  + RoPE(img spatial grid)
│       │   ├── [L] v_proj: Linear(D→D)
│       │   └── [L] out_proj: Linear(D→D)
│       ├── [C] AdaLN  (timestep conditioning)     [B,N,D] → [B,N,D]
│       │   ├── [N] LayerNorm(D, affine=False)
│       │   └── [L] Linear(D→2D) → (scale, shift)
│       └── [C] MLP                                [B,N,D] → [B,N,D]
│           ├── [L] Linear(D→4D)
│           ├── [A] GELU
│           └── [L] Linear(4D→D)
│
├── [C] BoxHead  (Head, 초록 박스)                 [B,N,D] → [B,N,4]  (v̂_t)
│   ├── [N] LayerNorm(D)
│   ├── [L] Linear(D→D)
│   ├── [A] SiLU
│   └── [L] Linear(D→4)
│
└── [C] FlowMatchingLoss                           v̂_t, u_t* → scalar
    └── MSE(v̂_t[mask] − u_t*[mask])²
```

---

## Inference Flow (ODE)

```
b₀ ~ N(0,I) [B,Q,4]  ──→  Euler ODE (num_steps=10)
  each step:
    t     = i / num_steps           [B]
    v̂_t   = BoxHead(FlowDiT(img, b_t, t))   [B,Q,4]
    b_t+1 = b_t + dt · v̂_t          [B,Q,4]
  ──→  b₁_pred  ──→  state_to_cxcywh  ──→  [B,Q,4] normalized cxcywh
```

---

## Box State Space 규약

```
b ∈ (cx, cy, w, h) ∈ ℝ² × ℝ₊²

입력 포맷    : [cx, cy, w, h]           normalized cxcywh ∈ (0,1)
모델 내부    : [cx, cy, log_w, log_h]   ∈ ℝ⁴
변환 함수    : dataset/box_ops.py
  cxcywh_to_state(boxes)  →  log-scale
  state_to_cxcywh(states) →  exp-scale
```
