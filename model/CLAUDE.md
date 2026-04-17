# model/CLAUDE.md

모델 설계 전반 가이드. 이 디렉토리 파일 수정 시 반드시 읽기.

**참조 논문:** LayoutFlow: Flow Matching for Layout Generation (ECCV 2024, arXiv 2403.18187)
**참조 GitHub:** https://github.com/JulianGuerreiro/LayoutFlow

> LayoutFlow에서 가져온 것: flow matching 학습 방식 + AdaLN-Zero time conditioning
> 우리가 추가한 것: image backbone + RoPE + image-box fusion

---

## 전체 Forward Pass — Shape 다이어그램

> 기준값: B=2, N=100, H=W=800, d_model=256

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 images   [B, 3, 800, 800]
 boxes    [B, N, 4]          ← normalized cxcywh (GT)
 t        [B]                ← U(0,1) at train / 0→1 at infer


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BRANCH A : IMAGE  (backbone.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 옵션 1: ResNet-50 + FPN
 ─────────────────────────────────────────────────────────────
 [B, 3, 800, 800]
      │
      ▼  stem
      │   conv(3→64, k=7, s=2, p=3) → BN → ReLU   [B, 64, 400, 400]
      │   MaxPool(k=3, s=2, p=1)                   [B, 64, 200, 200]
      │
      ▼  layer1 (C2) — Bottleneck × 3, stride=1
      │   conv(64→64, k=1) → BN → ReLU
      │   conv(64→64, k=3) → BN → ReLU
      │   conv(64→256, k=1) → BN
      │   + shortcut(conv 64→256, k=1)
 [B, 256, 200, 200]
      │
      ▼  layer2 (C3) — Bottleneck × 4, stride=2 (첫 블록)
      │   conv(256→128, k=1) → BN → ReLU
      │   conv(128→128, k=3, s=2) → BN → ReLU
      │   conv(128→512, k=1) → BN
      │   + shortcut(conv 256→512, k=1, s=2)
 [B, 512, 100, 100]
      │
      ▼  layer3 (C4) — Bottleneck × 6, stride=2 (첫 블록)  ← 여기서 추출
      │   conv(512→256, k=1) → BN → ReLU
      │   conv(256→256, k=3, s=2) → BN → ReLU
      │   conv(256→1024, k=1) → BN
      │   + shortcut(conv 512→1024, k=1, s=2)
 [B, 1024, 50, 50]
      │
      ▼  FPN lateral conv: conv(1024→256, k=1)
 P4 [B, 256, 50, 50]
      │
      ▼  flatten (50×50 → 2500)
 [B, 2500, 256]
      │
      ▼  Linear(256, d_model=256)
 image_tokens  [B, S=2500, 256]

 ─────────────────────────────────────────────────────────────
 옵션 2: DINOv2 (ViT-B/14)  ← frozen feature extractor
 ─────────────────────────────────────────────────────────────
 [B, 3, 448, 448]   ← 입력 크기를 patch_size=14의 배수로 맞춤
      │
      ▼  Patch Embed: conv(3→768, k=14, s=14)   (non-overlapping)
 [B, 32×32, 768] = [B, 1024, 768]   ← patch tokens (CLS 제외)
      │              ↑ 448/14 = 32
      ▼  Transformer (ViT-B: 12 layers, nhead=12, d=768)
         각 layer: Self-Attn + FFN + LayerNorm
 [B, 1024, 768]   ← 마지막 layer 출력 (patch tokens만)
      │
      ▼  Linear(768, d_model=256)   ← projection
 image_tokens  [B, S=1024, 256]

 ※ DINOv2 vs ResNet 비교
 ┌───────────────────────────────────────────────────────────┐
 │ ResNet-50+FPN   S=2500  학습 가능  로컬 feature 강함     │
 │ DINOv2 ViT-B    S=1024  frozen 권장  global context 강함 │
 │                         semantic feature 풍부             │
 │                         fine-tuning 시 학습 불안정 가능  │
 └───────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BRANCH B : BOX NOISE  (trajectory.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 boxes [B, N, 4]  normalized cxcywh
      │
      ▼  cxcywh_to_state()   (dataset/box_ops.py)
 b1   [B, N, 4]  Riemannian state  [cx, cy, log_w, log_h]
                  ↑ ℝ²  ×  ℝ₊² (log 변환)  → 통합 ℝ⁴ 공간

 b0 ~ N(0, I)  [B, N, 4]          ← Gaussian noise

 t    [B]  →  reshape [B, 1, 1]

 b_t = (1-t) * b0 + t * b1  →  [B, N, 4]   ← 모델 입력
 u_t =          b1 - b0      →  [B, N, 4]   ← loss target


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BOX EMBEDDING  (modules.py)  — Riemannian 통합 공간
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 b_t  [B, N, 4]   ← [cx, cy, log_w, log_h] 분리하지 않음
      │              ℝ² × ℝ₊² 를 통합 Riemannian state로 취급
      │
      ▼  Linear(4, d_model=256)
 x    [B, N, 256]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BRANCH C : TIME EMBEDDING — RoPE  (modules.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 t  [B]    ← raw scalar ∈ [0, 1]
      │
      ▼  Sinusoidal Encoding
         dim i: sin(t · 10000^(-2i/d)),  cos(t · 10000^(-2i/d))
 t_emb  [B, d_model=256]
      │
      ▼  MLP: Linear(256,256) → SiLU → Linear(256,256)
 t_emb  [B, 256]
      │
      └──▶ AdaLN-Zero 내부에서 사용 (각 DiTBlock마다)

 ※ RoPE는 additive embedding 대신 Q, K 회전에 적용
    Self-Attention 내부에서:
      Q, K 를 head_dim 단위로 쌍으로 묶어 t 기반 회전
      q' = R(t) · q   →   [B, heads, N, head_dim]
      k' = R(t) · k   →   [B, heads, N, head_dim]
      attn = softmax(q'·k'ᵀ / √d)
    → t 정보가 attention score에 직접 반영


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DiT BLOCK × 4  (dit.py)  — AdaLN-Zero + RoPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 입력:
   x            [B, N=100,  256]   ← box tokens
   image_tokens [B, S=2500, 256]   ← image patch tokens
   t_emb        [B, 256]           ← sinusoidal time embedding

 ┌──────────────────────────────────────────────────────────┐
 │ DiTBlock (AdaLN-Zero)                                    │
 │                                                          │
 │  t_emb [B, 256]                                         │
 │       │                                                  │
 │       ▼  Linear(256, 6×256=1536)                        │
 │       ▼  chunk(6)                                       │
 │  s1,b1,g1,  s2,b2,g2,  s3,b3,g3  각 [B, 256]          │
 │  (self-attn) (cross-attn) (FFN)                         │
 │  g1,g2,g3: zero init → 학습 초기 block = identity      │
 │                                                          │
 │  ── Self-Attention (box ↔ box) ─────────────────────── │
 │  x_norm = (1+s1)*LayerNorm(x) + b1   [B, N, 256]       │
 │  Q = x_norm · Wq  [B, N, 256]                          │
 │  K = x_norm · Wk  [B, N, 256]   RoPE 적용              │
 │  V = x_norm · Wv  [B, N, 256]   ↓                      │
 │  Q' = R(t)·Q  [B, 8, N=100, 32]                        │
 │  K' = R(t)·K  [B, 8, N=100, 32]                        │
 │  attn = softmax(Q'·K'ᵀ / √32)  [B, 8, N, N]            │
 │  out  = attn · V                [B, N, 256]             │
 │  x = x + g1 * out               [B, N, 256]             │
 │                                                          │
 │  ── Cross-Attention (box ↔ image) ──────────────────── │
 │  x_norm = (1+s2)*LayerNorm(x) + b2   [B, N, 256]       │
 │  Q = x_norm · Wq    [B, N,    256]                      │
 │  K = image_tokens · Wk  [B, S, 256]                     │
 │  V = image_tokens · Wv  [B, S, 256]                     │
 │  attn = softmax(Q·Kᵀ / √32)  [B, 8, N=100, S=2500]     │
 │  out  = attn · V              [B, N, 256]                │
 │  x = x + g2 * out             [B, N, 256]               │
 │                                                          │
 │  ── FFN ────────────────────────────────────────────── │
 │  x_norm = (1+s3)*LayerNorm(x) + b3   [B, N, 256]       │
 │  out = Linear(256,1024) → GELU → Linear(1024,256)      │
 │        [B, N, 1024]  →  [B, N, 256]                    │
 │  x = x + g3 * out   [B, N, 256]                        │
 └──────────────────────────────────────────────────────────┘
        × 4 반복


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 HEAD  (head.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 x [B, N, 256]
      │
      ▼  LayerNorm(256)
      ▼  Linear(256, 4)
 v_pred [B, N, 4]    ← 예측된 vector field


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LOSS  (loss.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 v_pred [B, N, 4]
 u_t    [B, N, 4]   (= b1 - b0)

 flow_loss   = MSE(v_pred, u_t)               [scalar]
 geom_l1     = L1(v_pred, u_t)               [scalar]
 giou_loss   = GIoU(b_t + v_pred, b1)        [scalar]

 L = 1.0 * flow_loss + 0.2 * geom_l1 + 1.0 * giou_loss


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INFERENCE — Euler ODE  (flow_matching.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 b0 ~ N(0, I)  [B, N, 4]

 for i in range(T=100):
     t = i / T                          [B]
     v = model(b_i, image_tokens, t)    [B, N, 4]
     b_{i+1} = b_i + (1/T) * v         [B, N, 4]

 b_T [B, N, 4]
      │
      ▼  state_to_cxcywh()
 [B, N, 4]  normalized cxcywh


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 IMAGE-BOX FUSION 방식 고민
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 현재: Cross-Attention (DETR 스타일)
   Q=box [B,N,256],  K=V=image [B,S,256]
   → box가 image를 일방향으로 참조
   → image는 box를 모름

 대안 비교:
 ┌───────────────────────────────────────────────────────────┐
 │                                                           │
 │ A. Cross-Attention (현재)                                │
 │    box → image 단방향                                    │
 │    attn [B, 8, N, S] = [B,8,100,2500]                   │
 │    + DETR에서 검증된 구조                                │
 │    - image는 box 위치를 모름                             │
 │                                                           │
 │ B. Joint Self-Attention  (concat)                        │
 │    [image_tokens; box_tokens] concat                     │
 │    → [B, S+N, 256] = [B, 2600, 256]                     │
 │    → Self-Attention 전체에 걸쳐                          │
 │    attn [B, 8, 2600, 2600]                               │
 │    + 양방향 (image ↔ box 모두 참조)                     │
 │    - 비용: (S+N)² >> N² + N·S                           │
 │                                                           │
 │ C. FiLM (Feature-wise Linear Modulation)                 │
 │    image feature → GAP → [B, 256]                       │
 │    → Linear(256, 2×256) → scale, shift                  │
 │    → box token에 AdaLN처럼 적용                          │
 │    + cross-attn 없이 가벼움                              │
 │    - global image 정보만 반영, 공간 정보 손실            │
 │                                                           │
 │ D. Decoupled Cross-Attention  (이미지 위치 인식 강화)    │
 │    box의 현재 위치에서 image feature RoI Align           │
 │    → 해당 영역 feature만 K, V로 사용                    │
 │    + 공간적으로 relevant한 feature만 참조               │
 │    - b_t가 noisy해서 t 작을 때 RoI 위치 부정확          │
 │                                                           │
 │ → 현재 A(Cross-Attention)로 시작, B는 ablation 후보     │
 └───────────────────────────────────────────────────────────┘
```

---

## 주요 Shape 요약표

| 단계 | Tensor | Shape |
|------|--------|-------|
| 입력 이미지 | `images` | `[B, 3, 800, 800]` |
| ResNet C4 | feature map | `[B, 1024, 50, 50]` |
| FPN P4 / DINOv2 | `image_tokens` | `[B, 2500, 256]` / `[B, 1024, 256]` |
| GT box → state | `b1` | `[B, N, 4]` |
| Noisy box | `b_t` | `[B, N, 4]` |
| Box embed | `x` | `[B, N, 256]` |
| Time embed | `t_emb` | `[B, 256]` |
| AdaLN-Zero params | `s,b,g ×3` | `[B, 256]` × 9 per block |
| RoPE Q, K | — | `[B, 8, N, 32]` |
| Self-attn scores | — | `[B, 8, 100, 100]` |
| Cross-attn scores | — | `[B, 8, 100, 2500]` |
| DiTBlock 출력 | `x` | `[B, N, 256]` |
| Vector field | `v_pred` | `[B, N, 4]` |

---

## 설계 확정 사항

| 항목 | 결정 |
|------|------|
| Backbone | ResNet-50 + FPN P4 (기본) / DINOv2 ViT-B (실험) |
| Box embedding | Linear(4, 256) 통합 — Riemannian state 분리 안 함 |
| Time conditioning | Sinusoidal → MLP + RoPE (Q,K 회전) |
| Normalization | AdaLN-Zero (6 params: s,b,g × attn/cross/FFN) |
| Image-box fusion | Cross-Attention (기본) |
| FPN 스케일 | P4 단일 (S=2500) |
| N (num queries) | 100 |
| d_model | 256 |
| num_layers | 4 |
