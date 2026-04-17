# LayoutFlow — 원본 모델 아키텍처

**논문:** LayoutFlow: Flow Matching for Layout Generation (ECCV 2024, arXiv 2403.18187)
**GitHub:** https://github.com/JulianGuerreiro/LayoutFlow
**기본 backbone:** `LayoutDMBackbone` (AdaLN 기반 custom TransformerEncoder)

> 이미지 입력 없음. 순수 layout (box + category) 생성 모델.

---

## 전체 Forward Pass — Shape 다이어그램

> 기준값: B=2, N=20 (N_max), d_model=512, latent_dim=128, B_cat=3 (PubLayNet 5class → ceil(log2(5))=3)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 geom       [B, N, 4]      ← (cx, cy, w, h) ∈ [-1, 1]  (GT → preprocessed)
 attr       [B, N, B_cat]  ← Analog Bits encoded category  (B_cat=3)
 cond_flags [B, N, 7]      ← conditioning mask  (4 geom dims + 3 attr dims)
                              1 = free (generate), 0 = conditioned (given)
 t          [B]            ← U(0,1) at train / 0→1 at infer


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FLOW MATCHING — Noisy Sample 생성  (trajectory.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 x1 = concat(geom, attr)  →  [B, N, 7]   ← GT (preprocessed)
 x0 ~ N(0, I)             →  [B, N, 7]   ← Gaussian noise

 t reshape → [B, 1, 1]

 x_t = (1-t) * x0 + t * x1  →  [B, N, 7]   ← 모델 입력
 u_t =          x1 - x0      →  [B, N, 7]   ← loss target

 ※ conditioned dims는 x1 값으로 대체:
    x_t = (1 - cond_flags) * x1 + cond_flags * x_t


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BOX + CATEGORY EMBEDDING  (LayoutDMBackbone, seq_type='stacked')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ┌─ Geometry Branch ──────────────────────────────────────────┐
 │                                                            │
 │  geom [B, N, 4]                                           │
 │       │                                                    │
 │       ├── center (cx, cy)  [B, N, 2]                      │
 │       │        │                                           │
 │       │        ▼  Linear(2, 64)                           │
 │       │   [B, N, 64]                                      │
 │       │        │  +  cond_enc_center                      │
 │       │             cond_flags[:,:,:2].sum(-1)  [B, N]    │
 │       │             → Embedding(3, 64)          [B, N, 64]│
 │       │   center_emb  [B, N, 64]                          │
 │       │                                                    │
 │       └── size (w, h)  [B, N, 2]                          │
 │                │                                           │
 │                ▼  Linear(2, 64)                           │
 │           [B, N, 64]                                      │
 │                │  +  cond_enc_size                        │
 │                     cond_flags[:,:,2:4].sum(-1)  [B, N]  │
 │                     → Embedding(3, 64)           [B, N, 64]
 │           size_emb  [B, N, 64]                            │
 │                                                            │
 │  cat(center_emb, size_emb)  →  geom_emb  [B, N, 128]     │
 └────────────────────────────────────────────────────────────┘

 ┌─ Category Branch ──────────────────────────────────────────┐
 │                                                            │
 │  attr [B, N, 3]   ← Analog Bits                           │
 │       │                                                    │
 │       ▼  Linear(3, 128)                                   │
 │  [B, N, 128]                                              │
 │       │  +  cond_enc_attr                                  │
 │            cond_flags[:,:,-1]  [B, N]  (0 or 1)          │
 │            → Embedding(2, 128)  [B, N, 128]               │
 │  attr_emb  [B, N, 128]                                    │
 └────────────────────────────────────────────────────────────┘

 pack(geom_emb, attr_emb)  →  [B, N, 256]
       │
       ▼  elem_embed: Linear(256, d_model=512)
 x  [B, N, 512]

 ※ cond_enc_* 란?
 ┌──────────────────────────────────────────────────────────────┐
 │ cond_enc_center = nn.Embedding(3, 64)                       │
 │                                                             │
 │ cond_flags[:,:,:2]  (cx, cy 각각 0 or 1)                   │
 │   .sum(-1)  →  [B, N]  값: 0 / 1 / 2                      │
 │                                                             │
 │   sum=0  →  Embedding[0]  "cx, cy 둘 다 conditioned"       │
 │   sum=1  →  Embedding[1]  "cx, cy 중 하나만 free"          │
 │   sum=2  →  Embedding[2]  "cx, cy 둘 다 free"              │
 │                                                             │
 │   → [B, N, 64]  를 center_emb에 더함                       │
 │                                                             │
 │ 역할: 좌표값(Linear)과 conditioning 상태(Embedding)을       │
 │       embedding 단계에서 합산 → transformer 이전에          │
 │       모델이 "이 token을 생성해야 하는지" 인지하게 함       │
 │                                                             │
 │ cond_enc_size = nn.Embedding(3, 64)   ← size (w,h) 동일    │
 │ cond_enc_attr = nn.Embedding(2, 128)  ← attr는 1dim이라    │
 │                                          sum=0 or 1만 존재  │
 └──────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TIME EMBEDDING — AdaLayerNorm 내부  (simple_transformer.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 t  [B]    ← raw scalar ∈ [0, 1]
      │
      ▼  Rearrange('b -> b 1')
 [B, 1]
      ▼  Linear(1, 256)  →  ReLU
 [B, 256]
      ▼  Linear(256, 512)  →  SiLU
 [B, 512]
      ▼  Linear(512, 1024)  →  unsqueeze(1)
 [B, 1, 1024]
      ▼  chunk(dim=2)
 scale  [B, 1, 512]
 shift  [B, 1, 512]
      │
      └──▶  각 Block의 AdaLayerNorm에서 사용
            LayerNorm(x) * (1 + scale) + shift

 ※ Time Embedding 방식 비교 (참고)
 ┌──────────────────────────────────────────────────────────────┐
 │ A. Raw scalar MLP (LayoutFlow 방식 — 위)                    │
 │    t [B] → Linear(1,d//2) → ReLU → Linear → SiLU → [B,d]  │
 │    단순하지만 표현력 낮음                                    │
 │                                                              │
 │ B. Sinusoidal (DiT / DDPM 표준)                             │
 │    t [B] → sin/cos 주파수 인코딩 → [B, d]                  │
 │    → MLP → [B, d]                                           │
 │    고주파~저주파 동시 표현 → 표현력 높음                    │
 │    DiT, Stable Diffusion, DDPM 등 업계 표준                 │
 │                                                              │
 │ C. RoPE (Rotary Position Embedding)                         │
 │    원래 LLM 위치 인코딩용 (RoFormer, LLaMA 등)             │
 │    Q, K 벡터를 t에 따라 회전 행렬로 변환                   │
 │    q' = R(t) · q,  k' = R(t) · k                           │
 │    → attention score에 상대적 위치 정보가 자연스럽게 반영  │
 │    additive가 아닌 multiplicative 방식                      │
 │    최근 일부 flow/diffusion 모델에서 time에도 적용 시도     │
 │    (절대 시간 대신 상대 시간 관계를 attention에 직접 주입)  │
 └──────────────────────────────────────────────────────────────┘

 ※ Normalization + Conditioning 방식 비교
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │ A. AdaIN  (Adaptive Instance Normalization)                 │
 │    style transfer 기원 (Huang & Belongie 2017)              │
 │                                                              │
 │    condition → scale [B,C], shift [B,C]   ← 2 params       │
 │                                                              │
 │    out = scale * InstanceNorm(x) + shift                    │
 │          채널별 평균/분산 제거 후 condition으로 재조정      │
 │                                                              │
 │    x   [B, C, H, W]                                        │
 │         → InstanceNorm (H,W 축 정규화)                     │
 │         → * scale + shift                                   │
 │    out [B, C, H, W]                                        │
 │                                                              │
 ├──────────────────────────────────────────────────────────────┤
 │                                                              │
 │ B. AdaLN  (Adaptive Layer Normalization)                    │
 │    LayoutFlow 사용 방식                                      │
 │                                                              │
 │    t_emb → Linear(d, 2d) → chunk                           │
 │    scale [B, 1, d],  shift [B, 1, d]      ← 2 params       │
 │                                                              │
 │    out = (1 + scale) * LayerNorm(x) + shift                 │
 │          (1 + scale): scale=0 이면 일반 LayerNorm과 동일    │
 │                       → residual 초기화 안정성              │
 │                                                              │
 │    x   [B, N, d]                                           │
 │         → LayerNorm (d 축 정규화)                          │
 │         → * (1+scale) + shift                               │
 │    out [B, N, d]                                           │
 │                                                              │
 ├──────────────────────────────────────────────────────────────┤
 │                                                              │
 │ C. AdaLN-Zero  (DiT 논문 방식)                              │
 │    DiT (Peebles & Xie 2023) 에서 제안                       │
 │                                                              │
 │    t_emb → Linear(d, 6d) → chunk 6개    ← 6 params         │
 │                                                              │
 │    shift_1, scale_1, gate_1  ← self-attention 용           │
 │    shift_2, scale_2, gate_2  ← FFN 용                      │
 │                                                              │
 │    [Self-Attention 파트]                                    │
 │    x_norm = (1+scale_1) * LayerNorm(x) + shift_1           │
 │    x = x + gate_1 * SelfAttn(x_norm)                       │
 │                  ↑                                          │
 │              zero init → 학습 초기 block = identity        │
 │                                                              │
 │    [FFN 파트]                                               │
 │    x_norm = (1+scale_2) * LayerNorm(x) + shift_2           │
 │    x = x + gate_2 * FFN(x_norm)                            │
 │                  ↑                                          │
 │              zero init → 학습 초기 block = identity        │
 │                                                              │
 │    gate_1, gate_2 를 0으로 초기화                           │
 │    → 학습 초기에 전체 블록이 identity function             │
 │    → gradient flow 안정, 깊은 모델도 학습 용이             │
 │                                                              │
 ├──────────────────────────────────────────────────────────────┤
 │                                                              │
 │ 요약                                                         │
 │                                                              │
 │  방식         params   정규화축   gate   사용처             │
 │  AdaIN          2      H,W (공간)   X    style transfer     │
 │  AdaLN          2      d (채널)     X    LayoutFlow         │
 │  AdaLN-Zero     6      d (채널)     O    DiT (표준)         │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TRANSFORMER ENCODER × 4  (custom Block with AdaLN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 입력:
   x  [B, N=20, 512]
   t  [B]

 ┌──────────────────────────────────────────────────────────┐
 │ Block (pre-norm, norm_first=True)                        │
 │                                                          │
 │  x [B, N, 512]                                          │
 │       │                                                  │
 │       ▼  AdaLayerNorm(x, t)                             │
 │       │   LayerNorm(x) * (1 + scale) + shift            │
 │  x_norm [B, N, 512]                                     │
 │       │                                                  │
 │       ▼  Self-Attention  (layout element ↔ element)     │
 │       Q = K = V = x_norm  [B, N, 512]                   │
 │       → [B, 8heads, N=20, head_dim=64]                  │
 │       → attn scores  [B, 8, N=20, N=20]                 │
 │       → [B, N, 512]                                     │
 │       │                                                  │
 │       ▼  residual add                                    │
 │  x [B, N, 512]                                          │
 │       │                                                  │
 │       ▼  AdaLayerNorm(x, t)                             │
 │  x_norm [B, N, 512]                                     │
 │       │                                                  │
 │       ▼  FFN                                            │
 │       Linear(512, 2048) → GELU → Linear(2048, 512)     │
 │       [B, N, 2048]  →  [B, N, 512]                     │
 │       │                                                  │
 │       ▼  residual add                                    │
 │  x [B, N, 512]                                          │
 └──────────────────────────────────────────────────────────┘
        × 4 반복

 LayerNorm(512)  →  x [B, N, 512]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT HEAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 x [B, N, 512]
      │
      ▼  Linear(512, 4 + B_cat=3)
 v_pred  [B, N, 7]    ← 예측된 vector field (geom 4 + attr 3)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 v_pred     [B, N, 7]
 u_t        [B, N, 7]   (= x1 - x0)
 cond_flags [B, N, 7]   (1=free, 0=conditioned)

 L_CFM     = MSE(cond_flags * v_pred,  cond_flags * u_t)       [scalar]
 L_geo_L1  = L1 (cond_flags[:,:,:4] * v_pred[:,:,:4],
                  cond_flags[:,:,:4] * u_t[:,:,:4])            [scalar]

 L = L_CFM + 0.2 * L_geo_L1


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INFERENCE — Euler ODE (T=100 steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 x0 ~ N(0, I)  [B, N, 7]
 conditioned dims → x1 값으로 초기화

 for i in range(T=100):
     t = i / T                              [B]
     v = model(x_i, cond_flags, t)          [B, N, 7]
     x_{i+1} = x_i + (1/T) * v             [B, N, 7]
     x_{i+1} = (1-cond_flags)*x1 + cond_flags*x_{i+1}  ← conditioned dims 고정

 x_T [B, N, 7]
      ├── geom [B, N, 4]  → (x+1)/2 → [0,1]  →  (cx, cy, w, h)
      └── attr [B, N, 3]  → threshold(0.5) → Analog Bits decode → class label
```

---

## 주요 Shape 요약표

| 단계 | Tensor | Shape |
|------|--------|-------|
| 입력 geometry | `geom` | `[B, N, 4]` |
| 입력 category | `attr` | `[B, N, 3]` |
| conditioning mask | `cond_flags` | `[B, N, 7]` |
| noisy sample | `x_t` | `[B, N, 7]` |
| center embed | — | `[B, N, 64]` |
| size embed | — | `[B, N, 64]` |
| attr embed | — | `[B, N, 128]` |
| pack(geom, attr) | — | `[B, N, 256]` |
| elem_embed 출력 | `x` | `[B, N, 512]` |
| time → scale/shift | — | `[B, 1, 512]` each |
| self-attn scores | — | `[B, 8, 20, 20]` |
| Block 출력 (×4) | `x` | `[B, N, 512]` |
| vector field | `v_pred` | `[B, N, 7]` |

---

## Analog Bits — Category Encoding

```
class label (int)  →  B_cat = ceil(log2(num_class)) bits  →  continuous [-1, 1]

예시 (5 classes, B_cat=3):
  class 0  →  [0, 0, 0]  →  [-1, -1, -1]
  class 1  →  [1, 0, 0]  →  [ 1, -1, -1]
  class 2  →  [0, 1, 0]  →  [-1,  1, -1]
  class 3  →  [1, 1, 0]  →  [ 1,  1, -1]
  class 4  →  [0, 0, 1]  →  [-1, -1,  1]

decode: threshold at 0  →  binary  →  weighted sum
```

---

## Conditioning 시나리오 (cond_flags 구성)

| 모드 | cx,cy | w,h | category | 설명 |
|------|-------|-----|----------|------|
| **Uncond** | 1 | 1 | 1 | 전체 생성 |
| **Gen-Type** | 1 | 1 | 0 | category 주어짐, box 생성 |
| **Gen-TypeSize** | 1 | 0 | 0 | category+size 주어짐, center 생성 |
| **Elem-Compl** | 0 | 0 | 0 | 일부 element 주어짐, 나머지 생성 |

학습 시 배치 내에서 4가지 모드를 균등하게 랜덤 샘플링.

---

## 하이퍼파라미터 (LayoutFlow default)

| 파라미터 | 값 |
|---------|-----|
| `d_model` | 512 |
| `latent_dim` | 128 |
| `nhead` | 8 |
| `num_layers` | 4 |
| `dim_feedforward` | 2048 |
| `dropout` | 0.1 |
| `N_max` | 20 |
| `trajectory` | Linear |
| `T (inference steps)` | 100 |
| `optimizer` | AdamW lr=5e-4 |
| `L_geo_L1 weight` | 0.2 |
