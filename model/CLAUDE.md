# model/ — CLAUDE.md

## 파일 구조 및 역할

| 파일 | 클래스 | 역할 |
|------|--------|------|
| `backbone.py` | `Backbone` | 이미지 → 시각적 feature 추출 (ResNet + FPN) |
| `modules.py` | `SinusoidalEmbedding`, `BoxEmbedding`, `ImageProjection`, `MLP` | 공유 primitive 블록 |
| `dit.py` | `DiTBlock`, `FlowDiT` | image-conditioned vector field 예측 (Diffusion Transformer) |
| `head.py` | `BoxHead` | DiT 출력 → 벡터장 `v̂_t [B,N,4]` + confidence |
| `flow_matching.py` | `FlowMatcher` | 학습 루프: `t` 샘플링, `b_t` 보간, loss 계산 |
| `trajectory.py` | `LinearTrajectory`, `RiemannianTrajectory` | `b_t = γ(t; b₀, b₁)`, `u_t*` 계산 |
| `loss.py` | `FlowMatchingLoss` | `‖v̂_t − u_t*‖²` |
| `__init__.py` | — | `build_model(config)` 진입점 |

---

## Box State Space 규약

```
입력 포맷    : [cx, cy, w, h]       normalized cxcywh ∈ (0,1)
모델 내부    : [cx, cy, log_w, log_h]   ∈ ℝ⁴  (ℝ² × ℝ₊²)
변환 함수    : dataset/box_ops.py
  cxcywh_to_state(boxes)  →  log-scale
  state_to_cxcywh(states) →  exp-scale back
```

**절대 규칙**: 모델 내부에서 `w`, `h`를 raw 값으로 직접 연산하지 않는다. 반드시 log-scale state를 사용한다.

---

## 텐서 shape 규약

| 심볼 | 의미 |
|------|------|
| `B` | batch size |
| `N` | box 수 (가변, collate 시 list로 처리) |
| `D` | model hidden dim |
| `H, W` | 이미지 height, width |
| `t` | flow time ∈ [0, 1], shape `[B]` |

---

## 모듈 작성 규칙

**docstring 필수** — 모든 `forward`에:
```python
def forward(self, ...):
    """
    Purpose: ...
    Inputs:
        x: [B, D]  float32
    Outputs:
        y: [B, D]  float32
    """
```

**`__main__` 블록 필수** — 각 파일 하단에 shape assert + sanity check.

---

## 의존 관계 (단방향)

```
backbone.py
modules.py
    ↓
dit.py  ←  modules.py
    ↓
head.py ←  modules.py
    ↓
flow_matching.py  ←  trajectory.py
                  ←  loss.py
    ↓
__init__.py  (build_model)
```

순환 import 금지. `flow_matching.py`가 `backbone`, `dit`, `head`를 조립하는 최상위.

---

## 학습 흐름 요약

```
image [B,3,H,W]  +  boxes_gt [B,N,4]
  │                      │
  │              cxcywh_to_state → b₁ [B,N,4]
  │              b₀ ~ N(0,I)
  │              t  ~ U[0,1]
  │              trajectory: b_t, u_t*
  ↓                      ↓
Backbone → feat    FlowDiT+BoxHead → v̂_t
                          ↓
                   loss = ‖v̂_t − u_t*‖²
```

## 추론 흐름 요약

```
b₀ ~ N(0,I)  →  ODE solver (Euler/RK4, NFE steps)
  각 step: v̂_t = FlowDiT+BoxHead(b_t, t, feat)
           b_{t+Δt} = b_t + Δt · v̂_t
→ b₁_pred  →  state_to_cxcywh  →  boxes
```
