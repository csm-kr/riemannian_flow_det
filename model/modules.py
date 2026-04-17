# File: model/modules.py
# Role: 공통 빌딩 블록 — SinusoidalTimeEmbedding, BoxEmbedding, RoPE, AdaLNZero

import math
import torch
import torch.nn as nn


# ────────────────────────────────────────────────
# Sinusoidal Time Embedding
# ────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Purpose: scalar t ∈ [0,1] → d_model 차원 시간 임베딩 (DiT/DDPM 표준).
    Inputs:
        t:   [B], float32 — time ∈ [0, 1]
    Outputs:
        emb: [B, d_model], float32
    """
    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )  # [half]
        args = t[:, None] * freqs[None, :]            # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, d_model]
        return self.mlp(emb)


# ────────────────────────────────────────────────
# Box Embedding
# ────────────────────────────────────────────────

class BoxEmbedding(nn.Module):
    """
    Purpose: box state [cx, cy, log_w, log_h] → d_model embedding.
    ℝ² × ℝ₊² 를 통합 Riemannian state로 취급 — center/size 분리 없음.
    Inputs:
        boxes: [B, N, 4], float32 — log-scale state
    Outputs:
        emb:   [B, N, d_model], float32
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(4, d_model)

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        return self.proj(boxes)


# ────────────────────────────────────────────────
# RoPE (Rotary Position Embedding) — time 기반
# ────────────────────────────────────────────────

class RoPE(nn.Module):
    """
    Purpose: t 기반 Rotary Position Embedding — Q, K 벡터 회전.
    Cross-Attention에서 Q(box, time t)에만 적용 시 attention score에 시간 정보 반영.

    Inputs:
        x: [B, heads, N, head_dim], float32
        t: [B], float32 — time ∈ [0, 1]
    Outputs:
        x_rot: [B, heads, N, head_dim], float32
    """
    def __init__(self, head_dim: int):
        super().__init__()
        assert head_dim % 2 == 0
        half = head_dim // 2
        freqs = 1.0 / (10000.0 ** (
            torch.arange(0, half, dtype=torch.float32) / half
        ))  # [half]
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, heads, N, head_dim = x.shape
        half = head_dim // 2
        angles = t[:, None, None, None] * self.freqs[None, None, None, :]
        angles = angles.expand(B, heads, N, half)     # [B, heads, N, half]
        x1, x2 = x[..., :half], x[..., half:]
        cos, sin = torch.cos(angles), torch.sin(angles)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ────────────────────────────────────────────────
# AdaLN-Zero
# ────────────────────────────────────────────────

class AdaLNZero(nn.Module):
    """
    Purpose: DiT AdaLN-Zero — t_emb 으로 9개 파라미터 생성.
    3 sub-blocks (self-attn, cross-attn, FFN) × (scale, shift, gate) = 9개.
    gate 는 zero-init → 학습 초기 block = identity function.

    Inputs:
        t_emb: [B, d_model], float32
    Outputs:
        list of 9 tensors, 각 [B, 1, d_model]
        순서: s1,b1,g1 (self-attn) / s2,b2,g2 (cross-attn) / s3,b3,g3 (FFN)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(d_model, 9 * d_model)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: torch.Tensor):
        out = self.linear(self.silu(t_emb))           # [B, 9*d_model]
        chunks = out.chunk(9, dim=-1)                  # 9 × [B, d_model]
        return [c.unsqueeze(1) for c in chunks]        # 9 × [B, 1, d_model]


# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== modules.py sanity check ===")
    B, N, d = 2, 10, 256

    t = torch.rand(B)

    te = SinusoidalTimeEmbedding(d)
    emb = te(t)
    assert emb.shape == (B, d)
    print(f"SinusoidalTimeEmbedding : {emb.shape} ✓")

    be = BoxEmbedding(d)
    boxes = torch.randn(B, N, 4)
    out = be(boxes)
    assert out.shape == (B, N, d)
    print(f"BoxEmbedding            : {out.shape} ✓")

    heads, head_dim = 8, d // 8
    rope = RoPE(head_dim)
    x = torch.randn(B, heads, N, head_dim)
    x_rot = rope(x, t)
    assert x_rot.shape == (B, heads, N, head_dim)
    print(f"RoPE                    : {x_rot.shape} ✓")

    ada = AdaLNZero(d)
    params = ada(emb)
    assert len(params) == 9 and all(p.shape == (B, 1, d) for p in params)
    print(f"AdaLNZero               : 9 × {params[0].shape} ✓")

    print("All checks passed.")
