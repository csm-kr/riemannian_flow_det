import math
import torch
import torch.nn as nn


# ── Timestep Embedding ────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Purpose: Encode scalar timestep into sinusoidal embedding.
        Inputs:
            t: [B], float32, values in [0, 1]
        Outputs:
            emb: [B, dim], float32
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / (half - 1)
        )  # [half]
        args = t[:, None] * freqs[None, :]          # [B, half]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]
        return self.mlp(emb)


# ── Box State Embedding ───────────────────────────────────────────────────────

class BoxEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, b_t: torch.Tensor) -> torch.Tensor:
        """
        Purpose: Project box state vectors [cx, cy, log_w, log_h] to hidden dim.
        Inputs:
            b_t: [B, N, 4], float32, state space
        Outputs:
            tokens: [B, N, dim], float32
        """
        return self.proj(b_t)


# ── Adaptive LayerNorm ────────────────────────────────────────────────────────

class AdaLN(nn.Module):
    """Adaptive LayerNorm: modulates x with scale/shift derived from a context vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Purpose: LayerNorm x, then apply scale/shift conditioned on ctx.
        Inputs:
            x:   [B, N, dim], float32
            ctx: [B, dim],    float32  (e.g. timestep embedding)
        Outputs:
            out: [B, N, dim], float32
        """
        scale, shift = self.proj(ctx).chunk(2, dim=-1)   # each [B, dim]
        return self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purpose: Feed-forward block.
        Inputs/Outputs: [B, N, dim]
        """
        return self.net(x)


# ── RoPE ──────────────────────────────────────────────────────────────────────

def build_2d_rope_freqs(
    head_dim: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
) -> torch.Tensor:
    """
    Purpose: Build 2D RoPE frequency tensors from (cx, cy) positions.
             head_dim must be divisible by 4.
             First head_dim//4 dims encode cx, last head_dim//4 dims encode cy.
    Inputs:
        head_dim: int
        cx: [N], float32 in [0, 1]
        cy: [N], float32 in [0, 1]
    Outputs:
        freqs: [N, head_dim // 2], float32
    """
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4, got {head_dim}"
    quarter = head_dim // 4
    inv_x = 1.0 / (
        10000.0 ** (torch.arange(quarter, device=cx.device, dtype=cx.dtype) / quarter)
    )  # [quarter]
    inv_y = 1.0 / (
        10000.0 ** (torch.arange(quarter, device=cy.device, dtype=cy.dtype) / quarter)
    )
    freqs_cx = cx[:, None] * inv_x[None, :]   # [N, quarter]
    freqs_cy = cy[:, None] * inv_y[None, :]   # [N, quarter]
    return torch.cat([freqs_cx, freqs_cy], dim=-1)   # [N, head_dim//2]


def build_2d_grid_rope_freqs(
    head_dim: int,
    h: int,
    w: int,
    device,
    dtype,
) -> torch.Tensor:
    """
    Purpose: Build 2D RoPE frequency tensors for a regular H×W spatial grid.
    Inputs:
        head_dim: int (divisible by 4)
        h, w: grid dimensions
    Outputs:
        freqs: [H*W, head_dim // 2], float32
    """
    ys = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    return build_2d_rope_freqs(head_dim, grid_x.flatten(), grid_y.flatten())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Purpose: Apply rotary position embeddings to query and key.
    Inputs:
        q:       [B, Nq, num_heads, head_dim]
        k:       [B, Nk, num_heads, head_dim]
        q_freqs: [Nq, head_dim // 2]
        k_freqs: [Nk, head_dim // 2]
    Outputs:
        q_rot, k_rot: same shapes as q, k
    """
    def _rot(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        # freqs [N, D//2] → broadcast to [1, N, 1, D] via repeat
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, :, None, :]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, :, None, :]
        return x * cos + rotate_half(x) * sin

    return _rot(q, q_freqs), _rot(k, k_freqs)


if __name__ == "__main__":
    print("=== modules.py sanity check ===")
    B, N, D, H = 2, 10, 256, 8
    head_dim = D // H

    t = torch.rand(B)
    emb = SinusoidalEmbedding(D)(t)
    assert emb.shape == (B, D), f"SinusoidalEmbedding: {emb.shape}"
    print(f"SinusoidalEmbedding: {emb.shape} ✓")

    b_t = torch.randn(B, N, 4)
    tok = BoxEmbedding(D)(b_t)
    assert tok.shape == (B, N, D), f"BoxEmbedding: {tok.shape}"
    print(f"BoxEmbedding:        {tok.shape} ✓")

    ctx = torch.randn(B, D)
    out = AdaLN(D)(tok, ctx)
    assert out.shape == (B, N, D), f"AdaLN: {out.shape}"
    print(f"AdaLN:               {out.shape} ✓")

    out = MLP(D)(tok)
    assert out.shape == (B, N, D), f"MLP: {out.shape}"
    print(f"MLP:                 {out.shape} ✓")

    cx = torch.rand(N)
    cy = torch.rand(N)
    freqs = build_2d_rope_freqs(head_dim, cx, cy)
    assert freqs.shape == (N, head_dim // 2), f"rope freqs: {freqs.shape}"
    print(f"2D RoPE freqs:       {freqs.shape} ✓")

    grid_freqs = build_2d_grid_rope_freqs(head_dim, 25, 34, cx.device, cx.dtype)
    print(f"Grid RoPE freqs:     {grid_freqs.shape} ✓")

    q = torch.randn(B, N, H, head_dim)
    k = torch.randn(B, N, H, head_dim)
    qr, kr = apply_rope(q, k, freqs, freqs)
    assert qr.shape == q.shape and kr.shape == k.shape
    print(f"apply_rope:          {qr.shape} ✓")

    print("All checks passed.")
