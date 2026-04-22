import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import (
    AdaLN, MLP, SinusoidalEmbedding, BoxEmbedding,
    build_2d_rope_freqs, build_2d_grid_rope_freqs, apply_rope,
)


# ── Multi-Head Attention with optional RoPE ───────────────────────────────────

class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        q_in:    torch.Tensor,
        k_in:    torch.Tensor,
        v_in:    torch.Tensor,
        q_freqs: torch.Tensor | None = None,
        k_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Purpose: Scaled dot-product attention with optional 2D RoPE on Q and K.
        Inputs:
            q_in:    [B, Nq, dim]
            k_in:    [B, Nk, dim]
            v_in:    [B, Nk, dim]
            q_freqs: [Nq, head_dim // 2] or None
            k_freqs: [Nk, head_dim // 2] or None
        Outputs:
            out: [B, Nq, dim]
        """
        B, Nq, _ = q_in.shape
        Nk = k_in.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(q_in).view(B, Nq, H, D)
        k = self.k_proj(k_in).view(B, Nk, H, D)
        v = self.v_proj(v_in).view(B, Nk, H, D)

        if q_freqs is not None and k_freqs is not None:
            q, k = apply_rope(q, k, q_freqs, k_freqs)

        # sdpa expects [B, H, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)   # [B, H, Nq, D]
        out = out.transpose(1, 2).reshape(B, Nq, H * D)
        return self.out_proj(out)


# ── DiT Block ─────────────────────────────────────────────────────────────────

class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1      = nn.LayerNorm(dim)
        self.self_attn  = MultiHeadAttentionRoPE(dim, num_heads)
        self.norm2      = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttentionRoPE(dim, num_heads)
        self.ada_ln     = AdaLN(dim)
        self.mlp        = MLP(dim, mlp_ratio)

    def forward(
        self,
        box_tokens: torch.Tensor,
        img_tokens: torch.Tensor,
        t_emb:      torch.Tensor,
        box_freqs:  torch.Tensor,
        img_freqs:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Purpose: One DiT block — self-attn (box↔box) + cross-attn (box→img) + MLP,
                 all conditioned on timestep via adaLN.
        Inputs:
            box_tokens: [B, N, dim]
            img_tokens: [B, L, dim]
            t_emb:      [B, dim]          — sinusoidal timestep embedding
            box_freqs:  [N, head_dim//2]  — RoPE from box (cx, cy)
            img_freqs:  [L, head_dim//2]  — RoPE from image spatial grid
        Outputs:
            box_tokens: [B, N, dim]
        """
        # 1. Self-attention (box ↔ box) with box-position RoPE
        h = self.norm1(box_tokens)
        h = self.self_attn(h, h, h, q_freqs=box_freqs, k_freqs=box_freqs)
        box_tokens = box_tokens + h

        # 2. Cross-attention (box → image) with spatial RoPE on K
        h = self.norm2(box_tokens)
        h = self.cross_attn(h, img_tokens, img_tokens,
                            q_freqs=box_freqs, k_freqs=img_freqs)
        box_tokens = box_tokens + h

        # 3. adaLN-conditioned MLP
        h = self.ada_ln(box_tokens, t_emb)
        h = self.mlp(h)
        box_tokens = box_tokens + h

        return box_tokens


# ── FlowDiT ───────────────────────────────────────────────────────────────────

class FlowDiT(nn.Module):
    def __init__(
        self,
        backbone:    nn.Module,
        dim:         int = 256,
        depth:       int = 6,
        num_heads:   int = 8,
        mlp_ratio:   int = 4,
        num_queries: int = 300,
    ):
        super().__init__()
        self.backbone   = backbone
        self.box_embed  = BoxEmbedding(dim)
        self.time_embed = SinusoidalEmbedding(dim)
        # Learnable per-query positional embedding — 각 query 슬롯을 고유 식별.
        # 1-to-1 class-indexed 매칭(query_i → class_i)과 Hungarian 매칭 모두에 필요.
        self.query_embed = nn.Embedding(num_queries, dim)
        nn.init.normal_(self.query_embed.weight, mean=0.0, std=0.02)
        self.blocks     = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm_out   = nn.LayerNorm(dim)

        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.dim         = dim
        self.num_queries = num_queries

    def forward(
        self,
        images: torch.Tensor,
        b_t:    torch.Tensor,
        t:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Purpose: Predict box token representations from image and noisy box states.
        Inputs:
            images: [B, 3, H, W], float32
            b_t:    [B, N, 4],   float32, state space [cx, cy, log_w, log_h] at time t
            t:      [B],          float32, in [0, 1]
        Outputs:
            box_tokens: [B, N, dim], float32
        """
        B, N, _ = b_t.shape

        # Image features from backbone
        img_tokens, hw_list = self.backbone(images)   # [B, L_img, dim]

        # Timestep embedding
        t_emb = self.time_embed(t)   # [B, dim]

        # Box tokens from state + learnable per-query positional embedding
        box_tokens = self.box_embed(b_t)                         # [B, N, dim]
        query_ids  = torch.arange(N, device=b_t.device)
        # 학습된 num_queries보다 N이 작은 경우 앞쪽 슬롯만 사용
        assert N <= self.num_queries, (
            f"N={N} exceeds FlowDiT.num_queries={self.num_queries}; "
            f"learnable query embedding only covers the first {self.num_queries} slots."
        )
        box_tokens = box_tokens + self.query_embed(query_ids)[None]  # [B, N, dim]

        # RoPE for box tokens: use (cx, cy) from b_t
        # Batch-mean positions used for shared freqs across the batch.
        # This is an approximation; per-sample RoPE requires a sample loop.
        cx_mean = b_t[:, :, 0].mean(0).clamp(0.0, 1.0)   # [N]
        cy_mean = b_t[:, :, 1].mean(0).clamp(0.0, 1.0)   # [N]
        box_freqs = build_2d_rope_freqs(self.head_dim, cx_mean, cy_mean)  # [N, D_h//2]

        # RoPE for image tokens: regular spatial grid per FPN scale
        img_freqs_list = [
            build_2d_grid_rope_freqs(self.head_dim, H_k, W_k, images.device, images.dtype)
            for H_k, W_k in hw_list
        ]
        img_freqs = torch.cat(img_freqs_list, dim=0)   # [L_img, D_h//2]

        # Apply DiT blocks
        for block in self.blocks:
            box_tokens = block(box_tokens, img_tokens, t_emb, box_freqs, img_freqs)

        return self.norm_out(box_tokens)   # [B, N, dim]


if __name__ == "__main__":
    print("=== dit.py sanity check ===")
    from model.backbone import FPNBackbone

    B, N, D = 2, 20, 256
    backbone = FPNBackbone(dim=D, pretrained=False)
    model    = FlowDiT(backbone, dim=D, depth=2, num_heads=8)
    model.eval()

    images = torch.zeros(B, 3, 256, 256)
    b_t    = torch.randn(B, N, 4)
    t      = torch.rand(B)

    with torch.no_grad():
        out = model(images, b_t, t)

    assert out.shape == (B, N, D), f"shape mismatch: {out.shape}"
    print(f"FlowDiT output: {out.shape}  ✓")
    print("All checks passed.")
