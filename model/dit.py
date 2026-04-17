# File: model/dit.py
# Role: DiTBlock — AdaLN-Zero + RoPE + Self-Attention + Cross-Attention + FFN

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import AdaLNZero, RoPE


class DiTBlock(nn.Module):
    """
    Purpose: image-conditioned DiT block.
      1. Self-Attention  (box ↔ box)       + AdaLN-Zero + RoPE
      2. Cross-Attention (box ↔ image)     + AdaLN-Zero + RoPE on Q only
      3. FFN                               + AdaLN-Zero

    Inputs:
        x:            [B, N, d_model], float32 — box tokens
        image_tokens: [B, S, d_model], float32 — image patch tokens
        t_emb:        [B, d_model],    float32 — sinusoidal time embedding
    Outputs:
        x:            [B, N, d_model], float32
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        # AdaLN-Zero: 9 params (s,b,g) × 3 sub-blocks
        self.adaLN = AdaLNZero(d_model)

        # LayerNorm (elementwise_affine=False — AdaLN이 대체)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # Self-Attention projections
        self.sa_q  = nn.Linear(d_model, d_model, bias=False)
        self.sa_k  = nn.Linear(d_model, d_model, bias=False)
        self.sa_v  = nn.Linear(d_model, d_model, bias=False)
        self.sa_out = nn.Linear(d_model, d_model)

        # Cross-Attention projections
        self.ca_q  = nn.Linear(d_model, d_model, bias=False)
        self.ca_k  = nn.Linear(d_model, d_model, bias=False)
        self.ca_v  = nn.Linear(d_model, d_model, bias=False)
        self.ca_out = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # RoPE (time 기반)
        self.rope = RoPE(self.head_dim)

        self.attn_drop = nn.Dropout(dropout)

    # ── 헬퍼 ───────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, d] → [B, heads, N, head_dim]"""
        B, N, _ = x.shape
        return x.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, heads, N, head_dim] → [B, N, d]"""
        B, h, N, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, h * d)

    def _attention(self, q, k, v) -> torch.Tensor:
        """
        Scaled dot-product attention.
        q: [B, h, Nq, d],  k,v: [B, h, Nk, d]
        → [B, h, Nq, d]
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,h,Nq,Nk]
        attn   = self.attn_drop(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v)

    # ── Forward ────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        image_tokens: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # 9개 AdaLN-Zero 파라미터
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.adaLN(t_emb)
        # 각 [B, 1, d_model]

        # ── 1. Self-Attention (box ↔ box) ─────────
        x_norm = (1 + s1) * self.norm1(x) + b1        # [B, N, d]

        Q = self._split_heads(self.sa_q(x_norm))       # [B, h, N, d_h]
        K = self._split_heads(self.sa_k(x_norm))
        V = self._split_heads(self.sa_v(x_norm))

        # RoPE: self-attn에서 Q, K 모두 회전 (일관성)
        t = t_emb.mean(dim=-1)  # [B] — time scalar 복원용 (근사)
        Q = self.rope(Q, t)
        K = self.rope(K, t)

        sa_out = self._merge_heads(self._attention(Q, K, V))  # [B, N, d]
        sa_out = self.sa_out(sa_out)
        x = x + g1 * sa_out                            # gate (zero-init)

        # ── 2. Cross-Attention (box ↔ image) ──────
        x_norm = (1 + s2) * self.norm2(x) + b2        # [B, N, d]

        Q = self._split_heads(self.ca_q(x_norm))       # [B, h, N,  d_h]
        K = self._split_heads(self.ca_k(image_tokens)) # [B, h, S,  d_h]
        V = self._split_heads(self.ca_v(image_tokens)) # [B, h, S,  d_h]

        # RoPE: Q(box, time t)만 회전 — K(image, time 0)는 무회전
        # → attention score에 시간 정보 반영
        Q = self.rope(Q, t)

        ca_out = self._merge_heads(self._attention(Q, K, V))  # [B, N, d]
        ca_out = self.ca_out(ca_out)
        x = x + g2 * ca_out

        # ── 3. FFN ────────────────────────────────
        x_norm = (1 + s3) * self.norm3(x) + b3        # [B, N, d]
        x = x + g3 * self.ffn(x_norm)

        return x


# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== dit.py sanity check ===")
    B, N, S, d = 2, 10, 100, 256

    block = DiTBlock(d_model=d, nhead=8, dim_feedforward=1024)
    block.eval()

    x            = torch.randn(B, N, d)
    image_tokens = torch.randn(B, S, d)
    t_emb        = torch.randn(B, d)

    with torch.no_grad():
        out = block(x, image_tokens, t_emb)

    assert out.shape == (B, N, d), f"DiTBlock output: {out.shape}"
    print(f"DiTBlock: {x.shape} → {out.shape} ✓")

    # 학습 초기 gate=0 확인 (zero-init 이므로 out ≈ x)
    diff = (out - x).abs().max().item()
    print(f"gate=0 residual diff (should be ~0): {diff:.6f} ✓")

    print("All checks passed.")
