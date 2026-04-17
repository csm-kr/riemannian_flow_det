# File: model/head.py
# Role: transformer 출력 → 4-dim vector field projection

import torch
import torch.nn as nn


class VectorFieldHead(nn.Module):
    """
    Purpose: box token feature → vector field 예측.
    Inputs:
        x:      [B, N, d_model], float32
    Outputs:
        v_pred: [B, N, 4],       float32 — predicted vector field in box state space
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 4)

        # zero-init → 학습 초기 작은 vector field 예측
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:  x      [B, N, d_model]
        Outputs: v_pred [B, N, 4]
        """
        return self.proj(self.norm(x))


# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== head.py sanity check ===")
    B, N, d = 2, 10, 256

    head = VectorFieldHead(d_model=d)
    x = torch.randn(B, N, d)
    v = head(x)

    assert v.shape == (B, N, 4), f"VectorFieldHead: {v.shape}"
    print(f"VectorFieldHead: {x.shape} → {v.shape} ✓")

    # zero-init 확인
    assert v.abs().max().item() < 1e-6, "zero-init check failed"
    print(f"zero-init output: {v.abs().max().item():.2e} ✓")

    print("All checks passed.")
