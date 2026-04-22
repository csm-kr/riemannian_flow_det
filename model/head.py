import torch
import torch.nn as nn


class BoxHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 4),
        )

    def forward(self, box_tokens: torch.Tensor) -> torch.Tensor:
        """
        Purpose: Project box token representations to vector field predictions.
        Inputs:
            box_tokens: [B, N, dim], float32
        Outputs:
            v_hat: [B, N, 4], float32 — predicted vector field in state space
        """
        return self.net(box_tokens)


if __name__ == "__main__":
    print("=== head.py sanity check ===")
    B, N, D = 2, 20, 256

    head = BoxHead(dim=D)
    x    = torch.randn(B, N, D)
    out  = head(x)

    assert out.shape == (B, N, 4), f"shape mismatch: {out.shape}"
    print(f"BoxHead output: {out.shape}  ✓")
    print("All checks passed.")
