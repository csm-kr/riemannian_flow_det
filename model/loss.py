import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(
        self,
        v_hat: torch.Tensor,
        u_t:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Purpose: Flow matching loss — MSE between predicted and target vector field.
        Inputs:
            v_hat: [*, 4], float32 — model prediction in state space
            u_t:   [*, 4], float32 — target vector field in state space
        Outputs:
            loss: scalar (mean) or [*, 4] (none)
        """
        return F.mse_loss(v_hat, u_t, reduction=self.reduction)


if __name__ == "__main__":
    print("=== loss.py sanity check ===")
    B, N = 2, 10

    loss_fn = FlowMatchingLoss()
    v_hat   = torch.randn(B, N, 4)
    u_t     = torch.randn(B, N, 4)

    loss = loss_fn(v_hat, u_t)
    assert loss.shape == (), f"loss should be scalar, got {loss.shape}"
    assert loss.item() > 0
    print(f"loss: {loss.item():.4f}  ✓")

    # Zero loss when predictions are perfect
    zero_loss = loss_fn(u_t, u_t)
    assert zero_loss.item() < 1e-10
    print(f"zero loss (perfect pred): {zero_loss.item():.2e}  ✓")

    print("All checks passed.")
