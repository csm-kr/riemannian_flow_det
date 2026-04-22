import torch
import torch.nn as nn

from dataset.box_ops import cxcywh_to_state, state_to_cxcywh
from model.backbone import FPNBackbone, DINOv2Backbone
from model.dit import FlowDiT
from model.head import BoxHead
from model.loss import FlowMatchingLoss
from model.trajectory import (
    RiemannianTrajectory,
    RiemannianTrajectoryArbPrior,
    LinearTrajectory,
    LinearTrajectoryArbPrior,
)


class RiemannianFlowDet(nn.Module):
    """
    Top-level model for Riemannian flow matching-based object detection.

    Training:
        1. Convert boxes_gt to state space (log-scale)
        2. Sample t ~ U[0,1], noise b0 ~ N(0,I)
        3. Interpolate: b_t = (1-t)*b0 + t*b1  (Riemannian geodesic)
        4. Predict vector field v̂_t = BoxHead(FlowDiT(image, b_t, t))
        5. Minimize ‖v̂_t − u_t*‖²

    Inference:
        1. Sample b0 ~ N(0,I) in state space
        2. Euler ODE from t=0 to t=1
        3. Convert b1_pred to normalized cxcywh
    """

    def __init__(
        self,
        dim:             int = 256,
        depth:           int = 6,
        num_heads:       int = 8,
        mlp_ratio:       int = 4,
        num_queries:     int = 300,
        backbone_pretrained: bool = True,
        backbone_type:   str = "fpn",
        dinov2_model:    str = "dinov2_vits14",
        dinov2_freeze:   bool = False,
        trajectory_type: str = "riemannian",
    ):
        super().__init__()
        assert backbone_type in ("fpn", "dinov2"), \
            f"Unknown backbone_type: {backbone_type!r}. Choose 'fpn' or 'dinov2'."
        if backbone_type == "dinov2":
            backbone = DINOv2Backbone(
                dim=dim,
                model_name=dinov2_model,
                pretrained=backbone_pretrained,
                freeze=dinov2_freeze,
            )
        else:
            backbone = FPNBackbone(dim=dim, pretrained=backbone_pretrained)
        self.flow_dit  = FlowDiT(backbone, dim=dim, depth=depth,
                                 num_heads=num_heads, mlp_ratio=mlp_ratio,
                                 num_queries=num_queries)
        self.box_head  = BoxHead(dim=dim)
        self.loss_fn   = FlowMatchingLoss()
        self.num_queries = num_queries

        assert trajectory_type in (
            "riemannian", "linear", "linear_arb_prior", "riemannian_arb_prior"
        ), f"Unknown trajectory_type: {trajectory_type}"
        if trajectory_type == "riemannian":
            self.trajectory = RiemannianTrajectory()
        elif trajectory_type == "linear":
            self.trajectory = LinearTrajectory()
        elif trajectory_type == "linear_arb_prior":
            self.trajectory = LinearTrajectoryArbPrior()
        else:
            self.trajectory = RiemannianTrajectoryArbPrior()
        self.trajectory_type = trajectory_type

    # ── Training ──────────────────────────────────────────────────────────────

    def forward_train(
        self,
        images:        torch.Tensor,
        boxes_gt_list: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Purpose: Compute flow matching loss for a batch.
        Inputs:
            images:        [B, 3, H, W], float32, normalized
            boxes_gt_list: list of B tensors, each [Ni, 4] normalized cxcywh
        Outputs:
            dict with key "loss": scalar tensor
        """
        B = images.shape[0]

        # Pad GT boxes to max_N within batch; build validity mask
        max_N = max(b.shape[0] for b in boxes_gt_list)
        padded = images.new_zeros(B, max_N, 4)
        mask   = torch.zeros(B, max_N, dtype=torch.bool, device=images.device)
        for i, b in enumerate(boxes_gt_list):
            n = b.shape[0]
            padded[i, :n] = b
            mask[i, :n]   = True

        # Sample one t per image
        t = torch.rand(B, device=images.device)

        if self.trajectory_type in ("riemannian", "riemannian_arb_prior"):
            b1      = cxcywh_to_state(padded)           # [B, max_N, 4]
            b_t, u_t, _ = self.trajectory.sample(b1, t)
        else:
            b_t, u_t = self.trajectory.sample(padded, t)

        # Forward pass
        box_tokens = self.flow_dit(images, b_t, t)      # [B, max_N, dim]
        v_hat      = self.box_head(box_tokens)           # [B, max_N, 4]

        # Loss only on valid (non-padded) positions
        loss = self.loss_fn(v_hat[mask], u_t[mask])
        return {"loss": loss}

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_inference(
        self,
        images:      torch.Tensor,
        num_steps:   int = 10,
        num_queries: int | None = None,
    ) -> torch.Tensor:
        """
        Purpose: Run Euler ODE from noise to predicted boxes.
        Inputs:
            images:      [B, 3, H, W], float32
            num_steps:   number of Euler steps (higher → more accurate)
            num_queries: override self.num_queries if given
        Outputs:
            boxes: [B, Q, 4], float32, normalized cxcywh
        """
        B  = images.shape[0]
        Q  = num_queries if num_queries is not None else self.num_queries
        dt = 1.0 / num_steps

        # Start from the trajectory-specific prior (matches training distribution).
        b = self.trajectory.init_noise(B, Q, images.device, dtype=images.dtype)

        for i in range(num_steps):
            t_val = i / num_steps
            t     = torch.full((B,), t_val, device=images.device)
            box_tokens = self.flow_dit(images, b, t)
            v          = self.box_head(box_tokens)        # [B, Q, 4]
            b          = self.trajectory.ode_step(b, v, dt)

        boxes = state_to_cxcywh(b)   # [B, Q, 4] normalized cxcywh
        return boxes

    def forward(self, images, boxes_gt_list=None, **kwargs):
        """
        Purpose: Dispatch to forward_train or forward_inference.
        Inputs:
            images:        [B, 3, H, W]
            boxes_gt_list: list of [Ni, 4] (train) or None (inference)
        """
        if self.training and boxes_gt_list is not None:
            return self.forward_train(images, boxes_gt_list)
        return self.forward_inference(images, **kwargs)


if __name__ == "__main__":
    print("=== flow_matching.py sanity check ===")
    B, H, W = 2, 256, 256

    model = RiemannianFlowDet(
        dim=64, depth=2, num_heads=4, num_queries=10,
        backbone_pretrained=False,
    )
    model.eval()

    images = torch.zeros(B, 3, H, W)
    boxes_gt_list = [
        torch.rand(5, 4) * 0.5 + 0.1,   # 5 boxes
        torch.rand(3, 4) * 0.5 + 0.1,   # 3 boxes
    ]

    # Training forward
    model.train()
    out = model.forward_train(images, boxes_gt_list)
    assert "loss" in out and out["loss"].shape == ()
    print(f"train loss: {out['loss'].item():.4f}  ✓")

    # Inference forward
    model.eval()
    boxes = model.forward_inference(images, num_steps=5)
    assert boxes.shape == (B, model.num_queries, 4), f"shape: {boxes.shape}"
    print(f"inference boxes: {boxes.shape}  ✓")

    # Linear trajectory
    model_lin = RiemannianFlowDet(
        dim=64, depth=2, num_heads=4, num_queries=10,
        backbone_pretrained=False, trajectory_type="linear",
    )
    model_lin.train()
    out_lin = model_lin.forward_train(images, boxes_gt_list)
    print(f"linear traj loss: {out_lin['loss'].item():.4f}  ✓")

    # DINOv2 backbone
    print("\n[DINOv2 backbone]")
    model_dino = RiemannianFlowDet(
        dim=64, depth=2, num_heads=4, num_queries=10,
        backbone_pretrained=True,
        backbone_type="dinov2",
        dinov2_model="dinov2_vits14",
        dinov2_freeze=True,
    )
    # 224 is divisible by 14
    images_dino = torch.zeros(B, 3, 224, 224)
    model_dino.train()
    out_dino = model_dino.forward_train(images_dino, boxes_gt_list)
    assert "loss" in out_dino and out_dino["loss"].shape == ()
    print(f"DINOv2 train loss: {out_dino['loss'].item():.4f}  ✓")

    model_dino.eval()
    boxes_dino = model_dino.forward_inference(images_dino, num_steps=3)
    assert boxes_dino.shape == (B, model_dino.num_queries, 4)
    print(f"DINOv2 inference boxes: {boxes_dino.shape}  ✓")

    print("\nAll checks passed.")
