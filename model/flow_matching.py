# File: model/flow_matching.py
# Role: 전체 모델 조립 + forward + Euler ODE inference
# 외부에서는 이 파일만 import.

import torch
import torch.nn as nn

from dataset.box_ops import cxcywh_to_state, state_to_cxcywh
from model.backbone   import ImageBackbone
from model.modules    import SinusoidalTimeEmbedding, BoxEmbedding
from model.dit        import DiTBlock
from model.head       import VectorFieldHead
from model.trajectory import build_trajectory
from model.loss       import FlowMatchingLoss


class RiemannianFlowDet(nn.Module):
    """
    Purpose: image-conditioned flow matching detector.
    이미지에서 image_tokens 추출 → box state에 noise 섞어 → DiTBlock으로 vector field 예측.

    Inputs (forward):
        images:     [B, 3, H, W],  float32 — ImageNet normalized
        b_t:        [B, N, 4],     float32 — noisy box state (log-scale)
        t:          [B],           float32 — time ∈ [0, 1]
    Outputs:
        v_pred:     [B, N, 4],     float32 — predicted vector field
    """
    def __init__(
        self,
        d_model:         int   = 256,
        nhead:           int   = 8,
        num_layers:      int   = 4,
        dim_feedforward: int   = 1024,
        dropout:         float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.backbone  = ImageBackbone(d_model=d_model, pretrained=pretrained_backbone)
        self.box_embed = BoxEmbedding(d_model=d_model)
        self.time_embed = SinusoidalTimeEmbedding(d_model=d_model)

        self.blocks = nn.ModuleList([
            DiTBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.head = VectorFieldHead(d_model=d_model)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Purpose: 이미지 feature 추출 (inference 시 한 번만 호출).
        Inputs:  images       [B, 3, H, W]
        Outputs: image_tokens [B, S, d_model]
        """
        return self.backbone(images)

    def forward(
        self,
        images:       torch.Tensor,
        b_t:          torch.Tensor,
        t:            torch.Tensor,
        image_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Inputs:
            images:       [B, 3, H, W]   (image_tokens 없을 때 사용)
            b_t:          [B, N, 4]      float32 — noisy log-scale state
            t:            [B]            float32 — time ∈ [0, 1]
            image_tokens: [B, S, d_model] (미리 계산된 경우, images 무시)
        Outputs:
            v_pred: [B, N, 4]
        """
        if image_tokens is None:
            image_tokens = self.encode_image(images)   # [B, S, d]

        x     = self.box_embed(b_t)                    # [B, N, d]
        t_emb = self.time_embed(t)                     # [B, d]

        for block in self.blocks:
            x = block(x, image_tokens, t_emb)         # [B, N, d]

        v_pred = self.head(x)                          # [B, N, 4]
        return v_pred

    @torch.no_grad()
    def sample(
        self,
        images: torch.Tensor,
        num_boxes: int = 100,
        steps: int = 100,
    ) -> torch.Tensor:
        """
        Purpose: Euler ODE로 b0 → b1 생성 (inference).
        Inputs:
            images:    [B, 3, H, W]
            num_boxes: N (쿼리 수)
            steps:     T (Euler steps)
        Outputs:
            boxes: [B, N, 4], float32 — normalized cxcywh
        """
        B      = images.shape[0]
        device = images.device

        image_tokens = self.encode_image(images)                     # [B, S, d]
        b = torch.randn(B, num_boxes, 4, device=device)              # b0 ~ N(0,I)

        for i in range(steps):
            t_val = torch.full((B,), i / steps, device=device)      # [B]
            v = self.forward(images, b, t_val, image_tokens)
            b = b + (1.0 / steps) * v

        return state_to_cxcywh(b).clamp(0, 1)                       # normalized cxcywh


def build_model(cfg) -> RiemannianFlowDet:
    """
    Purpose: config에서 모델 생성.
    Inputs:  cfg — ConfigArgParse namespace (또는 dict-like)
    Outputs: RiemannianFlowDet
    """
    return RiemannianFlowDet(
        d_model         = getattr(cfg, "d_model",         256),
        nhead           = getattr(cfg, "nhead",           8),
        num_layers      = getattr(cfg, "num_layers",      4),
        dim_feedforward = getattr(cfg, "dim_feedforward", 1024),
        dropout         = getattr(cfg, "dropout",         0.1),
        pretrained_backbone = getattr(cfg, "pretrained_backbone", True),
    )


# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== flow_matching.py sanity check ===")
    B, N, H, W = 2, 10, 224, 224

    model = RiemannianFlowDet(
        d_model=256, nhead=8, num_layers=2,
        dim_feedforward=512, pretrained_backbone=False,
    )
    model.eval()

    images = torch.randn(B, 3, H, W)
    b_t    = torch.randn(B, N, 4)
    t      = torch.rand(B)

    # forward
    with torch.no_grad():
        v_pred = model(images, b_t, t)
    assert v_pred.shape == (B, N, 4), f"forward: {v_pred.shape}"
    print(f"forward  : images{list(images.shape)} + b_t{list(b_t.shape)} → v_pred{list(v_pred.shape)} ✓")

    # sample (inference)
    with torch.no_grad():
        boxes = model.sample(images, num_boxes=N, steps=10)
    assert boxes.shape == (B, N, 4)
    assert boxes.min() >= -0.1 and boxes.max() <= 1.1
    print(f"sample   : {list(boxes.shape)},  range [{boxes.min():.3f}, {boxes.max():.3f}] ✓")

    # loss
    from model.trajectory import LinearTrajectory
    from model.loss       import FlowMatchingLoss

    traj      = LinearTrajectory()
    criterion = FlowMatchingLoss()

    b0    = torch.randn(B, N, 4)
    b1    = torch.randn(B, N, 4)
    t_sca = torch.rand(B)
    b_t_  = traj.interpolate(b0, b1, t_sca)
    u_t   = traj.target_field(b0, b1, t_sca)

    with torch.no_grad():
        v_pred2 = model(images, b_t_, t_sca)
    losses = criterion(v_pred2, u_t, b_t_, b1)
    print(f"loss     : total={losses['total'].item():.4f}  flow={losses['flow'].item():.4f}  giou={losses['giou'].item():.4f} ✓")

    print("All checks passed.")
