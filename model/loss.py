# File: model/loss.py
# Role: Flow matching loss — CFM(MSE) + geometry L1 + GIoU

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.box_ops import state_to_cxcywh, cxcywh_to_xyxy


def generalized_box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Purpose: pairwise GIoU 계산 (xyxy normalized).
    Inputs:
        boxes_a: [N, 4], float32, xyxy normalized
        boxes_b: [N, 4], float32, xyxy normalized  (same N, per-pair)
    Outputs:
        giou:    [N], float32  ∈ [-1, 1]
    """
    # intersection
    inter_x1 = torch.max(boxes_a[:, 0], boxes_b[:, 0])
    inter_y1 = torch.max(boxes_a[:, 1], boxes_b[:, 1])
    inter_x2 = torch.min(boxes_a[:, 2], boxes_b[:, 2])
    inter_y2 = torch.min(boxes_a[:, 3], boxes_b[:, 3])
    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h

    # union
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union  = (area_a + area_b - inter).clamp(min=1e-6)
    iou    = inter / union

    # enclosing box
    enc_x1 = torch.min(boxes_a[:, 0], boxes_b[:, 0])
    enc_y1 = torch.min(boxes_a[:, 1], boxes_b[:, 1])
    enc_x2 = torch.max(boxes_a[:, 2], boxes_b[:, 2])
    enc_y2 = torch.max(boxes_a[:, 3], boxes_b[:, 3])
    enc    = ((enc_x2 - enc_x1) * (enc_y2 - enc_y1)).clamp(min=1e-6)

    giou = iou - (enc - union) / enc
    return giou                          # [N]


class FlowMatchingLoss(nn.Module):
    """
    Purpose: CFM loss + geometry L1 + GIoU (endpoint 근사).

    L = w_flow * MSE(v_pred, u_t)
      + w_l1   * L1(v_pred, u_t)
      + w_giou * (1 - GIoU(endpoint_pred, b1))

    Inputs:
        v_pred:     [B, N, 4], float32 — 예측 vector field
        u_t:        [B, N, 4], float32 — target vector field (b1 - b0)
        b_t:        [B, N, 4], float32 — 현재 state (log-scale)
        b1:         [B, N, 4], float32 — GT state (log-scale)
        valid_mask: [B, N],    bool    — True = 유효한 box (패딩 제외)
    Outputs:
        loss_dict: dict with keys 'flow', 'geom_l1', 'giou', 'total'
    """
    def __init__(
        self,
        w_flow: float = 1.0,
        w_l1:   float = 0.2,
        w_giou: float = 1.0,
    ):
        super().__init__()
        self.w_flow = w_flow
        self.w_l1   = w_l1
        self.w_giou = w_giou

    def forward(
        self,
        v_pred:     torch.Tensor,
        u_t:        torch.Tensor,
        b_t:        torch.Tensor,
        b1:         torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> dict:
        """
        Inputs:
            v_pred:     [B, N, 4]
            u_t:        [B, N, 4]
            b_t:        [B, N, 4]
            b1:         [B, N, 4]
            valid_mask: [B, N] bool (None 이면 전체 유효)
        Outputs:
            loss_dict: {'flow', 'geom_l1', 'giou', 'total'}
        """
        B, N, _ = v_pred.shape

        if valid_mask is None:
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=v_pred.device)

        # valid box만 추출 → [M, 4]
        mask4 = valid_mask.unsqueeze(-1).expand_as(v_pred)  # [B, N, 4]

        v_valid  = v_pred[mask4].view(-1, 4)   # [M, 4]
        u_valid  = u_t[mask4].view(-1, 4)      # [M, 4]
        bt_valid = b_t[mask4].view(-1, 4)      # [M, 4]
        b1_valid = b1[mask4].view(-1, 4)       # [M, 4]

        # ── CFM loss (MSE) ───────────────────────
        flow_loss = F.mse_loss(v_valid, u_valid)

        # ── Geometry L1 ──────────────────────────
        geom_l1 = F.l1_loss(v_valid, u_valid)

        # ── GIoU (endpoint 근사) ─────────────────
        # endpoint 예측: b_t + v_pred ≈ b1 (1-step Euler)
        b_end_pred = bt_valid + v_valid          # [M, 4] log-scale state
        b_end_gt   = b1_valid

        # state → normalized cxcywh → xyxy
        pred_cxcywh = state_to_cxcywh(b_end_pred).clamp(0, 1)  # [M, 4]
        gt_cxcywh   = state_to_cxcywh(b_end_gt).clamp(0, 1)

        pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)  # [M, 4]
        gt_xyxy   = cxcywh_to_xyxy(gt_cxcywh)

        giou      = generalized_box_iou(pred_xyxy, gt_xyxy)  # [M]
        giou_loss = (1 - giou).mean()

        # ── Total ────────────────────────────────
        total = (
            self.w_flow * flow_loss
            + self.w_l1  * geom_l1
            + self.w_giou * giou_loss
        )

        return {
            "flow":    flow_loss,
            "geom_l1": geom_l1,
            "giou":    giou_loss,
            "total":   total,
        }


# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== loss.py sanity check ===")
    B, N = 2, 5

    b0     = torch.randn(B, N, 4)
    b1     = torch.randn(B, N, 4)
    t      = torch.rand(B, 1, 1)
    b_t    = (1 - t) * b0 + t * b1
    u_t    = b1 - b0
    v_pred = u_t + 0.01 * torch.randn_like(u_t)   # 거의 완벽한 예측
    valid  = torch.ones(B, N, dtype=torch.bool)

    criterion = FlowMatchingLoss(w_flow=1.0, w_l1=0.2, w_giou=1.0)
    losses = criterion(v_pred, u_t, b_t, b1, valid)

    for k, v in losses.items():
        print(f"  {k:8s}: {v.item():.6f}")
    assert losses["total"].item() > 0
    print("All checks passed.")
