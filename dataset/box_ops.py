# File: dataset/box_ops.py
# Role: 모든 박스 포맷 변환의 단일 진입점
# Pipeline: dataset 로드 직후, transform 전후, 모델 state space 진입 시 사용
# Formats:
#   xyxy      — [x1, y1, x2, y2] pixel
#   cxcywh    — [cx, cy, w, h]   pixel or normalized
#   state     — [cx, cy, log_w, log_h]  (모델 내부 state space, ℝ² × ℝ₊²)

import torch


# ────────────────────────────────────────────────
# 포맷 변환
# ────────────────────────────────────────────────

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Inputs:  boxes [N,4] or [B,N,4], float32, xyxy
    Outputs: boxes [N,4] or [B,N,4], float32, cxcywh
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Inputs:  boxes [N,4] or [B,N,4], float32, cxcywh
    Outputs: boxes [N,4] or [B,N,4], float32, xyxy
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def normalize_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    cxcywh pixel → normalized cxcywh [0,1]
    Inputs:  boxes [N,4] or [B,N,4], float32, cxcywh pixel
    Outputs: boxes [N,4] or [B,N,4], float32, normalized cxcywh
    """
    scale = boxes.new_tensor([img_w, img_h, img_w, img_h])
    return boxes / scale


def denormalize_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    normalized cxcywh → cxcywh pixel
    Inputs:  boxes [N,4] or [B,N,4], float32, normalized
    Outputs: boxes [N,4] or [B,N,4], float32, cxcywh pixel
    """
    scale = boxes.new_tensor([img_w, img_h, img_w, img_h])
    return boxes * scale


# ────────────────────────────────────────────────
# State space 변환 (모델 내부 전용, ℝ² × ℝ₊²)
# ────────────────────────────────────────────────

def cxcywh_to_state(boxes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    normalized cxcywh → state space [cx, cy, log_w, log_h]
    center는 Euclidean, scale은 log space (ℝ² × ℝ₊²)

    Inputs:  boxes [N,4] or [B,N,4], float32, normalized cxcywh
    Outputs: state [N,4] or [B,N,4], float32, [cx, cy, log_w, log_h]
    """
    cx, cy, w, h = boxes.unbind(-1)
    log_w = torch.log(w.clamp(min=eps))
    log_h = torch.log(h.clamp(min=eps))
    return torch.stack([cx, cy, log_w, log_h], dim=-1)


def state_to_cxcywh(states: torch.Tensor) -> torch.Tensor:
    """
    state space → normalized cxcywh

    Inputs:  states [N,4] or [B,N,4], float32, [cx, cy, log_w, log_h]
    Outputs: boxes  [N,4] or [B,N,4], float32, normalized cxcywh
    """
    cx, cy, log_w, log_h = states.unbind(-1)
    w = torch.exp(log_w)
    h = torch.exp(log_h)
    return torch.stack([cx, cy, w, h], dim=-1)


# ────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────

def clip_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """
    xyxy pixel 박스를 이미지 경계 내로 clamp
    Inputs/Outputs: [N,4] or [B,N,4], xyxy pixel
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    x1 = x1.clamp(0, img_w)
    y1 = y1.clamp(0, img_h)
    x2 = x2.clamp(0, img_w)
    y2 = y2.clamp(0, img_h)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    cxcywh 또는 xyxy 박스 면적 계산
    Inputs:  [N,4] or [B,N,4]
    Outputs: [N]   or [B,N]
    Note: cxcywh의 경우 w*h, xyxy의 경우 (x2-x1)*(y2-y1)
    """
    return boxes[..., 2] * boxes[..., 3]


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    두 박스 집합 간 pairwise IoU 계산 (xyxy pixel 기준)

    Inputs:
        boxes_a: [N, 4], float32, xyxy
        boxes_b: [M, 4], float32, xyxy
    Outputs:
        iou:     [N, M], float32
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # [N]
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # [M]

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h  # [N, M]

    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


if __name__ == "__main__":
    print("=== box_ops.py sanity check ===")

    # xyxy ↔ cxcywh
    xyxy = torch.tensor([[10., 20., 50., 80.],
                         [0.,  0.,  100., 100.]])
    cxcywh = xyxy_to_cxcywh(xyxy)
    xyxy_back = cxcywh_to_xyxy(cxcywh)
    assert torch.allclose(xyxy, xyxy_back), "xyxy round-trip failed"
    print(f"xyxy → cxcywh: {cxcywh}")

    # normalize ↔ denormalize
    norm = normalize_boxes(cxcywh, img_w=100, img_h=100)
    denorm = denormalize_boxes(norm, img_w=100, img_h=100)
    assert torch.allclose(cxcywh, denorm), "normalize round-trip failed"
    print(f"normalized: {norm}")

    # state space round-trip
    state = cxcywh_to_state(norm)
    norm_back = state_to_cxcywh(state)
    assert torch.allclose(norm, norm_back, atol=1e-6), "state round-trip failed"
    print(f"state [cx,cy,log_w,log_h]: {state}")

    # IoU
    a = torch.tensor([[0., 0., 10., 10.]])
    b = torch.tensor([[5., 5., 15., 15.], [20., 20., 30., 30.]])
    iou = box_iou(a, b)
    print(f"IoU: {iou}")  # [25/175, 0]
    assert iou[0, 1] == 0.0, "non-overlapping IoU should be 0"

    print("All checks passed.")
