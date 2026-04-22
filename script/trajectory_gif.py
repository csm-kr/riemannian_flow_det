"""
Trajectory GIF: Riemannian (state space flow, ours) vs Linear (Euclidean cxcywh).

1장 MNIST Box 샘플에서 두 trajectory를 각각 overfit → ODE 20 step 궤적을 캡처
→ 좌 Riemannian / 우 Linear 나란히 frame 만들어 GIF 저장.

Canvas는 MNIST 256을 중앙에 두고 bigger (pad)로 확장해 init box(이미지 밖)도 보이게.

실행:
    python script/trajectory_gif.py                      # 기본 (steps 1500)
    python script/trajectory_gif.py --train_steps 2000 --ode_steps 30
"""

import argparse
import os
import time
from pathlib import Path

import yaml
import numpy as np
import cv2
import torch
from PIL import Image

from dataset import build_dataloader
from dataset.mnist_box import denormalize_image
from dataset.box_ops import cxcywh_to_xyxy, state_to_cxcywh
from model import build_model


# ────────────────────────────────────────────────
# config
# ────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ────────────────────────────────────────────────
# 학습 + 궤적 캡처
# ────────────────────────────────────────────────

def train_model(
    cfg:         dict,
    trajectory:  str,
    batch:       dict,
    device:      torch.device,
    steps:       int,
    lr:          float,
    seed:        int = 0,
    lr_schedule: str = "const",
) -> torch.nn.Module:
    """trajectory='riemannian' | 'linear' 로 1장 overfit."""
    torch.manual_seed(seed); np.random.seed(seed)
    cfg["trajectory"] = trajectory
    model = build_model(cfg).to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=cfg.get("weight_decay", 0.0))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps, eta_min=lr * 0.05)
        if lr_schedule == "cosine" else None
    )
    images = batch["images"].to(device)
    boxes  = [b.to(device) for b in batch["boxes"]]

    t0 = time.time()
    for step in range(1, steps + 1):
        out  = model.forward_train(images, boxes)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if scheduler is not None:
            scheduler.step()
        if step % max(steps // 5, 1) == 0 or step == 1:
            print(f"  [{trajectory:10s}] step {step:4d}/{steps}  "
                  f"loss={loss.item():.4f}  elapsed={time.time()-t0:.1f}s")
    return model


@torch.no_grad()
def ode_trace(
    model:     torch.nn.Module,
    images:    torch.Tensor,
    num_steps: int,
    seed:      int = 0,
) -> list[torch.Tensor]:
    """
    forward_inference를 그대로 흉내내면서 각 Euler step의 normalized cxcywh를 저장.
    Returns: list of length (num_steps + 1), 각 원소 [1, Q, 4] normalized cxcywh.
    """
    model.eval()
    device = images.device
    Q      = model.num_queries

    torch.manual_seed(seed)
    b  = torch.randn(1, Q, 4, device=device)  # state space
    dt = 1.0 / num_steps

    states = [state_to_cxcywh(b).cpu()]  # t=0
    for i in range(num_steps):
        t_val = i / num_steps
        t     = torch.full((1,), t_val, device=device)
        tokens = model.flow_dit(images, b, t)
        v      = model.box_head(tokens)
        b      = model.trajectory.ode_step(b, v, dt)
        states.append(state_to_cxcywh(b).cpu())
    return states


# ────────────────────────────────────────────────
# 시각화 (확대 canvas)
# ────────────────────────────────────────────────

_GT_COLOR = (0, 255, 255)  # yellow
_CLASS_COLORS = [
    ( 66, 135, 245), (245, 158,  66), ( 66, 245, 143), (245,  66, 203),
    ( 66, 245, 245), (203,  66, 245), (245,  66,  66), (143, 245,  66),
    (245, 245,  66), (100, 100, 245),
]


def _clip_rect(x1, y1, x2, y2, W, H, margin=4):
    """canvas를 벗어나는 좌표를 경계로 clip (그려질 수 있도록)."""
    x1 = int(np.clip(x1, -margin, W + margin))
    y1 = int(np.clip(y1, -margin, H + margin))
    x2 = int(np.clip(x2, -margin, W + margin))
    y2 = int(np.clip(y2, -margin, H + margin))
    return x1, y1, x2, y2


def make_frame(
    mnist_bgr:   np.ndarray,    # [256, 256, 3] BGR (MNIST canvas)
    gt_cxcywh:   torch.Tensor,  # [10, 4] normalized cxcywh
    q_cxcywh:    torch.Tensor,  # [10, 4] normalized cxcywh (current b_t)
    t_val:       float,
    title:       str,
    canvas_size: int = 768,
    img_size:    int = 256,
) -> np.ndarray:
    """
    확장된 canvas 중앙에 MNIST 이미지 + GT(노랑) + 현재 query box(클래스 색상)를 그림.
    Returns: BGR uint8 [canvas_size, canvas_size, 3]
    """
    H = W = canvas_size
    canvas = np.full((H, W, 3), 24, dtype=np.uint8)  # 어두운 회색 배경

    # MNIST 이미지 중앙 배치
    x_off = (W - img_size) // 2
    y_off = (H - img_size) // 2
    canvas[y_off:y_off + img_size, x_off:x_off + img_size] = mnist_bgr

    # MNIST 영역 경계선 (박스 밖/안 구분용)
    cv2.rectangle(canvas, (x_off - 1, y_off - 1),
                  (x_off + img_size, y_off + img_size), (90, 90, 90), 1)

    # GT 그리기 (노랑)
    gt_px = cxcywh_to_xyxy(gt_cxcywh) * img_size
    for i in range(gt_px.shape[0]):
        x1, y1, x2, y2 = gt_px[i].tolist()
        x1 += x_off; y1 += y_off; x2 += x_off; y2 += y_off
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                      _GT_COLOR, 1)

    # 현재 query 박스 그리기 (클래스 색상, clip)
    q_px = cxcywh_to_xyxy(q_cxcywh) * img_size
    for i in range(q_px.shape[0]):
        x1, y1, x2, y2 = q_px[i].tolist()
        x1 += x_off; y1 += y_off; x2 += x_off; y2 += y_off
        x1, y1, x2, y2 = _clip_rect(x1, y1, x2, y2, W, H)
        color = _CLASS_COLORS[i % len(_CLASS_COLORS)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        # 라벨은 박스 좌상단 근처(canvas 내부 강제)
        tx = int(np.clip(x1 + 2, 4, W - 20))
        ty = int(np.clip(y1 - 4, 14, H - 4))
        cv2.putText(canvas, str(i), (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 타이틀 & t 값
    cv2.putText(canvas, title, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"t = {t_val:.2f}", (12, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 4) -> np.ndarray:
    H, W, _ = left.shape
    sep = np.full((H, gap, 3), 60, dtype=np.uint8)
    return np.concatenate([left, sep, right], axis=1)


# ────────────────────────────────────────────────
# GIF 저장
# ────────────────────────────────────────────────

def save_gif(frames: list[np.ndarray], path: str,
             fps: int = 10, hold_last: int = 10) -> None:
    """
    frames: list of BGR uint8 np.ndarray
    hold_last: 마지막 프레임을 몇 번 반복해서 끝에 잠시 머무르게 할지
    """
    pil_frames = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))
    pil_frames.extend([pil_frames[-1]] * hold_last)

    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )


# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/mnist_box.yaml")
    parser.add_argument("--train_steps", type=int,   default=1500,
                        help="각 trajectory 모델의 overfit step 수")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lr_schedule", default="const", choices=["const", "cosine"])
    parser.add_argument("--ode_steps",   type=int,   default=20,
                        help="ODE Euler step 수 (= GIF 프레임 수)")
    parser.add_argument("--canvas_size", type=int,   default=768,
                        help="확장 canvas 한 변 픽셀 (init box가 보이도록 여유)")
    parser.add_argument("--fps",         type=int,   default=8)
    parser.add_argument("--sample_idx",  type=int,   default=0)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--out_dir",     default="outputs/trajectory_gif")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}  [out] {out_dir}")

    # 1. 데이터 로드 (1장)
    cfg = load_config(args.config)
    cfg["mnist_num_samples_train"] = 1
    cfg["batch_size"]   = 1
    cfg["num_workers"]  = 0
    loader = build_dataloader(cfg, split="train")
    batch  = next(iter(loader))

    images_gpu = batch["images"].to(device)
    gt_cxcywh  = batch["boxes"][0]               # [10, 4] normalized cxcywh
    mnist_bgr  = denormalize_image(batch["images"][0])

    # 2. 두 trajectory로 각각 overfit
    print("\n=== Train Riemannian (ours) ===")
    m_rie = train_model(cfg, "riemannian", batch, device,
                        args.train_steps, args.lr, args.seed,
                        lr_schedule=args.lr_schedule)

    print("\n=== Train Linear (Euclidean baseline) ===")
    m_lin = train_model(cfg, "linear", batch, device,
                        args.train_steps, args.lr, args.seed,
                        lr_schedule=args.lr_schedule)

    # 3. 같은 b_0 seed로 ODE 궤적 캡처
    print("\n=== ODE trace ===")
    traj_rie = ode_trace(m_rie, images_gpu, args.ode_steps, seed=args.seed)
    traj_lin = ode_trace(m_lin, images_gpu, args.ode_steps, seed=args.seed)
    print(f"  captured {len(traj_rie)} frames per method (t=0 → t=1)")

    # 4. frame 합성
    frames = []
    for i in range(len(traj_rie)):
        t_val = i / args.ode_steps
        frm_l = make_frame(mnist_bgr, gt_cxcywh, traj_rie[i][0],
                           t_val, "Riemannian (ours)",
                           canvas_size=args.canvas_size)
        frm_r = make_frame(mnist_bgr, gt_cxcywh, traj_lin[i][0],
                           t_val, "Linear (Euclidean)",
                           canvas_size=args.canvas_size)
        frames.append(side_by_side(frm_l, frm_r))

    # 5. 저장
    gif_path = out_dir / "trajectory_compare.gif"
    save_gif(frames, str(gif_path), fps=args.fps, hold_last=args.fps * 2)
    print(f"\n[saved] GIF → {gif_path}")
    print(f"  frames: {len(frames)}  size: {frames[0].shape}  fps: {args.fps}")

    # 참고용: 첫/중간/마지막 PNG도 저장
    key_frames = {"t=0.00": 0, f"t={0.5:.2f}": len(frames) // 2,
                  "t=1.00": len(frames) - 1}
    for name, idx in key_frames.items():
        p = out_dir / f"frame_{name.replace('=', '_')}.png"
        cv2.imwrite(str(p), frames[idx])
        print(f"  [png] {name} → {p}")


if __name__ == "__main__":
    main()
