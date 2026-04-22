"""
MB5: MNIST Box single-image overfit (Phase 2.5 sanity check).

목적: 1장의 MNIST Box 샘플에 모델이 완전히 overfit되는지 확인 →
      모델·loss·trajectory 구현 정합성을 격리 검증.
범위: Phase 3 본 `script/train.py`가 아님. MB5 전용 최소 루프.

실행:
    python script/overfit_mnist_box.py                             # 기본 2000 step
    python script/overfit_mnist_box.py --max_steps 500 --lr 3e-4
    python script/overfit_mnist_box.py --num_samples 4             # 4장에 overfit
"""

import argparse
import os
import time
from pathlib import Path

import yaml
import torch
import numpy as np
import cv2

from dataset import build_dataloader
from dataset.mnist_box import draw_sample, denormalize_image
from dataset.box_ops import cxcywh_to_xyxy, state_to_cxcywh
from model import build_model


# ────────────────────────────────────────────────
# config 로딩
# ────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ────────────────────────────────────────────────
# 시각화: 한 샘플의 GT vs prediction
# ────────────────────────────────────────────────

_GT_COLOR   = (  0, 255, 255)  # yellow
_PRED_COLOR = (  0, 255,   0)  # green
_INIT_COLOR = (255, 128, 128)  # light blue  (b_0)


def _draw_boxes_xyxy(
    img: np.ndarray, xyxy: torch.Tensor, color: tuple,
    labels: list | None = None, clip: bool = True,
) -> None:
    """xyxy pixel 박스들을 img에 inplace로 그림. clip=True면 이미지 밖은 잘라서라도 그림."""
    H, W = img.shape[:2]
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i].tolist()
        if clip:
            x1 = max(-10, min(W + 10, x1)); x2 = max(-10, min(W + 10, x2))
            y1 = max(-10, min(H + 10, y1)); y2 = max(-10, min(H + 10, y2))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        if labels is not None:
            cv2.putText(img, str(labels[i]), (x1, max(y1 - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def draw_gt_vs_pred(
    sample:     dict,
    pred_boxes: torch.Tensor,       # [10, 4] normalized cxcywh
    init_boxes: torch.Tensor | None = None,   # [10, 4] normalized cxcywh (state→cxcywh 변환 후)
    image_size: int = 256,
) -> np.ndarray:
    """
    GT (노랑) + final pred (녹색) + 선택적으로 init b_0 (옅은 파랑) 시각화.
    init은 이미지 밖 좌표라도 canvas 가장자리 근처로 clip해서 보이게 그림.
    """
    img = denormalize_image(sample["image"])
    gt  = cxcywh_to_xyxy(sample["boxes"].clone()) * image_size
    pr  = cxcywh_to_xyxy(pred_boxes.cpu().clone()) * image_size

    if init_boxes is not None:
        init = cxcywh_to_xyxy(init_boxes.cpu().clone()) * image_size
        _draw_boxes_xyxy(img, init, _INIT_COLOR,
                         labels=list(range(init.shape[0])), clip=True)
    _draw_boxes_xyxy(img, gt, _GT_COLOR, labels=list(range(gt.shape[0])), clip=False)
    _draw_boxes_xyxy(img, pr, _PRED_COLOR, labels=None, clip=False)
    return img


# ────────────────────────────────────────────────
# Overfit loop
# ────────────────────────────────────────────────

def run_overfit(cfg: dict, args: argparse.Namespace) -> dict:
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 재현성
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Dataloader: num_samples를 overfit 크기로
    cfg["mnist_num_samples_train"] = args.num_samples
    cfg["batch_size"] = args.num_samples   # 전부 한 배치
    cfg["num_workers"] = 0                 # 소규모이므로 main thread
    loader = build_dataloader(cfg, split="train")
    batch  = next(iter(loader))
    images = batch["images"].to(device)
    boxes  = [b.to(device) for b in batch["boxes"]]
    print(f"[data] images={tuple(images.shape)}  num_samples={args.num_samples}")

    # Model
    model = build_model(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg['backbone_type']}  params={n_param/1e6:.2f}M")

    # Optimizer
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=cfg.get("weight_decay", 0.0),
    )

    # LR scheduler
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.max_steps, eta_min=args.lr * 0.05,
        )
    else:
        scheduler = None

    # 학습 루프
    model.train()
    log = []
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        out = model.forward_train(images, boxes)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=cfg.get("grad_clip_norm", 1.0)
        )
        optim.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_interval == 0 or step == 1:
            sec = time.time() - t0
            ips = step / max(sec, 1e-6)
            lr_now = optim.param_groups[0]["lr"]
            print(f"  step {step:5d}/{args.max_steps}  loss={loss.item():.5f}  "
                  f"lr={lr_now:.2e}  elapsed={sec:.1f}s  ({ips:.1f} it/s)")
        log.append((step, float(loss.item())))

    # ── 추론: 학습된 모델로 같은 image에 ODE 돌려 box 예측 ───
    model.eval()
    with torch.no_grad():
        # 같은 seed로 b_0 샘플링해서 viz에도 사용 (정확히 같은 궤적)
        torch.manual_seed(cfg["seed"])
        b0_state = torch.randn(1, 10, 4, device=device)  # [1, 10, 4] state
        init_boxes = state_to_cxcywh(b0_state)[0].cpu()   # [10, 4] normalized cxcywh

        # forward_inference는 내부에서 다시 randn을 뽑으므로, 여기서도 같은 seed 보장 위해 한 번 더.
        torch.manual_seed(cfg["seed"])
        pred_boxes = model.forward_inference(
            images[:1], num_steps=args.ode_steps, num_queries=10,
        )[0].cpu()  # [10, 4] normalized cxcywh

    # ── 결과 저장 ──────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # loss curve (텍스트 로그)
    with open(out_dir / "loss_log.txt", "w") as f:
        f.write("step\tloss\n")
        for s, l in log:
            f.write(f"{s}\t{l:.6f}\n")

    # GT vs prediction + init b_0 시각화
    sample0 = {"image": batch["images"][0], "boxes": batch["boxes"][0],
               "labels": batch["labels"][0]}
    vis = draw_gt_vs_pred(sample0, pred_boxes, init_boxes=init_boxes)
    cv2.imwrite(str(out_dir / "overfit_gt_vs_pred.png"), vis)

    # init box 통계 (정말 random 분포인지 확인용)
    init_wh_px = init_boxes[:, 2:] * 256
    print(f"[init b_0] cxcywh pixel 통계:")
    print(f"  cx/cy px: min={(init_boxes[:, :2]*256).min():.1f}  "
          f"max={(init_boxes[:, :2]*256).max():.1f}")
    print(f"  w/h  px: min={init_wh_px.min():.1f}  max={init_wh_px.max():.1f}  "
          f"(canvas=256, 대부분 화면 밖)")

    # GT / pred 수치 요약
    err = (pred_boxes - sample0["boxes"]).abs()
    # 마지막 100 step loss 평균 — t 샘플링 노이즈 완화
    tail_loss = float(np.mean([l for _, l in log[-100:]])) if len(log) >= 100 \
                else float(np.mean([l for _, l in log]))
    report = {
        "tag":          args.tag,
        "trajectory":   cfg.get("trajectory", "riemannian"),
        "hidden_dim":   cfg.get("hidden_dim", 256),
        "num_layers":   cfg.get("num_layers", 6),
        "lr":           args.lr,
        "lr_schedule":  args.lr_schedule,
        "max_steps":    args.max_steps,
        "ode_steps":    args.ode_steps,
        "num_samples":  args.num_samples,
        "final_loss":   log[-1][1],
        "tail100_loss": tail_loss,
        "initial_loss": log[0][1],
        "mean_box_err": float(err.mean().item()),
        "max_box_err":  float(err.max().item()),
        "mean_err_px":  float(err.mean().item() * 256),
        "max_err_px":   float(err.max().item() * 256),
    }
    with open(out_dir / "report.txt", "w") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")
    import json
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report, vis


# ────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/mnist_box.yaml")
    parser.add_argument("--num_samples",  type=int,   default=1,
                        help="overfit할 샘플 수 (기본 1)")
    parser.add_argument("--max_steps",    type=int,   default=2000)
    parser.add_argument("--lr",           type=float, default=1.0e-4)
    parser.add_argument("--lr_schedule",  default="const", choices=["const", "cosine"],
                        help="cosine은 lr → lr*0.05로 감쇠")
    parser.add_argument("--log_interval", type=int,   default=100)
    parser.add_argument("--ode_steps",    type=int,   default=10)
    parser.add_argument("--tag",          default="",
                        help="결과 JSON에 기록될 실험 태그")
    parser.add_argument("--out_dir",      default="outputs/mb5_overfit")
    parser.add_argument("--show",         action=argparse.BooleanOptionalAction,
                        default=False,
                        help="학습 후 GT vs pred 창 띄우기")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[config] {args.config}")
    report, vis = run_overfit(cfg, args)

    print("\n=== MB5 report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"  [saved] → {args.out_dir}/")

    if args.show:
        cv2.imshow("GT (yellow) vs Pred (green)", vis)
        print("press any key to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
