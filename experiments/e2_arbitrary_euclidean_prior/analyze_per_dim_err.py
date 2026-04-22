"""
e2 follow-up: per-dim error 분석.

관찰: Riemannian 결과에서 **박스 크기는 잘 맞는데 위치가 어긋난다**.
가설: state space 에서 MSE loss 는 4 dim (cx, cy, log_w, log_h) 을 동등하게
보지만, cxcywh 로 변환 시 `w = exp(log_w)` 에서 d(w) = w·d(log_w) 이므로
**size 오차는 w ≈ 0.14 배로 축소, position 오차는 그대로**. 같은 state-space
regression 오차가 cxcywh 공간에서 **position 이 size 대비 ~3× 크게 보인다**.

검증: 훈련된 riemannian / riemannian_arb_prior 를 각각 학습 후 추론,
per-dim (cx, cy, w, h) 오차를 픽셀 단위로 분리 집계.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import build_dataloader
from dataset.box_ops import state_to_cxcywh
from model import build_model


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_and_eval(cfg: dict, steps: int = 5000, lr: float = 3e-4,
                   seed: int = 0, device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg["mnist_num_samples_train"] = 1
    cfg["batch_size"] = 1
    cfg["num_workers"] = 0
    loader = build_dataloader(cfg, split="train")
    batch  = next(iter(loader))
    images = batch["images"].to(device)
    boxes  = [b.to(device) for b in batch["boxes"]]

    model = build_model(cfg).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    model.train()
    for step in range(1, steps + 1):
        out = model.forward_train(images, boxes)
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    # Inference
    model.eval()
    with torch.no_grad():
        torch.manual_seed(seed)
        pred = model.forward_inference(images[:1], num_steps=50, num_queries=10)[0].cpu()
    gt = batch["boxes"][0].cpu()   # [10, 4] normalized cxcywh
    return pred, gt


def summarize_per_dim(pred: torch.Tensor, gt: torch.Tensor, img_size: int = 256) -> dict:
    """Per-dim abs error in normalized + pixel units."""
    err_norm = (pred - gt).abs()               # [10, 4] in normalized cxcywh
    err_px   = err_norm * img_size
    names = ["cx", "cy", "w", "h"]
    out = {}
    for i, n in enumerate(names):
        e = err_px[:, i].numpy()
        out[n] = {
            "mean": float(e.mean()),
            "max":  float(e.max()),
            "p50":  float(np.percentile(e, 50)),
            "p90":  float(np.percentile(e, 90)),
        }
    # Aggregated
    pos_err  = err_px[:, :2].norm(dim=-1).numpy()   # sqrt(cx²+cy²) per box (pixel)
    size_err = err_px[:, 2:].norm(dim=-1).numpy()
    out["pos_l2"]  = {"mean": float(pos_err.mean()),  "max": float(pos_err.max())}
    out["size_l2"] = {"mean": float(size_err.mean()), "max": float(size_err.max())}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", nargs="+",
        default=[
            "riemannian:experiments/e2_arbitrary_euclidean_prior/variants/riemannian.yaml",
            "riemannian_arb_prior:experiments/e2_arbitrary_euclidean_prior/variants/riemannian_arb_prior.yaml",
        ],
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr",    type=float, default=3e-4)
    parser.add_argument("--out_dir", default="outputs/e2_arbitrary_euclidean_prior/per_dim")
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for spec in args.variants:
        name, cfg_path = spec.split(":", 1)
        print(f"\n=== {name}  (cfg={cfg_path}, {args.steps} step) ===")
        cfg = load_config(cfg_path)
        pred, gt = train_and_eval(cfg, args.steps, args.lr, device=device)
        stats = summarize_per_dim(pred, gt)
        results[name] = stats

        print(f"  per-dim abs err (px):")
        print(f"    cx  mean={stats['cx']['mean']:6.2f}  max={stats['cx']['max']:6.2f}  p90={stats['cx']['p90']:6.2f}")
        print(f"    cy  mean={stats['cy']['mean']:6.2f}  max={stats['cy']['max']:6.2f}  p90={stats['cy']['p90']:6.2f}")
        print(f"    w   mean={stats['w']['mean']:6.2f}  max={stats['w']['max']:6.2f}  p90={stats['w']['p90']:6.2f}")
        print(f"    h   mean={stats['h']['mean']:6.2f}  max={stats['h']['max']:6.2f}  p90={stats['h']['p90']:6.2f}")
        print(f"  L2 aggregate:")
        print(f"    pos (cx,cy)  mean={stats['pos_l2']['mean']:6.2f}  max={stats['pos_l2']['max']:6.2f}")
        print(f"    size (w,h)   mean={stats['size_l2']['mean']:6.2f}  max={stats['size_l2']['max']:6.2f}")
        print(f"    ratio pos/size  = {stats['pos_l2']['mean']/max(stats['size_l2']['mean'], 1e-6):.2f}")

    with open(out_dir / "per_dim_err.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_dir/'per_dim_err.json'}")


if __name__ == "__main__":
    main()
