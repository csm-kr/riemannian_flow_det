"""
Compare training loss curves across multiple variants (e.g., Riemannian vs Euclidean).

Reads each variant's `loss_log.txt` (tab-separated: step\tloss), draws raw +
EMA-smoothed curves side by side, and reports tail statistics to judge robustness.

실행:
    python script/plot_loss_compare.py \
        --variants riemannian:outputs/e1_unified_prior_fair_compare/riemannian/loss_log.txt \
                   euclidean:outputs/e1_unified_prior_fair_compare/euclidean/loss_log.txt \
        --out outputs/e1_unified_prior_fair_compare/loss_compare.png \
        --title "Riemannian vs Euclidean — 1-image overfit (5000 step)"
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# variant 색상 (GIF의 타이틀 컨벤션과 일관)
_COLORS = {
    "riemannian": "#ea5547",   # warm red
    "euclidean":  "#3b82f6",   # cool blue
    "linear":     "#3b82f6",   # alias
}
_DEFAULT_COLOR = "#808080"


def load_loss(path: str) -> tuple[np.ndarray, np.ndarray]:
    """step, loss 두 배열 반환 (헤더 한 줄 skip)."""
    data = np.loadtxt(path, delimiter="\t", skiprows=1, dtype=np.float64)
    return data[:, 0].astype(int), data[:, 1]


def ema(x: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """지수 이동 평균 (alpha=0.02 → 대략 50 step 윈도우)."""
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def tail_stats(loss: np.ndarray, tail_frac: float = 0.1) -> dict:
    tail = loss[-max(int(len(loss) * tail_frac), 1):]
    return {
        "mean":   float(tail.mean()),
        "median": float(np.median(tail)),
        "std":    float(tail.std()),
        "p90":    float(np.percentile(tail, 90)),
        "p99":    float(np.percentile(tail, 99)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", nargs="+", required=True,
        help="포맷: name:path[:color]  (예: riemannian:path/to/log.txt:#ea5547)",
    )
    parser.add_argument("--out",   default="outputs/loss_compare.png")
    parser.add_argument("--title", default="Loss comparison")
    parser.add_argument("--ema",   type=float, default=0.02,
                        help="EMA alpha (작을수록 부드럽게)")
    parser.add_argument("--yscale", default="log", choices=["log", "linear"])
    args = parser.parse_args()

    # Parse --variants
    entries = []
    for v in args.variants:
        parts = v.split(":")
        name, path = parts[0], parts[1]
        color = parts[2] if len(parts) > 2 else _COLORS.get(name, _DEFAULT_COLOR)
        entries.append((name, path, color))

    # Figure 2 panels: [raw+EMA] + [smoothed zoom on tail]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=120)
    ax_full, ax_tail = axes

    summary_rows = []
    for name, path, color in entries:
        step, loss = load_loss(path)
        loss_ema = ema(loss, args.ema)
        stats = tail_stats(loss, tail_frac=0.1)

        # Left: raw (faded) + EMA (solid)
        ax_full.plot(step, loss, color=color, alpha=0.22, linewidth=0.5,
                     label=f"{name} (raw)")
        ax_full.plot(step, loss_ema, color=color, linewidth=2.0,
                     label=f"{name} (EMA α={args.ema})")

        # Right: EMA only, zoom to last 40%
        cut = int(len(step) * 0.6)
        ax_tail.plot(step[cut:], loss_ema[cut:], color=color, linewidth=2.0,
                     label=f"{name}")
        # tail variance band (mean ± std)
        ax_tail.fill_between(
            step[cut:],
            loss_ema[cut:] - loss[cut:].std() * 0.3,
            loss_ema[cut:] + loss[cut:].std() * 0.3,
            color=color, alpha=0.12,
        )

        summary_rows.append((name, stats))

    ax_full.set_yscale(args.yscale)
    ax_full.set_xlabel("step")
    ax_full.set_ylabel("loss")
    ax_full.set_title("Full trajectory")
    ax_full.legend(loc="upper right", fontsize=9)
    ax_full.grid(True, alpha=0.3)

    ax_tail.set_yscale(args.yscale)
    ax_tail.set_xlabel("step")
    ax_tail.set_ylabel("loss (EMA)")
    ax_tail.set_title("Tail zoom (last 40%)")
    ax_tail.legend(loc="upper right", fontsize=9)
    ax_tail.grid(True, alpha=0.3)

    fig.suptitle(args.title, fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[saved] {args.out}")

    # Print summary table
    print("\nTail (last 10% of steps) statistics:")
    print(f"{'variant':<14}  {'mean':>8}  {'median':>8}  {'std':>8}  {'p90':>8}  {'p99':>8}")
    for name, s in summary_rows:
        print(f"{name:<14}  {s['mean']:>8.4f}  {s['median']:>8.4f}  "
              f"{s['std']:>8.4f}  {s['p90']:>8.4f}  {s['p99']:>8.4f}")


if __name__ == "__main__":
    main()
