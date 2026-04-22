"""
e2: target field 수렴성 / Lipschitz 분석 (학습 없이).

목적
----
3가지 prior 설정에서 same GT 분포 기준으로 (b0, b1, t) 샘플 → u_t 계산,
u_t 의 크기·Lipschitz·1/w 발산 빈도를 비교해 **학습 전에** 각 설정의
수학적 난이도를 정량화한다.

세 variant
----------
1. riemannian            : state N(0,I) prior, state interp, u_t = b1-b0 (const in x_t)
2. euclidean             : state N(0,I) prior, cxcywh interp, u_t_w = Δw/w_t
3. euclidean_arb_prior   : cxcywh clip(N(0.5, 1/6), 0.02, 1) prior, cxcywh interp

산출물
------
- {out_dir}/target_field_stats.json  : 전체 수치
- {out_dir}/target_field_stats.md    : markdown 표
- {out_dir}/target_field_stats.png   : ||u_t|| histogram + min(w_t) CDF
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from dataset.box_ops import cxcywh_to_state
from model.trajectory import (
    RiemannianTrajectory,
    RiemannianTrajectoryArbPrior,
    LinearTrajectory,
    LinearTrajectoryArbPrior,
)


SCALE_RANGE = (14.0 / 256.0, 56.0 / 256.0)   # MNIST Box setting
CENTER_RANGE = (0.1, 0.9)


def sample_gt_boxes(N: int, device) -> torch.Tensor:
    """MNIST-box-like GT cxcywh."""
    cx = torch.empty(N, device=device).uniform_(*CENTER_RANGE)
    cy = torch.empty(N, device=device).uniform_(*CENTER_RANGE)
    w  = torch.empty(N, device=device).uniform_(*SCALE_RANGE)
    h  = torch.empty(N, device=device).uniform_(*SCALE_RANGE)
    return torch.stack([cx, cy, w, h], dim=-1)


def stats_of(t: torch.Tensor) -> dict:
    arr = t.detach().cpu().numpy()
    return {
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr)),
        "p50":   float(np.percentile(arr, 50)),
        "p90":   float(np.percentile(arr, 90)),
        "p99":   float(np.percentile(arr, 99)),
        "max":   float(np.max(arr)),
    }


def run_variant(name: str, traj, n_samples: int, device, seed: int = 0):
    """sample (b0, b1, t) x n, compute u_t + derived quantities."""
    torch.manual_seed(seed)
    gt = sample_gt_boxes(n_samples, device).unsqueeze(1)    # [n,1,4] cxcywh
    t  = torch.rand(n_samples, device=device)

    u_list, bt_list = [], []
    BATCH = 20_000
    for i in range(0, n_samples, BATCH):
        gt_b = gt[i:i+BATCH]; t_b = t[i:i+BATCH]
        if name in ("riemannian", "riemannian_arb_prior"):
            b1_state = cxcywh_to_state(gt_b)
            b_t, u_t, _ = traj.sample(b1_state, t_b)
        else:
            b_t, u_t = traj.sample(gt_b, t_b)
        u_list.append(u_t.view(-1, 4))
        bt_list.append(b_t.view(-1, 4))   # always state space

    u   = torch.cat(u_list, dim=0)           # [n,4] state
    b_t = torch.cat(bt_list, dim=0)          # [n,4] state
    w_t = b_t[..., 2].exp()                  # state log_w → w
    h_t = b_t[..., 3].exp()

    # ||u_t|| in state space
    u_norm = u.norm(dim=-1)
    min_wh = torch.minimum(w_t, h_t)

    return {
        "u_norm":  u_norm.cpu().numpy(),
        "u_abs":   u.abs().cpu().numpy(),        # [n,4]
        "min_wh":  min_wh.cpu().numpy(),
        "w_t":     w_t.cpu().numpy(),
    }


def assemble_stats(name: str, data: dict) -> dict:
    u_abs = data["u_abs"]
    stats = {
        "name":    name,
        "u_norm":  stats_of(torch.tensor(data["u_norm"])),
        "u_cx":    stats_of(torch.tensor(u_abs[:, 0])),
        "u_cy":    stats_of(torch.tensor(u_abs[:, 1])),
        "u_w":     stats_of(torch.tensor(u_abs[:, 2])),
        "u_h":     stats_of(torch.tensor(u_abs[:, 3])),
        "min_wh":  stats_of(torch.tensor(data["min_wh"])),
    }
    # Empirical Lipschitz for Euclidean variants:
    # u_t_{log_w} = Δw / w_t (per construction in LinearTrajectory*)
    # ∂u_t_{log_w}/∂state.log_w_t = ∂(Δw/w_t)/∂log_w_t · 1
    #                              = (∂/∂w_t Δw/w_t) · w_t
    #                              = (-Δw / w_t²) · w_t
    #                              = -Δw / w_t = -u_t_{log_w}
    # → |dL/dx| = |u_t_{log_w}|
    # For riemannian / riemannian_arb_prior, u_t = b1-b0 independent of x_t → L ≡ 0.
    if name not in ("riemannian", "riemannian_arb_prior"):
        L = np.maximum(u_abs[:, 2], u_abs[:, 3])
        stats["lipschitz_wh"] = stats_of(torch.tensor(L))
    # Small-w tail fractions
    mwh = data["min_wh"]
    stats["frac_wh_lt_0.05"] = float((mwh < 0.05).mean())
    stats["frac_wh_lt_0.02"] = float((mwh < 0.02).mean())
    stats["frac_wh_lt_0.01"] = float((mwh < 0.01).mean())
    return stats


def format_md(all_stats: dict, n_samples: int, arr_min_wh: dict) -> str:
    lines = ["# Target field statistics (no-training analysis)\n"]
    lines.append(
        f"Samples per variant: **{n_samples}**, b₁ ~ uniform in "
        f"cx,cy ∈ [{CENTER_RANGE[0]},{CENTER_RANGE[1]}], "
        f"w,h ∈ [{SCALE_RANGE[0]:.3f},{SCALE_RANGE[1]:.3f}] (MNIST Box).\n"
    )

    lines.append("\n## 1. ||u_t||₂ (state space, what the model regresses)\n")
    lines.append("| variant | mean | std | p50 | p90 | p99 | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, s in all_stats.items():
        u = s["u_norm"]
        lines.append(
            f"| {name} | {u['mean']:.3f} | {u['std']:.3f} | "
            f"{u['p50']:.3f} | {u['p90']:.3f} | {u['p99']:.3f} | {u['max']:.2f} |"
        )

    lines.append("\n## 2. Per-dim |u_t| p99 / max  (state space)\n")
    lines.append("| variant | |u_cx| p99 | |u_cy| p99 | |u_w| p99 | |u_w| max | |u_h| p99 | |u_h| max |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, s in all_stats.items():
        lines.append(
            f"| {name} | {s['u_cx']['p99']:.3f} | {s['u_cy']['p99']:.3f} | "
            f"{s['u_w']['p99']:.3f} | {s['u_w']['max']:.2f} | "
            f"{s['u_h']['p99']:.3f} | {s['u_h']['max']:.2f} |"
        )

    lines.append("\n## 3. min(w_t, h_t) — 1/w 발산 노출\n")
    lines.append("| variant | p50 | p10 | p1 | p0.1 | frac < 0.05 | frac < 0.02 | frac < 0.01 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, s in all_stats.items():
        arr = arr_min_wh[name]
        p10 = float(np.percentile(arr, 10))
        p1  = float(np.percentile(arr, 1))
        p01 = float(np.percentile(arr, 0.1))
        lines.append(
            f"| {name} | {s['min_wh']['p50']:.4f} | {p10:.4f} | {p1:.4f} | {p01:.4f} | "
            f"{s['frac_wh_lt_0.05']*100:.2f}% | {s['frac_wh_lt_0.02']*100:.2f}% | {s['frac_wh_lt_0.01']*100:.3f}% |"
        )

    lines.append("\n## 4. Conditional Lipschitz  L̂(x_t) ≈ |u_t_{log_w}|  (state)\n")
    lines.append("Euclidean: `u_t_{log_w} = Δw / w_t`, so `∂u/∂log_w_t = -u_t_{log_w}` → |L̂| = |u_t_{log_w}|.")
    lines.append("Riemannian: `u_t = b₁ − b₀` does not depend on x_t → L̂ ≡ 0.\n")
    lines.append("| variant | L̂ p50 | p90 | p99 | max |")
    lines.append("|---|---|---|---|---|")
    for name in ("riemannian", "riemannian_arb_prior", "euclidean", "euclidean_arb_prior"):
        if name not in all_stats:
            continue
        if "lipschitz_wh" in all_stats[name]:
            s = all_stats[name]["lipschitz_wh"]
            lines.append(
                f"| {name} | {s['p50']:.3f} | {s['p90']:.3f} | {s['p99']:.3f} | {s['max']:.1f} |"
            )
        else:
            lines.append(f"| {name} | 0 | 0 | 0 | 0 |")

    lines.append(
        "\n## 요약 (2×2 ablation)\n"
        "| prior \\ interp | state | cxcywh |\n"
        "|---|---|---|\n"
        "| state N(0,I) | riemannian: L̂=0, ||u_t|| tight | euclidean: L̂ unbounded (229), log-normal 상단 꼬리 |\n"
        "| arb cxcywh   | riemannian_arb_prior: L̂=0, ||u_t|| 더 tight (p99 3.1) | euclidean_arb_prior: L̂ bounded (14.9), per-dim 신호 작음 |\n"
        "\n"
        "- **state interp 를 쓰면 prior 에 상관없이 Lipschitz 0** — 학습 target 이 x_t 에 무관한 상수.\n"
        "- **cxcywh interp 를 쓰면** 1/w 항이 생겨 Lipschitz 가 prior 에 따라 달라짐. arb prior 는 bounded 라 Lipschitz 도 bounded.\n"
        "- `||u_t||` 자체는 arb prior 쪽이 작다 — transport 거리가 짧음 (prior 가 target 에 가깝기 때문).\n"
    )
    return "\n".join(lines) + "\n"


def plot_stats(out_dir: Path, u_norms: dict, min_whs: dict) -> None:
    if not HAS_PLT:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))

    # (1) ||u_t|| 분포 (log-log)
    lo, hi = 1e-3, max(np.percentile(un, 99.99) for un in u_norms.values()) * 1.5
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)
    for name, un in u_norms.items():
        ax[0].hist(un, bins=bins, alpha=0.45, label=name)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("||u_t||₂  (state space)")
    ax[0].set_ylabel("count (log)")
    ax[0].set_title("Target field magnitude distribution")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")

    # (2) min(w_t) CDF (lower tail in log)
    for name, mwh in min_whs.items():
        srt = np.sort(mwh)
        p   = np.arange(len(srt)) / len(srt)
        ax[1].plot(srt, p, label=name, lw=1.5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("min(w_t, h_t)")
    ax[1].set_ylabel("CDF")
    ax[1].set_title("Lower-tail of interpolated w,h  (1/w blow-up exposure)")
    ax[1].axvline(0.02, color="gray", ls="--", alpha=0.5, label="clip ε")
    ax[1].axvline(0.05, color="orange", ls=":", alpha=0.5, label="0.05")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / "target_field_stats.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",   default="outputs/e2_arbitrary_euclidean_prior/analysis")
    parser.add_argument("--n_samples", type=int, default=200_000)
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    trajs = [
        ("riemannian",           RiemannianTrajectory()),
        ("euclidean",            LinearTrajectory()),
        ("riemannian_arb_prior", RiemannianTrajectoryArbPrior()),
        ("euclidean_arb_prior",  LinearTrajectoryArbPrior()),
    ]

    all_stats, u_norms, min_whs = {}, {}, {}
    for name, traj in trajs:
        print(f"[run] {name}  n={args.n_samples}")
        data = run_variant(name, traj, args.n_samples, device, seed=args.seed)
        all_stats[name] = assemble_stats(name, data)
        u_norms[name]   = data["u_norm"]
        min_whs[name]   = data["min_wh"]

    # Save JSON
    with open(out_dir / "target_field_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # Save Markdown
    md = format_md(all_stats, args.n_samples, min_whs)
    with open(out_dir / "target_field_stats.md", "w") as f:
        f.write(md)
    print(md)

    # Plot
    plot_stats(out_dir, u_norms, min_whs)
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()
