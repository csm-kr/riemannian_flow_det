"""
Aggregate multi-seed results from outputs/.../multiseed/<variant>/seed{0,1,2}/report.json
→ mean ± std table (markdown + JSON).
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_runs(root: Path, variant: str) -> list[dict]:
    runs = []
    for seed_dir in sorted(root.glob(f"{variant}/seed*")):
        rpt = seed_dir / "report.json"
        if rpt.exists():
            with open(rpt) as f:
                runs.append(json.load(f))
    return runs


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--keys",
                        default="tail100_loss,mean_err_px,max_err_px,final_loss",
                        help="report.json 에서 집계할 키 CSV")
    args = parser.parse_args()

    root = Path(args.root)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]

    table = []
    all_stats = {}
    for variant in args.variants:
        runs = load_runs(root, variant)
        if not runs:
            print(f"[warn] no runs for {variant}")
            continue
        row = {"variant": variant, "n_seeds": len(runs)}
        for k in keys:
            vals = [r[k] for r in runs if k in r]
            mean, std = summarize(vals)
            row[f"{k}_mean"] = mean
            row[f"{k}_std"]  = std
        table.append(row)
        all_stats[variant] = {
            "n_seeds": len(runs),
            "per_seed": {k: [r.get(k) for r in runs] for k in keys},
            "mean":   {k: row[f"{k}_mean"] for k in keys},
            "std":    {k: row[f"{k}_std"]  for k in keys},
        }

    # Markdown
    header = ["variant", "n"] + [f"{k} mean±std" for k in keys]
    sep    = ["---"] * len(header)
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in table:
        cells = [row["variant"], str(row["n_seeds"])]
        for k in keys:
            cells.append(f"{row[f'{k}_mean']:.3f} ± {row[f'{k}_std']:.3f}")
        md.append("| " + " | ".join(cells) + " |")
    md_text = "\n".join(md) + "\n"
    print(md_text)

    # save
    out_md   = root / "aggregate.md"
    out_json = root / "aggregate.json"
    with open(out_md,   "w") as f: f.write(md_text)
    with open(out_json, "w") as f: json.dump(all_stats, f, indent=2)
    print(f"[saved] {out_md}\n[saved] {out_json}")


if __name__ == "__main__":
    main()
