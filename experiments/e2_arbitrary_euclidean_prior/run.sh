#!/usr/bin/env bash
# e2 — 2×2 ablation: (state N(0,I) vs arbitrary cxcywh clip) × (state interp vs cxcywh interp).
# 4 variants: riemannian / euclidean / riemannian_arb_prior / euclidean_arb_prior.
# 학습 전 target field 수렴성(Lipschitz, ||u_t||) 분석도 함께.
set -euo pipefail

EXP_DIR="experiments/e2_arbitrary_euclidean_prior"
OUT_ROOT="outputs/e2_arbitrary_euclidean_prior"
export PYTHONPATH="${PYTHONPATH:-.}:."

mkdir -p "$OUT_ROOT"

# ── 0. Target field analysis (학습 없이, Lipschitz / 수렴성 지표) ───────────────
echo "=============================================="
echo " Target field analysis (Lipschitz / 1-over-w)"
echo "=============================================="
python "$EXP_DIR/analyze_target_field.py" \
  --out_dir "$OUT_ROOT/analysis"

# ── 1. 4 variants training ────────────────────────────────────────────────────
rows=(
  "riemannian|$EXP_DIR/variants/riemannian.yaml"
  "euclidean|$EXP_DIR/variants/euclidean.yaml"
  "riemannian_arb_prior|$EXP_DIR/variants/riemannian_arb_prior.yaml"
  "euclidean_arb_prior|$EXP_DIR/variants/euclidean_arb_prior.yaml"
)

for row in "${rows[@]}"; do
  IFS='|' read -r tag cfg <<< "$row"
  out="$OUT_ROOT/$tag"
  echo ""
  echo "=============================================="
  echo " Variant: $tag   (out=$out)"
  echo "=============================================="
  python script/overfit_mnist_box.py \
    --config "$cfg" \
    --tag "$tag" \
    --max_steps 5000 --lr 3e-4 --lr_schedule cosine --ode_steps 50 \
    --log_interval 1000 \
    --out_dir "$out" \
    --no-show
done

# ── 2. Loss curve compare (4 variants) ────────────────────────────────────────
echo ""
echo "=============================================="
echo " Loss curve compare (robustness)"
echo "=============================================="
python script/plot_loss_compare.py \
  --variants riemannian:"$OUT_ROOT/riemannian/loss_log.txt" \
             euclidean:"$OUT_ROOT/euclidean/loss_log.txt" \
             riemannian_arb_prior:"$OUT_ROOT/riemannian_arb_prior/loss_log.txt" \
             euclidean_arb_prior:"$OUT_ROOT/euclidean_arb_prior/loss_log.txt" \
  --out "$OUT_ROOT/loss_compare.png" \
  --title "e2 2x2: (prior) x (interp) — 1-image overfit (5000 step, cosine, ODE 50)"

# ── 3. 4-panel trajectory GIF ─────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " Trajectory-compare GIF (4-panel)"
echo "=============================================="
python script/trajectory_gif.py \
  --variants 'riemannian:Rm | state prior' \
             'linear:Eu | state prior (log-normal)' \
             'riemannian_arb_prior:Rm | arb prior' \
             'linear_arb_prior:Eu | arb prior' \
  --train_steps 5000 --lr 3e-4 --lr_schedule cosine --ode_steps 50 --fps 12 \
  --out_dir "$OUT_ROOT/gif"

# canonical 위치(docs/assets/)에 e2 prefix 로 동기화 — README/report 임베드용
mkdir -p docs/assets
cp "$OUT_ROOT/gif/trajectory_compare.gif" docs/assets/e2_trajectory_compare.gif
cp "$OUT_ROOT/gif/frame_t_0.00.png" docs/assets/e2_frame_t_0.00.png
cp "$OUT_ROOT/gif/frame_t_0.50.png" docs/assets/e2_frame_t_0.50.png
cp "$OUT_ROOT/gif/frame_t_1.00.png" docs/assets/e2_frame_t_1.00.png
echo "[sync] outputs → docs/assets/ (e2_* prefix)"

echo ""
echo "=============================================="
echo " Results summary"
echo "=============================================="
for row in "${rows[@]}"; do
  IFS='|' read -r tag _ <<< "$row"
  echo "--- $tag ---"
  cat "$OUT_ROOT/$tag/report.json"
  echo ""
done
