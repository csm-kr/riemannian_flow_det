#!/usr/bin/env bash
# e1 trajectory 공정 비교: 동일 prior + 동일 seed → 유일한 차이는 interpolation 공간
# 실행: bash experiments/e1_unified_prior_fair_compare/run.sh
set -euo pipefail

EXP_DIR="experiments/e1_unified_prior_fair_compare"
OUT_ROOT="outputs/e1_unified_prior_fair_compare"
export PYTHONPATH="${PYTHONPATH:-.}:."

# (tag, config) — tags는 variant 파일명과 일치
rows=(
  "riemannian|$EXP_DIR/variants/riemannian.yaml"
  "euclidean|$EXP_DIR/variants/euclidean.yaml"
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

echo ""
echo "=============================================="
echo " Trajectory-compare GIF (same seed, same init b_0)"
echo "=============================================="
python script/trajectory_gif.py \
  --train_steps 5000 --lr 3e-4 --lr_schedule cosine --ode_steps 50 --fps 12 \
  --out_dir "$OUT_ROOT/gif"

# canonical 위치(docs/assets/)에 동기화 — README.md 임베드용
mkdir -p docs/assets
cp "$OUT_ROOT/gif/trajectory_compare.gif" docs/assets/
cp "$OUT_ROOT/gif/frame_t_0.00.png" docs/assets/
cp "$OUT_ROOT/gif/frame_t_0.50.png" docs/assets/
cp "$OUT_ROOT/gif/frame_t_1.00.png" docs/assets/
echo "[sync] outputs → docs/assets/ 갱신 완료"

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
