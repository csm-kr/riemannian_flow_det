#!/usr/bin/env bash
# MB5 overfit ablation 실행 스크립트
# 용도: variants/*.yaml 4개를 순차 실행 → outputs/e0_mb5_overfit/<variant>/report.json
# 실행: bash experiments/e0_mb5_overfit/run_ablation.sh
set -euo pipefail

EXP_DIR="experiments/e0_mb5_overfit"
OUT_ROOT="outputs/e0_mb5_overfit"
export PYTHONPATH="${PYTHONPATH:-.}:."

# (tag, config, max_steps, lr, lr_schedule, ode_steps)
rows=(
  "A_baseline|$EXP_DIR/variants/A_baseline.yaml|1500|1e-4|const|10"
  "B_longer|$EXP_DIR/variants/B_longer.yaml|5000|3e-4|cosine|10"
  "C_ode50|$EXP_DIR/variants/C_ode50.yaml|1500|1e-4|const|50"
  "D_combined|$EXP_DIR/variants/D_combined.yaml|5000|3e-4|cosine|50"
)

for row in "${rows[@]}"; do
  IFS='|' read -r tag cfg max_steps lr sched ode <<< "$row"
  out="$OUT_ROOT/$tag"
  echo ""
  echo "=============================================="
  echo " Variant: $tag   (out=$out)"
  echo "=============================================="
  python script/overfit_mnist_box.py \
    --config "$cfg" \
    --tag "$tag" \
    --max_steps "$max_steps" \
    --lr "$lr" \
    --lr_schedule "$sched" \
    --ode_steps "$ode" \
    --log_interval 500 \
    --out_dir "$out" \
    --no-show
done

echo ""
echo "=============================================="
echo " Ablation complete. Results:"
echo "=============================================="
for row in "${rows[@]}"; do
  IFS='|' read -r tag _ _ _ _ _ <<< "$row"
  echo "--- $tag ---"
  cat "$OUT_ROOT/$tag/report.json"
  echo ""
done
