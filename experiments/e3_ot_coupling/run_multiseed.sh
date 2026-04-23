#!/usr/bin/env bash
# e3 — OT-coupling verification.
# e2 의 4 variant 에 ot_coupling: true 만 켜고 3 seed × 4 variant = 12 run.
# 목적: arb_prior 의 13× variance 가 baseline 수준으로 회복되는지 검증.
set -euo pipefail

EXP_DIR="experiments/e3_ot_coupling"
OUT_ROOT="outputs/e3_ot_coupling/multiseed"
export PYTHONPATH="${PYTHONPATH:-.}:."

mkdir -p "$OUT_ROOT"

variants=(
  "riemannian_ot|$EXP_DIR/variants/riemannian_ot.yaml"
  "euclidean_ot|$EXP_DIR/variants/euclidean_ot.yaml"
  "riemannian_arb_prior_ot|$EXP_DIR/variants/riemannian_arb_prior_ot.yaml"
  "euclidean_arb_prior_ot|$EXP_DIR/variants/euclidean_arb_prior_ot.yaml"
)
seeds=(0 1 2)

for row in "${variants[@]}"; do
  IFS='|' read -r tag cfg <<< "$row"
  for seed in "${seeds[@]}"; do
    out="$OUT_ROOT/$tag/seed${seed}"
    tmp_cfg="$OUT_ROOT/${tag}_seed${seed}.yaml"
    sed "s/^seed:.*/seed: ${seed}/" "$cfg" > "$tmp_cfg"

    echo ""
    echo "=============================================="
    echo " ${tag}  seed=${seed}"
    echo "=============================================="
    python script/overfit_mnist_box.py \
      --config "$tmp_cfg" \
      --tag "${tag}_seed${seed}" \
      --max_steps 5000 --lr 3e-4 --lr_schedule cosine --ode_steps 50 \
      --log_interval 5000 \
      --out_dir "$out" \
      --no-show 2>&1 | tail -5
  done
done

# Aggregate
echo ""
echo "=============================================="
echo " Aggregate (mean ± std over 3 seeds)"
echo "=============================================="
python experiments/e2_arbitrary_euclidean_prior/aggregate_multiseed.py \
  --root "$OUT_ROOT" \
  --variants "${variants[@]%|*}"
