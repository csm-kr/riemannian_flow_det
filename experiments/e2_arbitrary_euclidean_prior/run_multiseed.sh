#!/usr/bin/env bash
# e2 multi-seed sweep — 4 variants × 3 seeds = 12 separate python processes.
# 목적: single-seed run 결과의 variance 를 정량화해서 mean±std 로 재결론.
set -euo pipefail

EXP_DIR="experiments/e2_arbitrary_euclidean_prior"
OUT_ROOT="outputs/e2_arbitrary_euclidean_prior/multiseed"
export PYTHONPATH="${PYTHONPATH:-.}:."

mkdir -p "$OUT_ROOT"

variants=(
  "riemannian|$EXP_DIR/variants/riemannian.yaml"
  "euclidean|$EXP_DIR/variants/euclidean.yaml"
  "riemannian_arb_prior|$EXP_DIR/variants/riemannian_arb_prior.yaml"
  "euclidean_arb_prior|$EXP_DIR/variants/euclidean_arb_prior.yaml"
)
seeds=(0 1 2)

for row in "${variants[@]}"; do
  IFS='|' read -r tag cfg <<< "$row"
  for seed in "${seeds[@]}"; do
    out="$OUT_ROOT/$tag/seed${seed}"
    # config 에 seed 는 0 이 박혀있으므로 env 로 override 불가.
    # 대신 yaml 을 임시로 sed 치환해서 사본을 만든다.
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

# Aggregation
echo ""
echo "=============================================="
echo " Aggregate (mean ± std over 3 seeds)"
echo "=============================================="
python "$EXP_DIR/aggregate_multiseed.py" --root "$OUT_ROOT" \
  --variants "${variants[@]%|*}"
