# experiments/

각 실험은 **하나의 목표 + 한 개의 config 계열 + 하나의 report**로 구성되는 자기완결 단위.

## 디렉토리 규약

```
experiments/
├── README.md                         # (이 파일) — 규약 설명
└── e<번호>_<주제>/                    # 실험 ID 접두사 e0, e1, ...
    ├── report.md                     # 목표 · 설정 · 결과 · 결론
    ├── variants/                     # ablation config (≥ 1개)
    │   ├── A_<이름>.yaml
    │   └── B_<이름>.yaml
    └── run_ablation.sh                # variant 전부 실행하는 스크립트
```

## 원칙

1. **재현 가능성** — variant config와 run 스크립트만 있으면 누구나 동일 결과를 만들 수 있어야 함.
2. **결과 위치 분리** — 실행 아티팩트(ckpt/figures/metrics)는 `outputs/e<번호>_<주제>/<variant>/`에 저장. git 미추적.
3. **리포트는 요약만** — raw 로그/그림은 `outputs/…`에 두고, `report.md`에는 표/결론 중심으로 기록.
4. **winner 선정** — ablation 종료 시 report에 "winner config" 명시 + 해당 variant를 후속 실험의 기준선으로 사용.

관리 규약은 [`CLAUDE.md`](CLAUDE.md) 참조.

## 현재 실험 목록

| ID | 주제 | 상태 | report |
|----|------|------|--------|
| e0 | MB5 1-image overfit ablation (hyperparam sweep) | ✅ 완료 | [e0_mb5_overfit/report.md](e0_mb5_overfit/report.md) |
| e1 | Riemannian vs Euclidean 공정 비교 (통일 prior) | ✅ 완료 | [e1_unified_prior_fair_compare/report.md](e1_unified_prior_fair_compare/report.md) |
| e2 | 2×2 ablation: prior × interp space (arb prior 포함) | ✅ 완료 | [e2_arbitrary_euclidean_prior/report.md](e2_arbitrary_euclidean_prior/report.md) |
