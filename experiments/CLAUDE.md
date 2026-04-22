# experiments/ — CLAUDE.md

실험(ablation·비교·sweep)은 **모두 이 폴더에서 관리**한다.
모델·데이터 코드 변경이 동반된 "의미 있는" 결과를 냈다면 반드시 여기에 실험 단위(e<번호>)로 기록.

---

## 기본 원칙

1. **자기완결성**: 한 실험 폴더만 보고도 "무엇을 했는지 / 어떻게 재현하는지 / 무엇이 결론인지"가 나온다.
2. **재현 가능성**: config(variant yaml) + 실행 스크립트만으로 결과 복원 가능.
3. **결과물 위치 분리**: run 아티팩트(ckpt/figures/metrics/GIF)는 `outputs/<동일 이름>/`에 둔다. git 미추적.
4. **실험 ID**: `e0, e1, e2, ...` 순차. 접두사 `e`(experiment) 고정. 넘버는 **시간 순서** (의미 순서 아님).
5. **Winner 선정**: variant 하나 이상을 표로 비교하고 결론 섹션에 winner 명시.

## 폴더 구조

```
experiments/
├── README.md                          # 실험 목록 (가이드)
├── CLAUDE.md                          # (이 파일)
└── e<번호>_<주제>/
    ├── report.md                      # 필수 — 아래 8개 섹션 포함
    ├── variants/                      # 필수 — 최소 1개 config
    │   └── <tag>.yaml
    └── run.sh                         # 권장 — variant 전부 실행
```

`report.md` 필수 섹션 (생략해도 되는 것은 "—" 로 명시):

| # | 제목 | 내용 |
|---|------|------|
| 1 | 목표 | 한 문장 가설/질문 |
| 2 | 설계 | 방법론, 왜 이 구성인지 |
| 3 | 공통 설정 | 변동되지 않는 축 (dataset/model/train 세팅) |
| 4 | Variants | 변경되는 축 + 표 |
| 5 | 결과 | 지표 표 + 1~2 sentence summary |
| 6 | 관찰 | 숫자만으로 읽히지 않는 해석 |
| 7 | 결론 | winner config 또는 main finding |
| 8 | 다음 단계 | 이 실험에서 발견된 follow-up |

## 실험을 "언제" 만드나

- **신규 metric / 신규 dataset / 신규 모델 변경**을 검증할 때
- **비교 가능한 variant가 2개 이상** 존재할 때 (단일 run은 실험 아님)
- 발견이 **논문/블로그 수준의 claim을 지지**할 만할 때

→ 단순한 run (학습 재시작, 파라미터 확인용)은 실험이 아님. `outputs/` 에만 남긴다.

## 실험 폴더 쓰는 순서

1. `mkdir experiments/e<번호>_<주제>/{variants,}` + `touch report.md run.sh`
2. `report.md`의 1·2·3·4 먼저 쓴다 (가설 먼저)
3. `variants/*.yaml` 정의 + `run.sh` 작성
4. 실행 → 결과 수집 (`outputs/…/report.json`)
5. `report.md` 5·6·7·8 채운다
6. `experiments/README.md`의 목록에 한 줄 추가
7. `docs/TODO.md` "Done"에 체크박스 + 링크

## 기존 실험 참조

- [e0_mb5_overfit](e0_mb5_overfit/report.md) — MB5 1-image overfit 4-variant ablation
- [e1_unified_prior_fair_compare](e1_unified_prior_fair_compare/report.md) — Riemannian vs Euclidean 공정 비교 (통일 prior)
