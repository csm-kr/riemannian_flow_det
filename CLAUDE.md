# CLAUDE.md

Claude Code가 항상 적용하는 **최소 규칙** + 상세 문서 **포인터**.
상세 내용은 해당 작업을 할 때만 읽는다.

---

## 문서 구조 (역할 분리)

| 파일 | 역할 | 업데이트 시점 |
|------|------|--------------|
| `docs/ROADMAP.md` | **큰 설계** — Phase 단위 목적·이유·방향 | Phase 추가/재배치 등 구조 변경 시 |
| `docs/plans/<주제>_plan.md` | **세부 설계** — 개별 작업·실험의 구체 설계 (하위 폴더 없이 평면 구조) | 새 작업 착수 전 작성 |
| `docs/TODO.md` | **체크리스트** — ROADMAP Phase 구조를 미러링한 실행 트래커 | task 진행/완료 시마다 |
| `docs/ISSUES.md` | **이슈 로그** — 막혔을 때만 기록 | 블로커 발생 / 해결 시 |

---

## Plans & TODO (항상 적용)

1. 새 작업/실험/리팩토링 → `docs/plans/<주제>_plan.md`에 세부 설계 작성 (**하위 폴더 만들지 않음, 파일명은 `*_plan.md` 접미사로 통일**, 예: `docs/plans/mnist_box_plan.md`)
2. 해당 작업을 `docs/TODO.md`에 추가 — **ROADMAP.md의 Phase 구조를 그대로 따라가며 Phase 단위 + 하위 task 모두 표시**
3. 상태 표기는 **Unicode 이모지**로 통일 (GFM task list 대신 — 모든 미리보기 호환): ✅ 완료 · 🔄 진행 중 · ⬜ 예정. Phase 하위 task가 전부 ✅이면 Phase도 ✅
4. **ROADMAP과 TODO의 Phase 목록/순서는 항상 동기화** — Phase 추가·변경·재배치 시 두 파일 함께 업데이트
5. 막혔을 때만 `docs/ISSUES.md`에 템플릿 채워 기록
6. 모든 문서성 `.md`는 `docs/` 안에 둔다 (예외: `CLAUDE.md`, `README.md`, `LICENSE.md`, `model/CLAUDE.md`는 위치 고정)

---

## Reference (필요할 때만 읽기)

| 주제 | 파일 |
|------|------|
| 현재 작업 / 체크리스트 | `docs/TODO.md` |
| 전체 로드맵 (Phase 0~4) | `docs/ROADMAP.md` |
| 개별 작업 세부 설계 | `docs/plans/<주제>_plan.md` |
| 미해결 이슈 | `docs/ISSUES.md` |
| 프로젝트 개요 · 기술 스택 · 구조 · Docker | `docs/overview.md` |
| 코딩 규칙 · Git · 박스 포맷 · 주의사항 | `docs/conventions.md` |
| 모델 내부 설계 (DiT / flow / trajectory) | `docs/problem_statement.md`, `docs/model_plan.md`, `model/CLAUDE.md` |
| 실행 명령 전체 | `README.md` |

---

## Trigger Keywords

| 키워드 | 동작 |
|--------|------|
| `/test [파일]` | 해당 파일의 `__main__` 블록 실행으로 검증 |
| `/ship [파일]` | 테스트 통과 → `git add` → `git commit` → `git push`. 실패 시 중단 |
| `/done [항목ID]` | `docs/TODO.md`에서 해당 항목 ✅ 처리 후 다음 항목 출력 |
