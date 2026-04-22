# Issues

해결되지 않은 블로커, 디자인 dilemma, 재현 안 되는 버그를 기록. Claude는 막혔을 때만 읽는다.

> **새 이슈 발생 시**: `Open` 섹션에 템플릿을 채워 추가. 해결되면 `Resolved`로 이동하고 **해결 방식**과 **원인**을 남긴다.

---

## 템플릿

```markdown
### [제목 — 한 줄 요약]
- **Date**: YYYY-MM-DD
- **Area**: dataset / model / train / eval / infra
- **Files**: `path/to/file.py`, ...
- **Severity**: blocker / major / minor
- **Symptom**: 관찰된 현상 (error message, wrong output, flaky test 등)
- **Repro**: 재현 방법 (command / input)
- **Tried**: 시도한 해결책과 결과
- **Hypothesis**: 의심되는 원인
- **Status**: open / investigating / resolved
```

---

## 🔴 Open

_(없음)_

---

## 🟡 Investigating

_(없음)_

---

## ✅ Resolved

### FlowDiT에 per-query positional embedding 부재 → class-indexed 1-to-1 매칭 불가
- **Date**: 2026-04-22 (open) → 2026-04-22 (resolved)
- **Area**: model
- **Files**: `model/dit.py`, `model/flow_matching.py`
- **Severity**: major
- **Symptom**: MNIST Box 1장 overfit run에서 loss 진동 (~0.5, 2000 step), `max_box_err ≈ 0.6`.
- **Root cause**: FlowDiT의 box token이 `BoxEmbedding(b_t)`만 사용 → 추론 시 `b_0 ~ N(0,I)` 대칭이라 query들이 서로 구분되지 않음.
- **Fix**: `FlowDiT`에 `nn.Embedding(num_queries, dim)` 추가, `box_tokens = BoxEmbedding(b_t) + query_embed(arange(N))`. `RiemannianFlowDet`에서 `num_queries`를 FlowDiT로 전달.
- **Verification**: MB5 재실행 → loss 4.63 → 0.026, `mean_box_err` 0.17 → 0.022, `max_box_err` 0.62 → 0.11. GT vs pred 시각적으로 모든 digit 매칭.
