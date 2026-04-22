# Conventions

## 박스 포맷 규약

- **기준 포맷**: `cxcywh` normalized — `[cx, cy, w, h]` ∈ (0, 1)
- **State space** (모델 내부): `[cx, cy, log_w, log_h]` — `dataset/box_ops.py`의 `cxcywh_to_state` / `state_to_cxcywh` 사용
- 파이프라인: `xyxy pixel` (로드) → `cxcywh pixel` → `normalized cxcywh` (입력) → `log-scale state` (모델 내부)

> 모델 내부에서 `w`, `h`를 raw 값으로 직접 연산하지 않는다. 반드시 log-scale state를 사용.

---

## 코딩 규칙

**함수 docstring** — 모든 주요 함수에 필수:

```python
def forward(self, images, boxes, t):
    """
    Purpose: Predict vector field for box states at time t.
    Inputs:
        images: [B, 3, H, W], float32 — normalized
        boxes:  [B, N, 4],    float32 — normalized cxcywh
        t:      [B],          float32 — time in [0, 1]
    Outputs:
        v:      [B, N, 4],    float32 — vector field in box state space
    """
```

**모듈 테스트** — 각 파일 하단에 `if __name__ == "__main__":` 블록 포함.
최소 shape assert + sanity check. 추후 복잡해지면 `pytest` + `tests/` 디렉토리로 이동.

---

## Git 관리 규약

- 작업 단위마다 `feature/<한일>` 브랜치 생성 후 push
- 브랜치 네이밍: `feature/` 접두사 + 영문 소문자 + `-` 구분
- main 브랜치 직접 push 금지 — PR을 통해 병합

예시:

```bash
git checkout -b feature/add-voc-dataset
# ... 작업 ...
git add <파일>
git commit -m "feat: VOC dataset 래퍼 구현"
git push origin feature/add-voc-dataset
```

---

## 주의사항

- `problem_statement.md`에 정의된 box state space(`ℝ² × ℝ₊²`)와 구현이 어긋나지 않도록 주의
- Detectron2 API는 편의 도구 — 논문 기여와 혼동하지 말 것
- `outputs/`는 git 미추적. 체크포인트/로그 경로 하드코딩 금지, config에서 관리
- config 시스템 변경(ConfigArgParse → Hydra) 시 `utils/config.py`만 수정하면 되도록 추상화 유지
