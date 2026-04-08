# Plan & TODO

## Current Focus: Dataset

---

## Phase 0: 설계 결정

### D0-1. 내부 박스 표현 포맷 ✅ 확정
- **기준 포맷**: `cxcywh` — `[cx, cy, w, h]` (pixel 또는 normalized)
- **이유**: 다른 모델들과의 실험 비교를 위해 가장 범용적인 포맷
- `xyxy ↔ cxcywh` 변환은 `box_ops.py`에서 제공
- state space 변환 `(cx, cy, log_w, log_h)`은 별도 함수로 제공 (모델 내부 전용)

### D0-2. COCO/VOC 로딩 방식
- **결정**: `pycocotools` 직접 사용 (Detectron2 의존 없이)

### D0-3. Augmentation 범위
- **결정**: 최소 먼저 — resize, random flip, normalize

---

## Phase 1: box_ops.py

**목표**: 모든 박스 포맷 변환의 단일 진입점.
**기준 포맷**: `cxcywh` (`[cx, cy, w, h]`)

### TODO

- [ ] **D1-1** 포맷 변환
  - `xyxy_to_cxcywh(boxes)` — `[x1,y1,x2,y2] → [cx,cy,w,h]`
  - `cxcywh_to_xyxy(boxes)` — 역방향
  - `normalize_boxes(boxes, img_w, img_h)` — pixel → [0,1] (cxcywh 기준)
  - `denormalize_boxes(boxes, img_w, img_h)` — 역방향

- [ ] **D1-2** State space 변환 (모델 내부용, `ℝ² × ℝ₊²`)
  - `cxcywh_to_state(boxes)` — `[cx, cy, w, h] → [cx, cy, log_w, log_h]`
  - `state_to_cxcywh(states)` — 역방향
  - 논문의 box state space 정의와 직접 대응

- [ ] **D1-3** 유틸
  - `clip_boxes(boxes, img_w, img_h)` — 경계 clamp
  - `box_area(boxes)` — `[N,4]` 또는 `[B,N,4]` 지원
  - `box_iou(boxes_a, boxes_b)` — IoU 계산

**입출력 규약**: 모든 함수 `torch.Tensor [N,4]` 또는 `[B,N,4]` 입력, 동일 shape 출력.

---

## Phase 2: transforms.py

**목표**: 이미지 + 박스 동시 변환 파이프라인.
**설계 원칙**: transform 내부는 `xyxy pixel`로 처리, 마지막 단계에서 normalized `cxcywh`로 변환.

### TODO

- [ ] **D2-1** `ResizeShortestEdge(min_size, max_size)`
  - 이미지 resize + 박스 pixel 좌표 동일 비율 변환

- [ ] **D2-2** `RandomHorizontalFlip(p=0.5)`
  - 이미지 flip + `x1, x2` 좌표 반전

- [ ] **D2-3** `Normalize(mean, std)` — ImageNet 기본값

- [ ] **D2-4** `Compose(transforms)` — 파이프라인 합성

---

## Phase 3: coco.py / voc.py

**목표**: `torch.utils.data.Dataset` 서브클래스, 통일된 출력 포맷.

### 공통 sample 출력
```python
{
    "image":     torch.Tensor [3, H, W],  # float32, normalized
    "boxes":     torch.Tensor [N, 4],     # normalized cxcywh
    "labels":    torch.Tensor [N],        # int64
    "image_id":  int,
    "orig_size": (H, W),
}
```

### COCO TODO
- [ ] **D3-1** `COCODetection(root, split, transforms)`
  - `split`: `"train2017"`, `"val2017"`
  - `iscrowd=0`, `area > 0` 필터
- [ ] **D3-2** category id → contiguous 80-class 매핑

### VOC TODO
- [ ] **D3-3** `VOCDetection(root, year, split, transforms)`
  - `year`: `"2007"`, `"2012"`
  - XML annotation 파싱, `difficult` 플래그 옵션

---

## Phase 4: collate.py

### TODO

- [ ] **D4-1** `detection_collate_fn(batch)`
  - `image`: `[B, 3, H, W]` stack
  - `boxes`, `labels`: list of tensors (패딩 없이 — 박스 수 가변)

- [ ] **D4-2** `build_dataloader(dataset, batch_size, num_workers, shuffle)`

---

## Phase 5: dataset/__init__.py

- [ ] **D5-1** `build_dataset(config, split)` — COCO/VOC 선택
- [ ] **D5-2** `build_dataloader(config, split)` — DataLoader 포함

---

## 완료 기준

```python
from dataset import build_dataloader
loader = build_dataloader(config, split="train")
batch = next(iter(loader))
# batch["image"]:  torch.Tensor [B, 3, H, W]
# batch["boxes"]:  list of [Ni, 4], normalized cxcywh
# batch["labels"]: list of [Ni]
```

```python
from dataset.box_ops import cxcywh_to_state, state_to_cxcywh
states = cxcywh_to_state(boxes)   # [N,4] → [cx, cy, log_w, log_h]
assert torch.allclose(boxes, state_to_cxcywh(states))
```
