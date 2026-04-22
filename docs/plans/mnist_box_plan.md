# Plan: MNIST Box Dataset (`MNISTBoxDetection`)

**목적**: flow matching 박스 궤적을 제어된 환경에서 검증 & 시각화하기 위한 **합성 detection dataset**.
하나의 이미지 안에 **0~9 전체 10개 숫자**를 **14~56px scale 랜덤**으로 **겹치지 않게** 배치 → 모델이 noise 박스에서 GT 10개 박스로 흘러가는 궤적(`b₀→b₁`)을 학습/시각화.

---

## 1. 설계 요약

| 항목 | 값 |
|------|-----|
| Canvas | **256 × 256** (3ch, grayscale 복제) |
| 객체 수 `N` | **고정 10개** — 클래스 0~9 각각 1개씩 |
| 숫자 scale `s` | **uniform ∈ [14, 56]** (픽셀 단위 정사각형) |
| 배치 | **non-overlapping** (rejection sampling) |
| **Query–GT 매칭** | **1-to-1 (class-indexed)** — `boxes[i]` = digit `i`의 박스, `labels = [0..9]` 고정 |
| 회전/왜곡 | 없음 |
| 배경 | 0 (검정) 기본, `background="noise"` 옵션 |
| bbox 모드 | **`fixed`** (paste 영역 = s×s) 기본, `tight` 옵션 |

### 왜 이 스펙인가

- **`N=10` 고정 + 1-to-1 매칭**: Detection에서 "각 class는 항상 정답이 정확히 1개 존재"하는 toy setup. 모델의 `query_i`는 항상 digit `i`의 박스를 예측하도록 학습 → **Hungarian matching 불필요**, loss는 단순 position-wise MSE.
- **scale [14, 56] 랜덤**: `(log_w, log_h)` 축에도 **비자명한 변화**를 주어야 flow가 진짜 4D trajectory를 학습. 고정 크기면 사실상 `(cx, cy)`만 변해 검증 의미 약함.
- **non-overlap**: GT 박스가 겹치면 class-id 혼동이 생기므로 toy 단계에서 배제.
- **256×256**: 최대 digit(56) × 10개 최악 커버리지 = `31,360 / 65,536 ≈ 48%` → non-overlap 가능 여유 충분.

### 1-to-1 매칭 전제 (중요)

```python
# 항상 성립:
boxes[i]  = digit i의 GT 박스   (i = 0..9)
labels[i] = i                    # 즉, labels == torch.arange(10)
```

- 모델: `num_queries = 10`, query 순서 = class 순서
- Loss: `MSE(pred_boxes, gt_boxes)` — 각 query_i가 box_i 직접 예측
- 추후 multi-instance/scene 확장 시에만 Hungarian matching 도입

---

## 2. Dataset 사양

### 2.1 Output 규약 (VOCDetection과 동일)

```python
{
    "image":     torch.Tensor [3, 256, 256] float32, ImageNet-normalized,
    "boxes":     torch.Tensor [10, 4]       float32, normalized cxcywh ∈ (0,1),
                                            #  boxes[i] = digit i의 박스 (class-indexed)
    "labels":    torch.Tensor [10]          int64 — 항상 [0, 1, ..., 9] 고정,
    "image_id":  str,
    "orig_size": (256, 256),
}
```

`collate.py`의 `detection_collate_fn` 그대로 사용. 모델 파이프라인 무수정.

### 2.2 클래스 인덱스

- `labels[i] == i` (항상). Background class 없음 (항상 10개 존재).

### 2.3 재현성

- `seed + idx`로 `np.random.RandomState` 생성 → 같은 idx는 항상 같은 샘플.
- Train/val 간 분리: val은 `seed + 10_000_000 + idx`로 충돌 방지.

---

## 3. 생성 알고리즘

```
for idx in __getitem__:
    rng = RandomState(seed + idx)
    canvas = zeros(256, 256)
    placed_bboxes = []
    for digit in shuffled([0,...,9]):                    # 클래스 순서 랜덤
        src_img = mnist_image_of_class(digit, rng)        # MNIST에서 해당 digit 하나 샘플
        for attempt in range(MAX_TRIES=100):
            s  = rng.uniform(14, 56)                      # scale
            s  = int(round(s))
            cx = rng.randint(s//2, 256 - s//2)
            cy = rng.randint(s//2, 256 - s//2)
            box = (cx-s/2, cy-s/2, cx+s/2, cy+s/2)
            if no_overlap(box, placed_bboxes):
                paste(canvas, resize(src_img, s, s), cx, cy)
                placed_bboxes.append(box)
                break
        else:
            # 100회 실패 시: scale을 14로 고정하고 여유 자리 탐색
            fallback_place(...)  # 반드시 10개 채움
```

### 3.1 MNIST 인덱싱

- `__init__`에서 MNIST 전체 로드 후 **class별 index list** 미리 구축:
  ```python
  self.by_class: dict[int, np.ndarray]  # {0: [indices...], ..., 9: [...]}
  ```
  → `by_class[digit]`에서 random choice로 O(1) 샘플.

### 3.2 bbox 값

- **`fixed`** (기본): bbox = paste 영역 그대로 `[cx, cy, s, s]` → normalized로 나눠 `/256`.
- **`tight`** (옵션): resize된 MNIST patch의 nonzero pixel min/max로 축소. 약간 더 현실적이지만 scale 해석이 살짝 작아짐.

### 3.3 overlap 판정

- IoU > 0 이면 overlap (즉 면적 교집합 > 0). 단순 `xyxy` 교집합으로 판정, boxes가 작아 O(N²=100) OK.

---

## 4. API

### 4.1 `dataset/mnist_box.py`

```python
class MNISTBoxDetection(torch.utils.data.Dataset):
    """
    Synthetic detection dataset: 10 MNIST digits (0~9) placed on 256x256 canvas
    with random scale [14,56] and no overlap.
    """
    def __init__(
        self,
        root:        str = "data/mnist",
        split:       str = "train",          # train | val
        image_size:  int = 256,
        scale_range: tuple[int,int] = (14, 56),
        num_samples: int = 50000,
        bbox_mode:   str = "fixed",          # fixed | tight
        background:  str = "zero",           # zero | noise
        max_tries:   int = 100,
        seed:        int = 0,
        download:    bool = True,
    ):
        ...

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...
```

내부에 얇은 transforms 경로 (`ToTensor + Normalize`)만 보유 — canvas 크기가 고정이라 resize 불필요.

### 4.2 `dataset/__init__.py` 수정

`build_dataset`에 분기 추가:

```python
elif name == "mnist_box":
    return MNISTBoxDetection(
        root        = _get("mnist_root",       f"{root}/mnist"),
        split       = split,
        image_size  = _get("mnist_image_size", 256),
        scale_range = tuple(_get("mnist_scale_range", (14, 56))),
        num_samples = _get(
            "mnist_num_samples_train" if split == "train"
            else "mnist_num_samples_val",
            50000 if split == "train" else 5000,
        ),
        bbox_mode   = _get("mnist_bbox_mode", "fixed"),
        background  = _get("mnist_background", "zero"),
        seed        = _get("seed", 0),
    )
```

### 4.3 `configs/mnist_box.yaml` (신규)

```yaml
dataset: mnist_box
mnist_root: data/mnist
mnist_image_size: 256
mnist_scale_range: [14, 56]
mnist_num_samples_train: 50000
mnist_num_samples_val: 5000
mnist_bbox_mode: fixed
mnist_background: zero

# 모델 설정
hidden_dim: 192
num_layers: 4
num_heads: 6
num_queries: 10           # 정확히 10개 객체이므로 딱 맞춤
backbone_type: fpn
batch_size: 32

# transforms 생략 (dataset 내부 처리)
min_size: 256
max_size: 256
```

> `num_queries: 10` — 항상 10개 객체 고정이므로 query 수를 일치시켜 단순화. 나중에 여유 query 실험은 ablation.

---

## 5. 단일-이미지 overfitting 모드 (디버깅용)

궤적 학습 검증은 **"한 장에 overfit되는가"** 가 첫 sanity check:

```yaml
# 예: 한 장만 반복 학습
mnist_num_samples_train: 1
mnist_num_samples_val: 1
```

`seed + idx=0`이라 고정 이미지 1장. loss가 0 근처로 수렴 + ODE 궤적이 정확히 GT 박스에 도달하면 → 모델/loss/trajectory 구현 정상.

별도 `overfit_idx` 파라미터를 만들지 말고 **`num_samples=1`** 이라는 기존 파라미터로 커버 (최소 surface area 원칙).

---

## 6. 검증 플랜

### 6.1 `__main__` 블록 (unit)

```python
ds = MNISTBoxDetection(split="train", num_samples=32, download=True)
assert len(ds) == 32
s = ds[0]
assert s["image"].shape == (3, 256, 256)
assert s["boxes"].shape == (10, 4)
assert s["labels"].shape == (10,) and set(s["labels"].tolist()) == set(range(10))
assert 0.0 <= s["boxes"].min() and s["boxes"].max() <= 1.0

# non-overlap 검증
xyxy = cxcywh_to_xyxy(s["boxes"]) * 256
for i in range(10):
    for j in range(i+1, 10):
        assert box_iou_single(xyxy[i], xyxy[j]) == 0.0, "boxes overlap"

# scale range 검증
wh = s["boxes"][:, 2:] * 256
assert (wh >= 14).all() and (wh <= 56).all()

# determinism
s0 = MNISTBoxDetection(split="train", num_samples=32, seed=0)[5]["image"]
s1 = MNISTBoxDetection(split="train", num_samples=32, seed=0)[5]["image"]
assert torch.allclose(s0, s1)
```

### 6.2 시각화 (cv2)

`__main__` 블록에서 **`cv2`** 로 n×n 그리드 샘플에 bbox + class label 그려 `outputs/figures/mnist_box_sample.png` 저장.
cv2 선택 이유: matplotlib 의존 제거 + headless(Docker) 환경에서 바로 파일 저장 가능.

```bash
python dataset/mnist_box.py --n_grid 4 --save outputs/figures/mnist_box_sample.png
```

### 6.3 smoke training

```bash
# 1장 overfit test
python script/train.py --config configs/mnist_box.yaml \
    --mnist_num_samples_train 1 --max_steps 2000
# 기대: loss → ~0, ODE(10step) 궤적 → GT 10 boxes
```

---

## 7. 구현 순서 (TODO 항목 = MB*)

1. **MB1** `dataset/mnist_box.py`
   - 클래스 구현, scale 랜덤 + non-overlap rejection, class별 index map
   - `__main__` 블록: shape/class/non-overlap/scale/determinism 검증 + 샘플 그림 저장
2. **MB2** `dataset/__init__.py` — `build_dataset`에 `mnist_box` 분기
3. **MB3** smoke test: `build_dataloader`로 batch 1개 꺼내 shape 확인 (기존 collate 호환 검증)
4. **MB4** `configs/mnist_box.yaml` — *Phase 3 S2/S5 이후*
5. **MB5** single-image overfit run — *Phase 3 S1 (train 스크립트) 이후*

> MB1~MB3는 **지금 바로 착수 가능**.
> MB4, MB5는 Phase 3에 의존.

---

## 8. Open Questions

- **Q1**: 14 × 10 = 140 < 256 이라 최악 케이스라도 non-overlap placement 가능하나, 실제로는 첫 몇 개가 큰 scale로 자리를 차지하면 남은 숫자가 채울 자리 부족해질 수 있음.
  - 제안: (a) 큰 scale부터 배치 (내림차순), (b) `max_tries=100` 실패 시 해당 digit만 scale 14로 강제.
  - 이것으로 실패율 ~0%인지 첫 구현 후 측정 필요.
- **Q2**: MNIST는 28×28. 14로 줄이는 건 괜찮은데 **56으로 확대**(2배 upscale)하면 blocky해짐 → bilinear 리사이즈 허용할지?
  - 제안: **bilinear** 기본, 가장 자연스럽고 모델 입력은 어차피 normalize됨.
- **Q3**: bbox `fixed` 기본이 맞는지? `tight`가 더 현실적이지만 scale 해석이 불투명해짐.
  - 제안: **`fixed`** 기본. scale ↔ bbox 크기가 1:1이라 실험 해석 명확.
- **Q4**: background를 `"zero"`로만 두면 FPN/DINOv2가 trivial하게 풀 수 있음 → ablation용으로 `"noise"` 옵션 유지. 기본은 `"zero"`.

---

## 9. 범위 밖

- 회전/affine/scale jitter 외 augmentation
- 클래스 반복 (같은 digit 2번 등장) — 항상 1개씩 고정
- 시퀀스 형태 (src→tgt 두 장) — 별도 플랜
- 3D/RGB MNIST 변종
