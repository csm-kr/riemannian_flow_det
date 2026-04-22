"""
MNIST Box detection dataset (synthetic toy example).

Places all 10 MNIST digits (0~9) on a 256x256 canvas at random scales [14, 56]
with no overlap. 1-to-1 class-indexed mapping: boxes[i] = digit i's box,
labels is always [0, 1, ..., 9]. No Hungarian matching needed.

Sample output:
    image:     [3, 256, 256] float32, ImageNet-normalized
    boxes:     [10, 4]       float32, normalized cxcywh ∈ (0,1), indexed by class
    labels:    [10]          int64  — always [0..9]
    image_id:  str
    orig_size: (256, 256)
"""

import os
import numpy as np
import cv2

import torch
from torchvision.datasets import MNIST

from dataset.box_ops import xyxy_to_cxcywh, normalize_boxes


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class MNISTBoxDetection(torch.utils.data.Dataset):
    """
    Synthetic detection dataset: 10 MNIST digits (0~9) on a 256x256 canvas,
    random scale [14, 56], non-overlapping, 1-to-1 class-indexed output.

    Args:
        root:        MNIST 저장 루트 (e.g. "data/mnist")
        split:       "train" | "val"
        image_size:  canvas 한 변 (default 256)
        scale_range: (min, max) digit 변 길이, 픽셀 단위 정사각형
        num_samples: __len__ (train 50000 / val 5000 기본)
        bbox_mode:   "fixed" (paste 영역) | "tight" (nonzero pixel min/max)
        background:  "zero" | "noise"
        max_tries:   배치 실패 시 재시도 횟수
        seed:        재현성 seed (val은 내부에서 +10_000_000 offset)
        download:    torchvision MNIST 자동 다운로드
    """

    def __init__(
        self,
        root:        str = "data/mnist",
        split:       str = "train",
        image_size:  int = 256,
        scale_range: tuple = (14, 56),
        num_samples: int = 50000,
        bbox_mode:   str = "fixed",
        background:  str = "zero",
        max_tries:   int = 100,
        seed:        int = 0,
        download:    bool = True,
    ):
        assert split in ("train", "val"), f"split must be train|val, got {split}"
        assert bbox_mode in ("fixed", "tight")
        assert background in ("zero", "noise")
        assert scale_range[0] < scale_range[1]

        self._mnist = MNIST(root=root, train=(split == "train"), download=download)
        targets = self._mnist.targets.numpy()
        self.by_class = {d: np.where(targets == d)[0] for d in range(10)}

        self.image_size  = image_size
        self.scale_range = (int(scale_range[0]), int(scale_range[1]))
        self.num_samples = num_samples
        self.bbox_mode   = bbox_mode
        self.background  = background
        self.max_tries   = max_tries
        self.seed        = seed + (0 if split == "train" else 10_000_000)

        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Purpose: 한 샘플 합성 (canvas + 10 digits + class-indexed boxes).
        """
        rng = np.random.RandomState(self.seed + idx)
        H = W = self.image_size

        canvas = self._make_canvas(rng)
        boxes_xyxy = np.zeros((10, 4), dtype=np.float32)

        # 큰 scale부터 배치하면 non-overlap 성공률 ↑
        scales = rng.randint(self.scale_range[0], self.scale_range[1] + 1, size=10)
        order  = np.argsort(-scales)  # descending

        placed = []  # list of (x1,y1,x2,y2)
        for i in order:
            digit = int(i)
            s     = int(scales[i])
            box   = self._place_digit(canvas, digit, s, placed, rng)
            boxes_xyxy[digit] = box
            placed.append(tuple(box.tolist()))

        # canvas: uint8 [H,W] → float32 [3,H,W], ImageNet-normalized
        img = np.stack([canvas, canvas, canvas], axis=0).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img)
        img_t = (img_t - self._mean) / self._std

        # xyxy pixel → normalized cxcywh
        boxes_t = torch.from_numpy(boxes_xyxy)
        boxes_t = xyxy_to_cxcywh(boxes_t)
        boxes_t = normalize_boxes(boxes_t, img_w=W, img_h=H)

        labels_t = torch.arange(10, dtype=torch.int64)

        return {
            "image":     img_t,
            "boxes":     boxes_t,
            "labels":    labels_t,
            "image_id":  f"mnist_box_{idx}",
            "orig_size": (H, W),
        }

    # ────────────────────────────────────────────────
    # 내부 헬퍼
    # ────────────────────────────────────────────────

    def _make_canvas(self, rng: np.random.RandomState) -> np.ndarray:
        H = W = self.image_size
        if self.background == "zero":
            return np.zeros((H, W), dtype=np.uint8)
        return rng.randint(0, 50, size=(H, W)).astype(np.uint8)

    def _place_digit(
        self,
        canvas:  np.ndarray,
        digit:   int,
        s:       int,
        placed:  list,
        rng:     np.random.RandomState,
    ) -> np.ndarray:
        """
        digit을 scale s로 canvas에 non-overlap 배치. 실패 시 scale 축소 → scan 폴백.
        Returns: [4] xyxy pixel float32
        """
        src_idx = int(rng.choice(self.by_class[digit]))
        src_pil, _ = self._mnist[src_idx]
        src = np.array(src_pil, dtype=np.uint8)  # [28,28]

        s_curr = s
        coords = self._rejection_sample(s_curr, placed, rng)
        if coords is None:
            # fallback: minimum scale로 축소해서 다시 시도
            s_curr = self.scale_range[0]
            coords = self._rejection_sample(s_curr, placed, rng)
            if coords is None:
                coords = self._scan_place(s_curr, placed)

        x1, y1, x2, y2 = coords
        scaled = cv2.resize(src, (s_curr, s_curr), interpolation=cv2.INTER_LINEAR)
        canvas[y1:y2, x1:x2] = np.maximum(canvas[y1:y2, x1:x2], scaled)

        if self.bbox_mode == "tight":
            nz = np.argwhere(scaled > 0)
            if len(nz) > 0:
                ymin, xmin = nz.min(0)
                ymax, xmax = nz.max(0) + 1
                return np.array(
                    [x1 + xmin, y1 + ymin, x1 + xmax, y1 + ymax],
                    dtype=np.float32,
                )
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _rejection_sample(
        self, s: int, placed: list, rng: np.random.RandomState,
    ):
        H = W = self.image_size
        for _ in range(self.max_tries):
            x1 = int(rng.randint(0, W - s + 1))
            y1 = int(rng.randint(0, H - s + 1))
            x2, y2 = x1 + s, y1 + s
            if not self._overlaps_any((x1, y1, x2, y2), placed):
                return (x1, y1, x2, y2)
        return None

    def _scan_place(self, s: int, placed: list):
        H = W = self.image_size
        step = max(1, s // 4)
        for y1 in range(0, H - s + 1, step):
            for x1 in range(0, W - s + 1, step):
                box = (x1, y1, x1 + s, y1 + s)
                if not self._overlaps_any(box, placed):
                    return box
        raise RuntimeError(
            f"cannot place digit at scale={s}: canvas saturated. "
            f"Consider smaller scale_range or larger image_size."
        )

    @staticmethod
    def _overlaps_any(box, others) -> bool:
        x1, y1, x2, y2 = box
        for ox1, oy1, ox2, oy2 in others:
            ix1 = max(x1, ox1); iy1 = max(y1, oy1)
            ix2 = min(x2, ox2); iy2 = min(y2, oy2)
            if ix2 > ix1 and iy2 > iy1:
                return True
        return False


# ────────────────────────────────────────────────
# cv2 시각화 유틸 (test & debug용)
# ────────────────────────────────────────────────

_COLORS_BGR = [
    ( 66, 135, 245), (245, 158,  66), ( 66, 245, 143), (245,  66, 203),
    ( 66, 245, 245), (203,  66, 245), (245,  66,  66), (143, 245,  66),
    (245, 245,  66), (100, 100, 245),
]


def denormalize_image(img_t: torch.Tensor) -> np.ndarray:
    """
    ImageNet-normalized [3,H,W] float → uint8 [H,W,3] BGR (cv2 convention).
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(3, 1, 1)
    img = img_t.detach().cpu() * std + mean
    img = (img.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def draw_sample(
    sample:     dict,
    image_size: int = 256,
    thickness:  int = 1,
) -> np.ndarray:
    """
    sample dict → bgr uint8 image with class-colored bboxes + digit labels.
    """
    img_bgr = denormalize_image(sample["image"])
    boxes   = sample["boxes"].clone() * image_size
    labels  = sample["labels"].tolist()

    x1 = (boxes[:, 0] - boxes[:, 2] / 2).round().to(torch.int32).tolist()
    y1 = (boxes[:, 1] - boxes[:, 3] / 2).round().to(torch.int32).tolist()
    x2 = (boxes[:, 0] + boxes[:, 2] / 2).round().to(torch.int32).tolist()
    y2 = (boxes[:, 1] + boxes[:, 3] / 2).round().to(torch.int32).tolist()

    for i, lbl in enumerate(labels):
        color = _COLORS_BGR[lbl % len(_COLORS_BGR)]
        cv2.rectangle(img_bgr, (x1[i], y1[i]), (x2[i], y2[i]), color, thickness)
        cv2.putText(
            img_bgr, str(lbl), (x1[i] + 1, max(y1[i] - 2, 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )
    return img_bgr


def make_grid(images: list, n_cols: int, gap: int = 4,
              gap_color: tuple = (40, 40, 40)) -> np.ndarray:
    """
    tile 간 구분 gap을 넣은 grid 합성.
    gap: 타일 사이 경계 픽셀 폭 (0이면 기존처럼 딱 붙임)
    """
    if not images:
        raise ValueError("no images")
    H, W, C = images[0].shape
    if gap <= 0:
        rows = []
        for r in range(0, len(images), n_cols):
            row = images[r:r + n_cols]
            if len(row) < n_cols:
                row += [np.zeros_like(row[0])] * (n_cols - len(row))
            rows.append(np.concatenate(row, axis=1))
        return np.concatenate(rows, axis=0)

    n_rows  = (len(images) + n_cols - 1) // n_cols
    out_h   = H * n_rows + gap * (n_rows + 1)
    out_w   = W * n_cols + gap * (n_cols + 1)
    canvas  = np.full((out_h, out_w, C), gap_color, dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, n_cols)
        y0 = gap + r * (H + gap)
        x0 = gap + c * (W + gap)
        canvas[y0:y0 + H, x0:x0 + W] = img
    return canvas


# ────────────────────────────────────────────────
# __main__ — unit check + cv2 시각화
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root",        default="data/mnist")
    parser.add_argument("--split",       default="train", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--bbox_mode",   default="fixed", choices=["fixed", "tight"])
    parser.add_argument("--background",  default="zero",  choices=["zero", "noise"])
    parser.add_argument("--n_grid",      type=int, default=4,
                        help="sqrt of sample count drawn in grid")
    parser.add_argument("--save",        default="outputs/figures/mnist_box_sample.png")
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--show",        action=argparse.BooleanOptionalAction, default=True,
                        help="cv2.imshow로 샘플 대화형 확인 (기본 ON). 끄려면 --no-show. ESC=종료, 그 외 key=다음")
    parser.add_argument("--show_n",      type=int, default=10,
                        help="--show 시 순회할 샘플 수 (기본 10)")
    args = parser.parse_args()

    # ── 1. dataset 정의 ────────────────────────────────────
    print(f"[build] MNISTBoxDetection split={args.split} num={args.num_samples}")
    ds = MNISTBoxDetection(
        root=args.root, split=args.split, num_samples=args.num_samples,
        bbox_mode=args.bbox_mode, background=args.background, seed=args.seed,
        download=True,
    )
    assert len(ds) == args.num_samples

    s = ds[0]
    print(f"  image:    {tuple(s['image'].shape)}  dtype={s['image'].dtype}")
    print(f"  boxes:    {tuple(s['boxes'].shape)}  (normalized cxcywh)")
    print(f"  labels:   {s['labels'].tolist()}")
    print(f"  image_id: {s['image_id']}")

    # ── 2. cv2.imshow로 샘플 순회 (기본 10개) ──────────────
    if args.show:
        print(f"\n[show] 샘플 {args.show_n}개 순회. ESC=종료, 그 외 key=다음")
        win = "MNIST Box"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        for i in range(min(args.show_n, len(ds))):
            vis = draw_sample(ds[i])
            cv2.imshow(win, vis)
            print(f"  sample {i}: image_id={ds[i]['image_id']}")
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
        cv2.destroyAllWindows()

 