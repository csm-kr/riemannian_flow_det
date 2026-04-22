"""
COCO 2017 detection dataset.

Sample output:
    image:     torch.Tensor [3, H, W], float32, ImageNet-normalized
    boxes:     torch.Tensor [N, 4],    float32, normalized cxcywh
    labels:    torch.Tensor [N],       int64, contiguous 0-79
    image_id:  int
    orig_size: (H, W)
"""

import os
import numpy as np
from PIL import Image

import torch
from pycocotools.coco import COCO

from dataset.box_ops import xyxy_to_cxcywh, normalize_boxes
from dataset.transforms import build_transforms, Compose


# COCO 80-class contiguous mapping
# Original category IDs (1-90, non-contiguous) → 0-79
COCO_ORIG_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
COCO_ID_TO_CONTIGUOUS = {orig: i for i, orig in enumerate(COCO_ORIG_IDS)}


class COCODetection(torch.utils.data.Dataset):
    """
    COCO 2017 detection dataset.

    Args:
        root:       이미지 디렉토리 (e.g. "data/coco/train2017")
        ann_file:   annotation JSON 경로 (e.g. "data/coco/annotations/instances_train2017.json")
        transforms: Compose 인스턴스. None이면 build_transforms(split) 사용.
        split:      "train" | "val"  — transforms=None일 때 split에 맞는 aug 적용
        min_size:   resize shortest edge
        max_size:   resize longest edge 상한
    """

    def __init__(
        self,
        root:       str,
        ann_file:   str,
        transforms: Compose | None = None,
        split:      str  = "train",
        min_size:   int  = 800,
        max_size:   int  = 1333,
    ):
        self.root       = root
        self.coco       = COCO(ann_file)
        self.transforms = transforms or build_transforms(split, min_size, max_size)

        # iscrowd=0 & area>0 이미지만 유지
        self.img_ids = sorted([
            img_id for img_id in self.coco.imgs
            if self._has_valid_annotations(img_id)
        ])

    def _has_valid_annotations(self, img_id: int) -> bool:
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)
        return any(
            a["area"] > 0
            and a["category_id"] in COCO_ID_TO_CONTIGUOUS
            for a in anns
        )

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Outputs:
            image:     [3, H, W] float32, ImageNet-normalized
            boxes:     [N, 4]    float32, normalized cxcywh
            labels:    [N]       int64, contiguous 0-79
            image_id:  int
            orig_size: (H, W)
        """
        img_id   = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.root, img_info["file_name"])

        img_pil  = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size

        boxes_xyxy, labels = self._load_annotations(img_id, orig_w, orig_h)

        # Apply transforms
        img_tensor, boxes_xyxy = self.transforms(img_pil, boxes_xyxy)
        _, new_h, new_w = img_tensor.shape

        # xyxy pixel → normalized cxcywh
        if len(boxes_xyxy):
            boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
            boxes_t = xyxy_to_cxcywh(boxes_t)
            boxes_t = normalize_boxes(boxes_t, img_w=new_w, img_h=new_h)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image":     img_tensor,
            "boxes":     boxes_t,
            "labels":    torch.as_tensor(labels, dtype=torch.int64),
            "image_id":  img_id,
            "orig_size": (orig_h, orig_w),
        }

    def _load_annotations(
        self, img_id: int, img_w: int, img_h: int
    ) -> tuple[np.ndarray, list[int]]:
        """
        COCO annotation → xyxy pixel boxes + contiguous label indices.

        Outputs:
            boxes:  [N, 4] float32, xyxy pixel
            labels: list[int], 0-79
        """
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in COCO_ID_TO_CONTIGUOUS:
                continue
            if ann["area"] <= 0:
                continue

            # COCO bbox: [x, y, w, h] pixel
            x, y, bw, bh = ann["bbox"]
            x1, y1 = x, y
            x2, y2 = x + bw, y + bh

            # Clamp to image bounds
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            x2 = min(float(img_w), x2)
            y2 = min(float(img_h), y2)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(COCO_ID_TO_CONTIGUOUS[cat_id])

        if boxes:
            return np.array(boxes, dtype=np.float32), labels
        return np.zeros((0, 4), dtype=np.float32), []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     default="data/coco/val2017")
    parser.add_argument("--ann",      default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--split",    default="val", choices=["train", "val"])
    args = parser.parse_args()

    print(f"Loading COCO from {args.root} ...")
    ds = COCODetection(root=args.root, ann_file=args.ann, split=args.split)
    print(f"  총 샘플 수: {len(ds)}")

    sample = ds[0]
    assert sample["image"].shape[0] == 3
    assert sample["boxes"].ndim == 2 and sample["boxes"].shape[-1] == 4
    assert sample["labels"].shape[0] == sample["boxes"].shape[0]
    assert sample["boxes"].max() <= 1.0 + 1e-4
    assert sample["boxes"].min() >= -1e-4
    print(f"  image:    {sample['image'].shape}")
    print(f"  boxes:    {sample['boxes'].shape}  (normalized cxcywh)")
    print(f"  labels:   {sample['labels'].tolist()}")
    print(f"  image_id: {sample['image_id']}")
    print(f"  orig_size:{sample['orig_size']}")
    print("  Shape checks passed.")
