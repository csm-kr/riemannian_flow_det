"""
VOC 2007/2012 detection dataset.

Sample output:
    image:     torch.Tensor [3, H, W], float32, ImageNet-normalized
    boxes:     torch.Tensor [N, 4],    float32, normalized cxcywh
    labels:    torch.Tensor [N],       int64
    image_id:  str
    orig_size: (H, W)
"""

import os
import numpy as np
from PIL import Image

import torch
from torchvision.datasets import VOCDetection as _TorchVOC

from dataset.box_ops import xyxy_to_cxcywh, normalize_boxes
from dataset.transforms import build_transforms, Compose


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}


class VOCDetection(torch.utils.data.Dataset):
    """
    VOC 2007/2012 detection dataset.

    Args:
        root:           데이터 루트 (e.g. "data/voc")
        year:           "2007" | "2012"
        split:          "train" | "val" | "trainval" | "test"
        transforms:     Compose 인스턴스. None이면 build_transforms(split) 사용.
        download:       True면 자동 다운로드
        skip_difficult: difficult=1 객체 제외
        min_size:       resize shortest edge (transforms=None일 때만 사용)
        max_size:       resize longest edge 상한
    """

    def __init__(
        self,
        root:           str,
        year:           str  = "2007",
        split:          str  = "train",
        transforms:     Compose | None = None,
        download:       bool = False,
        skip_difficult: bool = True,
        min_size:       int  = 800,
        max_size:       int  = 1333,
    ):
        assert year in ("2007", "2012"), f"year must be 2007 or 2012, got {year}"
        self._voc = _TorchVOC(
            root=root, year=year, image_set=split,
            download=download, transform=None, target_transform=None,
        )
        self.split          = split
        self.skip_difficult = skip_difficult
        self.transforms     = transforms or build_transforms(split, min_size, max_size)

    def __len__(self) -> int:
        return len(self._voc)

    def __getitem__(self, idx: int) -> dict:
        """
        Outputs:
            image:     [3, H, W] float32, ImageNet-normalized
            boxes:     [N, 4]    float32, normalized cxcywh
            labels:    [N]       int64
            image_id:  str
            orig_size: (H, W)
        """
        img_pil, target = self._voc[idx]
        orig_w, orig_h  = img_pil.size   # PIL: (W, H)

        boxes_xyxy, labels = self._parse_annotation(target)

        # Apply transforms: (PIL, np[N,4] xyxy) → (Tensor[3,H,W], np[N,4] xyxy)
        img_tensor, boxes_xyxy = self.transforms(img_pil, boxes_xyxy)

        # img_tensor: [3, H', W'] after ToTensor+Normalize
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
            "image_id":  str(idx),
            "orig_size": (orig_h, orig_w),
        }

    def _parse_annotation(self, target: dict) -> tuple[np.ndarray, list[int]]:
        """
        VOC XML annotation dict → xyxy pixel boxes + label indices.

        Outputs:
            boxes:  [N, 4] float32, xyxy pixel
            labels: list[int]
        """
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        boxes, labels = [], []
        for obj in objects:
            if self.skip_difficult and int(obj.get("difficult", 0)) == 1:
                continue
            cls_name = obj["name"]
            if cls_name not in VOC_CLASS_TO_IDX:
                continue
            bb = obj["bndbox"]
            x1 = float(bb["xmin"]) - 1   # VOC: 1-indexed → 0-indexed
            y1 = float(bb["ymin"]) - 1
            x2 = float(bb["xmax"]) - 1
            y2 = float(bb["ymax"]) - 1
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(VOC_CLASS_TO_IDX[cls_name])

        if boxes:
            return np.array(boxes, dtype=np.float32), labels
        return np.zeros((0, 4), dtype=np.float32), []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     default="data/voc")
    parser.add_argument("--year",     default="2007", choices=["2007", "2012"])
    parser.add_argument("--split",    default="val",  choices=["train", "val", "trainval", "test"])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    print(f"Loading VOC {args.year} {args.split} from {args.root} ...")
    ds = VOCDetection(root=args.root, year=args.year, split=args.split, download=args.download)
    print(f"  총 샘플 수: {len(ds)}")

    sample = ds[0]
    assert sample["image"].shape[0] == 3
    assert sample["boxes"].ndim == 2 and sample["boxes"].shape[-1] == 4
    assert sample["labels"].shape[0] == sample["boxes"].shape[0]
    assert sample["boxes"].max() <= 1.0 + 1e-4
    assert sample["boxes"].min() >= -1e-4
    print(f"  image:    {sample['image'].shape}")
    print(f"  boxes:    {sample['boxes'].shape}  (normalized cxcywh)")
    print(f"  labels:   {[VOC_CLASSES[l] for l in sample['labels'].tolist()]}")
    print(f"  orig_size:{sample['orig_size']}")
    print("  Shape checks passed.")
