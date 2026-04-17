# File: dataset/voc.py
# Role: VOC 2007/2012 detection dataset 래퍼
# Pipeline: DataLoader → collate → model 입력
# 의존: torchvision.datasets.VOCDetection (다운로드 + XML 파싱)
#       커스텀 resize/flip augmentation (detectron2 불필요)
# Outputs per sample:
#   image:     torch.Tensor [3, H, W], float32, normalized
#   boxes:     torch.Tensor [N, 4],    float32, normalized cxcywh
#   labels:    torch.Tensor [N],       int64
#   image_id:  str  (파일명)
#   orig_size: tuple (H, W)

import os
import random
import torch
import numpy as np
from PIL import Image
from torchvision.datasets import VOCDetection as _TorchVOCDetection
from torchvision import transforms as T

from dataset.box_ops import xyxy_to_cxcywh, normalize_boxes


# ────────────────────────────────────────────────
# VOC 클래스 레이블
# ────────────────────────────────────────────────

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_IDX = {cls: i for i, cls in enumerate(VOC_CLASSES)}


# ────────────────────────────────────────────────
# 커스텀 augmentation 유틸
# ────────────────────────────────────────────────

def _resize_shortest_edge(
    img_np: np.ndarray,
    boxes_xyxy: np.ndarray,
    min_size: int = 800,
    max_size: int = 1333,
):
    """
    Purpose: shortest edge 기준 리사이즈 — 이미지와 boxes 동시 변환.
    Inputs:
        img_np:     np.ndarray [H, W, 3], uint8
        boxes_xyxy: np.ndarray [N, 4],    float32, xyxy pixel
        min_size:   shortest edge 목표 크기
        max_size:   longest edge 최대 크기
    Outputs:
        img_resized:   np.ndarray [H', W', 3], uint8
        boxes_resized: np.ndarray [N, 4],      float32, xyxy pixel
    """
    h, w = img_np.shape[:2]
    scale = min_size / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resized = np.array(Image.fromarray(img_np).resize((new_w, new_h), Image.BILINEAR))

    boxes_resized = boxes_xyxy * scale if len(boxes_xyxy) > 0 else boxes_xyxy.copy()
    return img_resized, boxes_resized


def _random_flip(img_np: np.ndarray, boxes_xyxy: np.ndarray, p: float = 0.5):
    """
    Purpose: 수평 flip (확률 p) — 이미지와 boxes 동시 변환.
    Inputs:
        img_np:     np.ndarray [H, W, 3]
        boxes_xyxy: np.ndarray [N, 4], float32, xyxy pixel
        p:          flip 확률
    Outputs:
        img_out:    np.ndarray [H, W, 3]
        boxes_out:  np.ndarray [N, 4], float32, xyxy pixel
    """
    if random.random() >= p:
        return img_np, boxes_xyxy

    img_out = img_np[:, ::-1, :].copy()
    if len(boxes_xyxy) == 0:
        return img_out, boxes_xyxy.copy()

    w = img_np.shape[1]
    boxes_out = boxes_xyxy.copy()
    boxes_out[:, 0] = w - boxes_xyxy[:, 2]
    boxes_out[:, 2] = w - boxes_xyxy[:, 0]
    return img_out, boxes_out


# ────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────

class VOCDetection(torch.utils.data.Dataset):
    """
    VOC 2007 / 2012 detection dataset.

    Args:
        root:        데이터 루트 경로 (e.g. "data/voc")
        year:        "2007" or "2012"
        split:       "train", "val", "trainval", "test" (2007만 test 지원)
        download:    True면 자동 다운로드
        min_size:    resize shortest edge
        max_size:    resize longest edge 상한
        skip_difficult: True면 difficult=1 객체 제외
        mean:        normalize mean (ImageNet 기본값)
        std:         normalize std  (ImageNet 기본값)
    """

    def __init__(
        self,
        root: str,
        year: str = "2007",
        split: str = "train",
        download: bool = False,
        min_size: int = 800,
        max_size: int = 1333,
        skip_difficult: bool = True,
        mean: tuple = (0.485, 0.456, 0.406),
        std:  tuple = (0.229, 0.224, 0.225),
    ):
        assert year in ("2007", "2012"), f"year must be 2007 or 2012, got {year}"
        image_set = split

        self._torch_voc = _TorchVOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=None,
            target_transform=None,
        )

        self.split = split
        self.skip_difficult = skip_difficult
        self.min_size = min_size
        self.max_size = max_size
        self.normalize = T.Normalize(mean=mean, std=std)
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self._torch_voc)

    def __getitem__(self, idx: int) -> dict:
        img_pil, target = self._torch_voc[idx]
        orig_w, orig_h = img_pil.size  # PIL은 (W, H)

        # XML annotation 파싱
        boxes_xyxy, labels = self._parse_annotation(target, orig_w, orig_h)

        # Augmentation 적용
        img_np = np.array(img_pil)  # [H, W, 3], uint8
        img_aug, boxes_aug = _resize_shortest_edge(img_np, boxes_xyxy, self.min_size, self.max_size)
        if self.split == "train":
            img_aug, boxes_aug = _random_flip(img_aug, boxes_aug)

        aug_h, aug_w = img_aug.shape[:2]

        # Tensor 변환 + normalize
        img_tensor = self.to_tensor(img_aug)    # [3, H', W'], float32, [0,1]
        img_tensor = self.normalize(img_tensor) # ImageNet normalize

        # Box: xyxy pixel → cxcywh pixel → normalized cxcywh
        if len(boxes_aug) > 0:
            boxes_t = torch.as_tensor(boxes_aug, dtype=torch.float32)
            boxes_t = xyxy_to_cxcywh(boxes_t)
            boxes_t = normalize_boxes(boxes_t, img_w=aug_w, img_h=aug_h)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)

        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        return {
            "image":     img_tensor,          # [3, H', W'], float32
            "boxes":     boxes_t,             # [N, 4], normalized cxcywh
            "labels":    labels_t,            # [N], int64
            "image_id":  str(idx),
            "orig_size": (orig_h, orig_w),
        }

    def _parse_annotation(self, target: dict, img_w: int, img_h: int):
        """
        VOC XML annotation dict → xyxy boxes, label indices

        Outputs:
            boxes_xyxy: np.ndarray [N, 4], float32, xyxy pixel
            labels:     list[int]
        """
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):  # 객체가 1개일 때 dict로 옴
            objects = [objects]

        boxes, labels = [], []
        for obj in objects:
            if self.skip_difficult and int(obj.get("difficult", 0)) == 1:
                continue

            cls_name = obj["name"]
            if cls_name not in VOC_CLASS_TO_IDX:
                continue

            bb = obj["bndbox"]
            x1 = float(bb["xmin"]) - 1  # VOC는 1-indexed
            y1 = float(bb["ymin"]) - 1
            x2 = float(bb["xmax"]) - 1
            y2 = float(bb["ymax"]) - 1

            # 유효성 검사
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(VOC_CLASS_TO_IDX[cls_name])

        if boxes:
            return np.array(boxes, dtype=np.float32), labels
        else:
            return np.zeros((0, 4), dtype=np.float32), []


# ────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────

def visualize_sample(sample: dict, idx: int = 0, save_path: str = None):
    """
    Dataset sample을 박스와 함께 시각화 (cv2 기반).

    Inputs:
        sample:    __getitem__ 반환값
        idx:       표시용 인덱스
        save_path: None이면 창 표시 후 아무 키나 누르면 종료,
                   경로 지정 시 파일 저장
    """
    import cv2

    # ImageNet denormalize → uint8 BGR
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = (sample["image"] * std + mean).clamp(0, 1)            # [3, H, W]
    img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [H, W, 3] RGB
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    boxes  = sample["boxes"]   # [N, 4], normalized cxcywh
    labels = sample["labels"]  # [N]

    for box, label in zip(boxes, labels):
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        cls_name = VOC_CLASSES[label.item()]
        (tw, th), baseline = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - baseline - 4), (x1 + tw, y1), (0, 0, 0), -1)
        cv2.putText(img_bgr, cls_name, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_bgr)
        print(f"Saved: {save_path}")
    else:
        cv2.imshow(f"VOC sample #{idx}  |  {len(boxes)} objects", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ────────────────────────────────────────────────
# Main test
# ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root",     default="data/voc",  help="VOC 데이터 루트")
    parser.add_argument("--year",     default="2007",      choices=["2007", "2012"])
    parser.add_argument("--split",    default="val",       choices=["train", "val", "trainval", "test"])
    parser.add_argument("--download", action="store_true", help="데이터 자동 다운로드")
    parser.add_argument("--vis_idx",  type=int, default=0, help="시각화할 샘플 인덱스")
    parser.add_argument("--save",     default=None,        help="저장 경로 (없으면 plt.show)")
    args = parser.parse_args()

    print(f"Loading VOC {args.year} {args.split} from {args.root} ...")
    dataset = VOCDetection(
        root=args.root,
        year=args.year,
        split=args.split,
        download=args.download,
    )
    print(f"  총 샘플 수: {len(dataset)}")

    sample = dataset[args.vis_idx]

    # Shape 검증
    assert sample["image"].shape[0] == 3,          "image channel != 3"
    assert sample["boxes"].ndim == 2,              "boxes should be 2D"
    assert sample["boxes"].shape[-1] == 4,         "boxes last dim != 4"
    assert sample["labels"].shape[0] == sample["boxes"].shape[0], "boxes/labels mismatch"
    assert sample["boxes"].max() <= 1.0 + 1e-4,   "boxes not normalized (max > 1)"
    assert sample["boxes"].min() >= -1e-4,         "boxes not normalized (min < 0)"
    print(f"  image shape:  {sample['image'].shape}")
    print(f"  boxes shape:  {sample['boxes'].shape}  (normalized cxcywh)")
    print(f"  labels:       {[VOC_CLASSES[l] for l in sample['labels'].tolist()]}")
    print(f"  orig_size:    {sample['orig_size']}")
    print("  Shape checks passed.")

    # Visualization
    visualize_sample(sample, idx=args.vis_idx, save_path=args.save)
