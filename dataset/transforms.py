"""
Detection transforms — pure torchvision/PIL, no detectron2 dependency.

All transforms operate on (PIL.Image, np.ndarray[N,4] xyxy-pixel) pairs.
Final ToTensor + Normalize converts to torch.Tensor.
"""

import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: Image.Image, boxes: np.ndarray):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes


class ResizeShortestEdge:
    """
    Resize so shortest edge == min_size, longest edge <= max_size.
    Maintains aspect ratio. Boxes scaled accordingly.
    """

    def __init__(self, min_size: int = 800, max_size: int = 1333):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(
        self, image: Image.Image, boxes: np.ndarray
    ) -> tuple[Image.Image, np.ndarray]:
        """
        Inputs:
            image: PIL.Image  (W, H)
            boxes: [N, 4] float32, xyxy pixel
        Outputs:
            image: PIL.Image resized
            boxes: [N, 4] float32, xyxy pixel (scaled)
        """
        w, h = image.size
        scale = self.min_size / min(h, w)
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)

        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)

        if len(boxes):
            boxes = boxes * scale

        return image, boxes


class RandomHorizontalFlip:
    """
    Random horizontal flip. Boxes x-coords reflected accordingly.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: Image.Image, boxes: np.ndarray
    ) -> tuple[Image.Image, np.ndarray]:
        """
        Inputs:
            image: PIL.Image
            boxes: [N, 4] float32, xyxy pixel
        Outputs:
            image, boxes (possibly flipped)
        """
        if random.random() < self.p:
            w = image.width
            image = TF.hflip(image)
            if len(boxes):
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
        return image, boxes


class ToTensor:
    """PIL.Image [H,W,3] uint8 → torch.Tensor [3,H,W] float32 in [0,1]."""

    def __call__(
        self, image: Image.Image, boxes: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        return TF.to_tensor(image), boxes


class Normalize:
    """Normalize tensor with mean/std. Passes boxes through unchanged."""

    def __init__(
        self,
        mean: tuple = (0.485, 0.456, 0.406),
        std:  tuple = (0.229, 0.224, 0.225),
    ):
        self.mean = mean
        self.std  = std

    def __call__(
        self, image: torch.Tensor, boxes: np.ndarray
    ) -> tuple[torch.Tensor, np.ndarray]:
        return TF.normalize(image, self.mean, self.std), boxes


def build_transforms(
    split:    str = "train",
    min_size: int = 800,
    max_size: int = 1333,
) -> Compose:
    """
    Standard detection transform pipeline.

    train: ResizeShortestEdge → RandomHorizontalFlip → ToTensor → Normalize
    val:   ResizeShortestEdge → ToTensor → Normalize
    """
    tfms = [ResizeShortestEdge(min_size, max_size)]
    if split == "train":
        tfms.append(RandomHorizontalFlip(p=0.5))
    tfms += [ToTensor(), Normalize()]
    return Compose(tfms)


if __name__ == "__main__":
    print("=== transforms.py sanity check ===")
    import torch

    img = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    boxes = np.array([[50., 30., 200., 150.], [300., 100., 500., 400.]], dtype=np.float32)

    # Train pipeline
    tfm = build_transforms("train", min_size=800, max_size=1333)
    img_t, boxes_t = tfm(img, boxes.copy())
    assert isinstance(img_t, torch.Tensor) and img_t.shape[0] == 3
    assert isinstance(boxes_t, np.ndarray) and boxes_t.shape == (2, 4)
    print(f"  train: image {img_t.shape}, boxes {boxes_t.shape}  ✓")

    # Val pipeline (no flip)
    tfm_val = build_transforms("val", min_size=800, max_size=1333)
    img_v, boxes_v = tfm_val(img, boxes.copy())
    assert img_v.shape[0] == 3
    print(f"  val:   image {img_v.shape}, boxes {boxes_v.shape}  ✓")

    # Empty boxes
    img_e, boxes_e = tfm(img, np.zeros((0, 4), dtype=np.float32))
    assert boxes_e.shape == (0, 4)
    print(f"  empty boxes: {boxes_e.shape}  ✓")

    print("All checks passed.")
