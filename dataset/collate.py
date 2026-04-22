"""
Collate function and DataLoader builder for detection datasets.

boxes/labels are kept as lists (variable-length per image) — no padding here.
Padding is done inside the model's forward_train when needed.
"""

import torch
from torch.utils.data import DataLoader, Dataset


def detection_collate_fn(batch: list[dict]) -> dict:
    """
    Purpose: Collate a list of detection samples into a batch.
    Inputs:
        batch: list of dicts, each with keys:
               image [3,H,W], boxes [Ni,4], labels [Ni],
               image_id, orig_size
    Outputs:
        dict:
            images:    [B, 3, H, W]  float32  (zero-padded to max H,W in batch)
            boxes:     list of B tensors [Ni, 4]  normalized cxcywh
            labels:    list of B tensors [Ni]
            image_ids: list of B image_id values
            orig_sizes: list of B (H, W) tuples
    """
    images     = [s["image"]     for s in batch]
    boxes      = [s["boxes"]     for s in batch]
    labels     = [s["labels"]    for s in batch]
    image_ids  = [s["image_id"]  for s in batch]
    orig_sizes = [s["orig_size"] for s in batch]

    # Pad images to the same H, W within batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = torch.zeros(len(images), 3, max_h, max_w, dtype=images[0].dtype)
    for i, img in enumerate(images):
        _, h, w = img.shape
        padded[i, :, :h, :w] = img

    return {
        "images":     padded,
        "boxes":      boxes,
        "labels":     labels,
        "image_ids":  image_ids,
        "orig_sizes": orig_sizes,
    }


def build_dataloader(
    dataset:     Dataset,
    batch_size:  int  = 2,
    num_workers: int  = 4,
    shuffle:     bool = True,
    pin_memory:  bool = True,
    drop_last:   bool = False,
) -> DataLoader:
    """
    Purpose: Build a DataLoader with detection_collate_fn.
    Inputs:
        dataset:     torch.utils.data.Dataset instance
        batch_size:  samples per batch
        num_workers: parallel data loading workers
        shuffle:     True for training, False for eval
        pin_memory:  True when using GPU
        drop_last:   drop last incomplete batch
    Outputs:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = detection_collate_fn,
        pin_memory  = pin_memory,
        drop_last   = drop_last,
    )


if __name__ == "__main__":
    print("=== collate.py sanity check ===")
    from PIL import Image
    import numpy as np
    from dataset.transforms import build_transforms

    # Fake dataset with variable-size images
    class _FakeDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.tfm = build_transforms("val", min_size=200, max_size=400)
        def __len__(self): return 4
        def __getitem__(self, idx):
            h = 300 + idx * 50
            img_pil = Image.fromarray(
                np.random.randint(0, 255, (h, 400, 3), dtype=np.uint8)
            )
            boxes = np.array([[10., 10., 100., 100.]], dtype=np.float32)
            img_t, boxes_np = self.tfm(img_pil, boxes)
            _, ih, iw = img_t.shape
            from dataset.box_ops import xyxy_to_cxcywh, normalize_boxes
            boxes_t = normalize_boxes(xyxy_to_cxcywh(
                torch.as_tensor(boxes_np, dtype=torch.float32)
            ), img_w=iw, img_h=ih)
            return {
                "image":     img_t,
                "boxes":     boxes_t,
                "labels":    torch.zeros(1, dtype=torch.int64),
                "image_id":  idx,
                "orig_size": (h, 400),
            }

    ds     = _FakeDataset()
    loader = build_dataloader(ds, batch_size=2, num_workers=0, shuffle=False)
    batch  = next(iter(loader))

    assert batch["images"].ndim == 4 and batch["images"].shape[0] == 2
    assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
    assert isinstance(batch["labels"], list)
    print(f"  images:  {batch['images'].shape}  ✓")
    print(f"  boxes[0]: {batch['boxes'][0].shape}  ✓")
    print(f"  image_ids: {batch['image_ids']}  ✓")
    print("All checks passed.")
