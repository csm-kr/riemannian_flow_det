from torch.utils.data import DataLoader

from dataset.coco      import COCODetection
from dataset.voc       import VOCDetection
from dataset.mnist_box import MNISTBoxDetection
from dataset.collate   import build_dataloader as _build_dataloader


def build_dataset(config, split: str):
    """
    Purpose: Build a detection dataset from config.
    Inputs:
        config: dict or namespace with keys:
                dataset         — "coco" | "voc" | "mnist_box"
                data_root       — dataset root path
                --- COCO ---
                coco_ann_train  — annotation JSON for train
                coco_ann_val    — annotation JSON for val
                --- VOC ---
                voc_year        — "2007" | "2012"
                --- MNIST Box (toy) ---
                mnist_root               — default f"{data_root}/mnist"
                mnist_image_size         — default 256
                mnist_scale_range        — default (14, 56)
                mnist_num_samples_train  — default 50000
                mnist_num_samples_val    — default 5000
                mnist_bbox_mode          — "fixed" | "tight"  (default "fixed")
                mnist_background         — "zero"  | "noise"  (default "zero")
                seed                     — default 0
                --- common ---
                min_size        — resize shortest edge (default 800; ignored by mnist_box)
                max_size        — resize longest edge  (default 1333; ignored by mnist_box)
        split:  "train" | "val"
    Outputs:
        dataset: torch.utils.data.Dataset
    """
    def _get(key, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    name     = _get("dataset", "coco").lower()
    root     = _get("data_root", "data")
    min_size = _get("min_size", 800)
    max_size = _get("max_size", 1333)

    if name == "coco":
        if split == "train":
            img_dir  = _get("coco_train_dir",  f"{root}/coco/train2017")
            ann_file = _get("coco_ann_train",  f"{root}/coco/annotations/instances_train2017.json")
        else:
            img_dir  = _get("coco_val_dir",    f"{root}/coco/val2017")
            ann_file = _get("coco_ann_val",    f"{root}/coco/annotations/instances_val2017.json")
        return COCODetection(
            root=img_dir, ann_file=ann_file,
            split=split, min_size=min_size, max_size=max_size,
        )

    elif name == "voc":
        voc_root = _get("voc_root", f"{root}/voc")
        year     = str(_get("voc_year", "2007"))
        return VOCDetection(
            root=voc_root, year=year, split=split,
            min_size=min_size, max_size=max_size,
        )

    elif name == "mnist_box":
        default_n = 50000 if split == "train" else 5000
        return MNISTBoxDetection(
            root        = _get("mnist_root",        f"{root}/mnist"),
            split       = split,
            image_size  = _get("mnist_image_size",  256),
            scale_range = tuple(_get("mnist_scale_range", (14, 56))),
            num_samples = _get(
                "mnist_num_samples_train" if split == "train" else "mnist_num_samples_val",
                default_n,
            ),
            bbox_mode   = _get("mnist_bbox_mode",   "fixed"),
            background  = _get("mnist_background",  "zero"),
            seed        = _get("seed", 0),
            download    = _get("mnist_download",    True),
        )

    else:
        raise ValueError(
            f"Unknown dataset: {name!r}. Choose 'coco' | 'voc' | 'mnist_box'."
        )


def build_dataloader(config, split: str) -> DataLoader:
    """
    Purpose: Build dataset + DataLoader from config.
    Inputs:
        config: dict or namespace (same as build_dataset, plus):
                batch_size   — default 2
                num_workers  — default 4
                pin_memory   — default True
        split:  "train" | "val"
    Outputs:
        DataLoader
    """
    def _get(key, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    dataset     = build_dataset(config, split)
    batch_size  = _get("batch_size",  2)
    num_workers = _get("num_workers", 4)
    pin_memory  = _get("pin_memory",  True)
    shuffle     = (split == "train")

    return _build_dataloader(
        dataset,
        batch_size  = batch_size,
        num_workers = num_workers,
        shuffle     = shuffle,
        pin_memory  = pin_memory,
        drop_last   = shuffle,
    )


if __name__ == "__main__":
    """
    Smoke test: mnist_box dataset → DataLoader → 1 batch shape 검증 (MB3).
    collate_fn 호환 + 기존 detection 파이프라인과 동일한 출력 형태 확인.

    실행:
        python -m dataset            # mnist_box smoke
        python -m dataset --dataset voc --split val   # VOC로 바꿔 테스트 가능
    """
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="mnist_box",
                        choices=["mnist_box", "voc", "coco"])
    parser.add_argument("--split",      default="train", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg = {
        "dataset":       args.dataset,
        "batch_size":    args.batch_size,
        "num_workers":   args.num_workers,
        "pin_memory":    False,
        # mnist_box: 소규모 샘플로 빠르게
        "mnist_num_samples_train": 32,
        "mnist_num_samples_val":   16,
    }

    print(f"[build] dataset={args.dataset}  split={args.split}  batch_size={args.batch_size}")
    loader = build_dataloader(cfg, split=args.split)
    print(f"  len(dataset) = {len(loader.dataset)}")

    batch = next(iter(loader))
    images     = batch["images"]
    boxes_list = batch["boxes"]
    labels_list = batch["labels"]

    B = args.batch_size
    print(f"  images:      {tuple(images.shape)}  dtype={images.dtype}")
    print(f"  boxes[0]:    {tuple(boxes_list[0].shape)}")
    print(f"  labels[0]:   {tuple(labels_list[0].shape)} → {labels_list[0].tolist()}")
    print(f"  image_ids:   {batch['image_ids']}")
    print(f"  orig_sizes:  {batch['orig_sizes']}")

    # 공통 assert
    assert images.ndim == 4 and images.shape[0] == B
    assert isinstance(boxes_list,  list) and len(boxes_list)  == B
    assert isinstance(labels_list, list) and len(labels_list) == B
    for i in range(B):
        assert boxes_list[i].shape[-1] == 4
        assert boxes_list[i].shape[0]  == labels_list[i].shape[0]

    # mnist_box 전용 assert: 모든 샘플이 10개 박스 + labels == [0..9]
    if args.dataset == "mnist_box":
        for i in range(B):
            assert boxes_list[i].shape == (10, 4), \
                f"mnist_box box shape expected (10,4), got {tuple(boxes_list[i].shape)}"
            assert (labels_list[i] == torch.arange(10)).all(), \
                "mnist_box labels must be [0..9] (1-to-1 class-indexed)"
        assert images.shape[1:] == (3, 256, 256)
        print("  [ok] mnist_box: every sample has 10 class-indexed boxes")

    print("All checks passed.")
