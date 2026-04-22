from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.ops as tvops


# ── DINOv2 Backbone ───────────────────────────────────────────────────────────

_DINOV2_EMBED_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2Backbone(nn.Module):
    """
    DINOv2 ViT feature extractor with linear projection to hidden dim.

    Uses Meta's DINOv2 (facebook research) via torch.hub.
    Patch size = 14 → images padded to multiples of 14.
    Outputs single-scale patch token sequence (no FPN).

    Compatible drop-in for FPNBackbone — same forward() interface.
    """

    def __init__(
        self,
        dim:        int  = 256,
        model_name: str  = "dinov2_vits14",
        pretrained: bool = True,
        freeze:     bool = False,
    ):
        super().__init__()
        assert model_name in _DINOV2_EMBED_DIM, \
            f"Unknown DINOv2 model: {model_name}. " \
            f"Choose from {list(_DINOV2_EMBED_DIM)}"

        self.patch_size = 14
        self.embed_dim  = _DINOV2_EMBED_DIM[model_name]
        self.dim        = dim

        # Load DINOv2 from torch.hub
        self.dino = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
        )
        self.dino.eval()

        if freeze:
            for p in self.dino.parameters():
                p.requires_grad_(False)

        # Project DINOv2 embed_dim → hidden dim
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, dim),
            nn.LayerNorm(dim),
        )

    def _pad_to_patch(self, images: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad H,W to nearest multiple of patch_size=14."""
        _, _, H, W = images.shape
        p = self.patch_size
        H_pad = (H + p - 1) // p * p
        W_pad = (W + p - 1) // p * p
        if H_pad != H or W_pad != W:
            images = F.pad(images, (0, W_pad - W, 0, H_pad - H))
        return images, H_pad, W_pad

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """
        Purpose: Extract patch-level features via DINOv2 ViT,
                 project to hidden dim, and return token sequence.
        Inputs:
            images: [B, 3, H, W], float32, ImageNet-normalized
        Outputs:
            img_tokens: [B, H_p*W_p, dim], float32
                        H_p = H_pad // 14, W_p = W_pad // 14
            hw_list:    [(H_p, W_p)]  — single scale, for RoPE
        """
        images, H_pad, W_pad = self._pad_to_patch(images)
        H_p = H_pad // self.patch_size
        W_p = W_pad // self.patch_size

        # DINOv2 forward — get patch tokens (exclude CLS)
        with torch.set_grad_enabled(self.training and
                                    next(self.dino.parameters()).requires_grad):
            feats = self.dino.forward_features(images)

        # feats["x_norm_patchtokens"]: [B, H_p*W_p, embed_dim]
        patch_tokens = feats["x_norm_patchtokens"]   # [B, L, embed_dim]

        img_tokens = self.proj(patch_tokens)          # [B, L, dim]
        return img_tokens, [(H_p, W_p)]


# ── ResNet50 + FPN Backbone ───────────────────────────────────────────────────

class FPNBackbone(nn.Module):
    def __init__(self, dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        resnet  = tvm.resnet50(weights=weights)

        # ResNet body — split into stages for FPN feature extraction
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 256 ch, stride 4  (not used by FPN)
        self.layer2 = resnet.layer2   # 512 ch, stride 8   → P3
        self.layer3 = resnet.layer3   # 1024 ch, stride 16 → P4
        self.layer4 = resnet.layer4   # 2048 ch, stride 32 → P5

        self.fpn = tvops.FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],
            out_channels=dim,
        )
        self.dim = dim

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """
        Purpose: Extract multi-scale features via ResNet50+FPN,
                 flatten and concatenate into a single token sequence.
        Inputs:
            images: [B, 3, H, W], float32, ImageNet-normalized
        Outputs:
            img_tokens: [B, L_img, dim], float32
                        L_img = H/8*W/8 + H/16*W/16 + H/32*W/32
            hw_list:    [(H_P3,W_P3), (H_P4,W_P4), (H_P5,W_P5)]
                        used by DiT to build image-token RoPE frequencies
        """
        x  = self.stem(images)     # [B, 64,   H/4,  W/4]
        x  = self.layer1(x)        # [B, 256,  H/4,  W/4]
        c3 = self.layer2(x)        # [B, 512,  H/8,  W/8]
        c4 = self.layer3(c3)       # [B, 1024, H/16, W/16]
        c5 = self.layer4(c4)       # [B, 2048, H/32, W/32]

        feat_maps = self.fpn(OrderedDict([("0", c3), ("1", c4), ("2", c5)]))
        # feat_maps keys: "0" → P3 (H/8), "1" → P4 (H/16), "2" → P5 (H/32)

        tokens_list: list[torch.Tensor] = []
        hw_list:     list[tuple[int, int]] = []

        for key in ["0", "1", "2"]:
            feat = feat_maps[key]                        # [B, dim, H_k, W_k]
            B, D, H_k, W_k = feat.shape
            tok = feat.flatten(2).transpose(1, 2)        # [B, H_k*W_k, dim]
            tokens_list.append(tok)
            hw_list.append((H_k, W_k))

        img_tokens = torch.cat(tokens_list, dim=1)       # [B, L_img, dim]
        return img_tokens, hw_list


if __name__ == "__main__":
    print("=== backbone.py sanity check ===")

    # ── FPNBackbone ──────────────────────────────────────────────────────────
    print("\n[FPNBackbone]")
    fpn = FPNBackbone(dim=256, pretrained=False)
    fpn.eval()

    H, W = 800, 800
    x = torch.zeros(1, 3, H, W)

    with torch.no_grad():
        img_tokens, hw_list = fpn(x)

    expected_L = sum(h * w for h, w in hw_list)
    assert img_tokens.shape == (1, expected_L, 256), f"shape mismatch: {img_tokens.shape}"
    print(f"  img_tokens: {img_tokens.shape}  ✓")
    print(f"  hw_list:    {hw_list}")
    for i, (h, w) in enumerate(hw_list):
        print(f"    P{i+3}: {h}×{w}  (stride {H//h})")

    # ── DINOv2Backbone ───────────────────────────────────────────────────────
    print("\n[DINOv2Backbone — dinov2_vits14]")
    dino = DINOv2Backbone(dim=256, model_name="dinov2_vits14", pretrained=True)
    dino.eval()

    # Test 1: exact multiple of 14
    x1 = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        tok1, hw1 = dino(x1)
    expected_L1 = sum(h * w for h, w in hw1)
    assert tok1.shape == (1, expected_L1, 256), f"shape mismatch: {tok1.shape}"
    print(f"  224×224 → img_tokens: {tok1.shape}, hw_list: {hw1}  ✓")

    # Test 2: non-multiple of 14 (auto-padded)
    x2 = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        tok2, hw2 = dino(x2)
    expected_L2 = sum(h * w for h, w in hw2)
    assert tok2.shape == (1, expected_L2, 256), f"shape mismatch: {tok2.shape}"
    print(f"  256×256 → img_tokens: {tok2.shape}, hw_list: {hw2}  ✓  (padded to 266×266)")

    # Test 3: batch
    x3 = torch.zeros(2, 3, 448, 448)
    with torch.no_grad():
        tok3, hw3 = dino(x3)
    assert tok3.shape[0] == 2
    print(f"  batch=2 448×448 → {tok3.shape}, hw_list: {hw3}  ✓")

    print("\nAll checks passed.")

    # Verify stride relationship
    for i, (h, w) in enumerate(hw_list):
        stride = H // h
        print(f"  P{i+3}: {h}×{w}  (stride {stride})")

    print("All checks passed.")
