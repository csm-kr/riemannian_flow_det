# File: model/backbone.py
# Role: 이미지 feature 추출 — ResNet-50 + FPN P4 → patch tokens

import torch
import torch.nn as nn
import torchvision.models as tvm


class ImageBackbone(nn.Module):
    """
    Purpose: ResNet-50 layer3(C4) 추출 → FPN lateral conv → flatten patch tokens.
    Inputs:
        images: [B, 3, H, W], float32 — ImageNet normalized
    Outputs:
        tokens: [B, S, d_model], float32 — S = (H/16) * (W/16)
                H=W=800 → S = 50*50 = 2500
    """
    def __init__(self, d_model: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet50(weights=weights)

        # stem: conv(3→64,k=7,s=2,p=3) → BN → ReLU → MaxPool(k=3,s=2,p=1)
        self.stem = nn.Sequential(
            resnet.conv1,   # [B, 64, H/2, W/2]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # [B, 64, H/4, W/4]
        )
        # layer1 (C2): Bottleneck×3, stride=1 → [B, 256, H/4, W/4]
        self.layer1 = resnet.layer1
        # layer2 (C3): Bottleneck×4, stride=2 → [B, 512, H/8, W/8]
        self.layer2 = resnet.layer2
        # layer3 (C4): Bottleneck×6, stride=2 → [B, 1024, H/16, W/16]  ← 추출 지점
        self.layer3 = resnet.layer3

        # FPN lateral conv: 1024 → d_model
        self.lateral = nn.Sequential(
            nn.Conv2d(1024, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Inputs:  images [B, 3, H, W]
        Outputs: tokens [B, S, d_model]   S = (H/16)*(W/16)
        """
        x = self.stem(images)    # [B, 64,   H/4,  W/4]
        x = self.layer1(x)       # [B, 256,  H/4,  W/4]
        x = self.layer2(x)       # [B, 512,  H/8,  W/8]
        x = self.layer3(x)       # [B, 1024, H/16, W/16]
        x = self.lateral(x)      # [B, d,    H/16, W/16]

        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, d]
        return tokens


# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== backbone.py sanity check ===")
    B, d = 2, 256

    model = ImageBackbone(d_model=d, pretrained=False)
    model.eval()

    # H=W=800 → S = 50*50 = 2500
    images = torch.randn(B, 3, 800, 800)
    with torch.no_grad():
        tokens = model(images)
    assert tokens.shape == (B, 2500, d), f"Expected (2,2500,256), got {tokens.shape}"
    print(f"ResNet-50+FPN P4 [800×800]: {tokens.shape} ✓")

    # H=W=448 → S = 28*28 = 784
    images2 = torch.randn(B, 3, 448, 448)
    with torch.no_grad():
        tokens2 = model(images2)
    assert tokens2.shape == (B, 784, d), f"Expected (2,784,256), got {tokens2.shape}"
    print(f"ResNet-50+FPN P4 [448×448]: {tokens2.shape} ✓")

    print("All checks passed.")
