import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class SensorPolicyNetwork(nn.Module):
    """
    Input:
        x: [B, 3, H, W] baseline image
    Output:
        logits: [B, num_candidates]
    """
    def __init__(self, num_candidates: int = 27, pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        # torchvision MobileNetV3-Small classifier:
        # Sequential(
        #   (0): Linear(...)
        #   (1): Hardswish()
        #   (2): Dropout(...)
        #   (3): Linear(..., 1000)
        # )
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_candidates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)