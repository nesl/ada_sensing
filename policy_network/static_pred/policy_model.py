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
        base_model = mobilenet_v3_small(weights=weights)

        # torchvision MobileNetV3-Small classifier:
        # Sequential(
        #   (0): Linear(...)
        #   (1): Hardswish()
        #   (2): Dropout(...)
        #   (3): Linear(..., 1000)
        # )
        head_in_features = base_model.classifier[3].in_features
        self.backbone = base_model.features
        self.avgpool = base_model.avgpool
        self.feature_proj = nn.Sequential(
            base_model.classifier[0],
            base_model.classifier[1],
            base_model.classifier[2],
        )
        self.policy_head = nn.Linear(head_in_features, num_candidates)

    def freeze_backbone(self) -> None:
        for module in (self.backbone, self.avgpool, self.feature_proj):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for module in (self.backbone, self.avgpool, self.feature_proj):
            for param in module.parameters():
                param.requires_grad = True

    def unfreeze_backbone_tail(self, start_idx: int = 9) -> None:
        self.freeze_backbone()
        for module in list(self.backbone.children())[start_idx:]:
            for param in module.parameters():
                param.requires_grad = True
        for param in self.feature_proj.parameters():
            param.requires_grad = True

    def get_backbone_tail_parameters(self):
        params = []
        for module in list(self.backbone.children())[9:]:
            params.extend(list(module.parameters()))
        params.extend(list(self.feature_proj.parameters()))
        return [param for param in params if param.requires_grad]

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_proj(x)
        return self.policy_head(x)
