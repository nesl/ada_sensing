import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)


SUPPORTED_BACKBONES = ("mobilenet_v3_small", "resnet18")


class SensorPolicyNetwork(nn.Module):
    """
    Input:
        x: [B, 3, H, W] baseline image
    Output:
        logits: [B, num_candidates]
    """
    def __init__(
        self,
        num_candidates: int = 27,
        pretrained: bool = True,
        backbone_name: str = "mobilenet_v3_small",
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.partial_unfreeze_start_idx = 0

        if backbone_name == "mobilenet_v3_small":
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
            self.partial_unfreeze_start_idx = 9
        elif backbone_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
            head_in_features = base_model.fc.in_features
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            self.avgpool = nn.Identity()
            self.feature_proj = nn.Identity()
            self.partial_unfreeze_start_idx = 7
        else:
            raise ValueError(
                f"Unsupported backbone_name={backbone_name}. "
                f"Supported: {SUPPORTED_BACKBONES}"
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

    def unfreeze_backbone_tail(self, start_idx: int | None = None) -> None:
        if start_idx is None:
            start_idx = self.partial_unfreeze_start_idx
        self.freeze_backbone()
        for module in list(self.backbone.children())[start_idx:]:
            for param in module.parameters():
                param.requires_grad = True
        for param in self.feature_proj.parameters():
            param.requires_grad = True

    def get_backbone_tail_parameters(self, start_idx: int | None = None):
        if start_idx is None:
            start_idx = self.partial_unfreeze_start_idx
        params = []
        for module in list(self.backbone.children())[start_idx:]:
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


def infer_backbone_name_from_checkpoint(checkpoint: dict) -> str:
    if "backbone_name" in checkpoint:
        return checkpoint["backbone_name"]
    if "backbone" in checkpoint:
        return checkpoint["backbone"]
    return "mobilenet_v3_small"
