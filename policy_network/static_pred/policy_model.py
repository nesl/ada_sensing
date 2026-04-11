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
        input_mode: str = "single",
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.input_mode = input_mode
        self.partial_unfreeze_start_idx = 0
        self.num_input_views = 2 if input_mode == "dual" else 1
        if input_mode not in {"single", "dual"}:
            raise ValueError(f"Unsupported input_mode={input_mode}")

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

        self.feature_dim = head_in_features
        self.policy_head = nn.Linear(head_in_features * self.num_input_views, num_candidates)

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

    def encode_single_view(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.feature_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "single":
            if x.dim() != 4:
                raise ValueError(
                    f"Expected single-view input of shape [B, C, H, W], got {tuple(x.shape)}"
                )
            features = self.encode_single_view(x)
            return self.policy_head(features)

        if x.dim() != 5:
            raise ValueError(
                "Expected dual-view input of shape [B, 2, C, H, W], "
                f"got {tuple(x.shape)}"
            )
        batch_size, num_views = x.shape[:2]
        if num_views != self.num_input_views:
            raise ValueError(
                f"Expected {self.num_input_views} views for input_mode={self.input_mode}, got {num_views}"
            )
        x = x.view(batch_size * num_views, *x.shape[2:])
        features = self.encode_single_view(x)
        features = features.view(batch_size, num_views, self.feature_dim)
        features = features.reshape(batch_size, num_views * self.feature_dim)
        return self.policy_head(features)


def infer_backbone_name_from_checkpoint(checkpoint: dict) -> str:
    if "backbone_name" in checkpoint:
        return checkpoint["backbone_name"]
    if "backbone" in checkpoint:
        return checkpoint["backbone"]
    return "mobilenet_v3_small"


def infer_input_mode_from_checkpoint(checkpoint: dict) -> str:
    return checkpoint.get("input_mode", "single")
