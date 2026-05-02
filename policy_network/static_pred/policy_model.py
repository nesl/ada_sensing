import torch
import torch.nn as nn
import timm
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)


SUPPORTED_BACKBONES = (
    "mobilenet_v3_small",
    "resnet18",
    "tiny_conv_scratch",
    "dinov2_vits14",
)


class TinyConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def build_dinov2_vits14(pretrained: bool) -> nn.Module:
    try:
        return timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained,
            num_classes=0,
            img_size=224,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build dinov2_vits14 with timm model "
            "'vit_small_patch14_dinov2.lvd142m'. Check that the lens conda "
            "environment has a timm version with DINOv2 support and, for "
            "pretrained runs, access to the cached/downloadable weights."
        ) from exc


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
        num_input_views: int | None = None,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.input_mode = input_mode
        self.partial_unfreeze_start_idx = 0
        if input_mode == "single":
            self.num_input_views = 1
        elif input_mode == "dual":
            self.num_input_views = 2
        elif input_mode == "multiview":
            if num_input_views is None or num_input_views < 1:
                raise ValueError("num_input_views must be >= 1 when input_mode='multiview'.")
            self.num_input_views = int(num_input_views)
        else:
            raise ValueError(f"Unsupported input_mode={input_mode}")
        self.requires_full_training = False

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
        elif backbone_name == "tiny_conv_scratch":
            base_model = TinyConvBackbone()
            head_in_features = base_model.num_features
            self.backbone = base_model
            self.avgpool = nn.Identity()
            self.feature_proj = nn.Identity()
            self.partial_unfreeze_start_idx = 0
            self.requires_full_training = True
        elif backbone_name == "dinov2_vits14":
            base_model = build_dinov2_vits14(pretrained=pretrained)
            head_in_features = base_model.num_features
            self.backbone = base_model
            self.avgpool = nn.Identity()
            self.feature_proj = nn.Identity()
            self.partial_unfreeze_start_idx = -2
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
        if hasattr(self.backbone, "blocks"):
            blocks = list(self.backbone.blocks)
            for module in blocks[start_idx:]:
                for param in module.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            return
        for module in list(self.backbone.children())[start_idx:]:
            for param in module.parameters():
                param.requires_grad = True
        for param in self.feature_proj.parameters():
            param.requires_grad = True

    def get_backbone_tail_parameters(self, start_idx: int | None = None):
        if start_idx is None:
            start_idx = self.partial_unfreeze_start_idx
        if hasattr(self.backbone, "blocks"):
            params = []
            blocks = list(self.backbone.blocks)
            for module in blocks[start_idx:]:
                params.extend(list(module.parameters()))
            if hasattr(self.backbone, "norm"):
                params.extend(list(self.backbone.norm.parameters()))
            return [param for param in params if param.requires_grad]
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
                "Expected multi-view input of shape [B, V, C, H, W], "
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


def normalize_policy_checkpoint_state_dict(
    raw_state_dict: dict[str, torch.Tensor],
    backbone_name: str,
) -> dict[str, torch.Tensor]:
    """
    Keep evaluation scripts compatible with checkpoints saved by older model
    layouts. The tiny conv backbone used to save its feature stack directly as
    backbone.N.*, while the current TinyConvBackbone wraps it as
    backbone.features.N.*.
    """
    if backbone_name == "tiny_conv_scratch" and not any(
        key.startswith("backbone.features.") for key in raw_state_dict
    ):
        normalized: dict[str, torch.Tensor] = {}
        for key, value in raw_state_dict.items():
            if key.startswith("backbone.") and key.split(".", 2)[1].isdigit():
                new_key = "backbone.features." + key[len("backbone."):]
            else:
                new_key = key
            normalized[new_key] = value
        return normalized

    if backbone_name == "mobilenet_v3_small" and any(
        key.startswith("backbone.features.") for key in raw_state_dict
    ):
        normalized: dict[str, torch.Tensor] = {}
        for key, value in raw_state_dict.items():
            if key.startswith("backbone.features."):
                new_key = "backbone." + key[len("backbone.features."):]
            elif key.startswith("backbone.classifier.0"):
                new_key = "feature_proj.0" + key[len("backbone.classifier.0"):]
            elif key.startswith("backbone.classifier.1"):
                new_key = "feature_proj.1" + key[len("backbone.classifier.1"):]
            elif key.startswith("backbone.classifier.2"):
                new_key = "feature_proj.2" + key[len("backbone.classifier.2"):]
            elif key.startswith("backbone.classifier.3"):
                new_key = "policy_head" + key[len("backbone.classifier.3"):]
            else:
                new_key = key
            normalized[new_key] = value
        return normalized

    return raw_state_dict


def infer_backbone_name_from_checkpoint(checkpoint: dict) -> str:
    if "backbone_name" in checkpoint:
        return checkpoint["backbone_name"]
    if "backbone" in checkpoint:
        return checkpoint["backbone"]
    return "mobilenet_v3_small"


def infer_input_mode_from_checkpoint(checkpoint: dict) -> str:
    return checkpoint.get("input_mode", "single")


def infer_num_input_views_from_checkpoint(checkpoint: dict) -> int | None:
    if "num_input_views" in checkpoint:
        return int(checkpoint["num_input_views"])
    input_mode = infer_input_mode_from_checkpoint(checkpoint)
    if input_mode == "single":
        return 1
    if input_mode == "dual":
        return 2
    env_option_ids = checkpoint.get("env_option_ids") or []
    include_ae_input = bool(checkpoint.get("include_ae_input", False))
    if input_mode == "multiview" and env_option_ids:
        return len(env_option_ids) + int(include_ae_input)
    return None
