from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


PAPER_TITLE = "Adaptive Camera Sensor for Vision Models"
PAPER_TABLE = "Table 1"
PAPER_URL = "https://openreview.net/pdf?id=He2FGdmsas"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAPER_RESIZE_SIZE = 256
PAPER_CROP_SIZE = 224

DATASET_IN = "in"
DATASET_AE_ES = "ae_imagenet_es"
DATASET_AE_DIVERSE = "ae_imagenet_es_diverse"
DATASET_NAMES = (DATASET_IN, DATASET_AE_ES, DATASET_AE_DIVERSE)

DINOV2_HUB_REF = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    paper_name: str
    family: str
    source: str
    identifier: str
    checkpoint: str
    pretraining: str
    output_classes: int
    native_resolution: int
    native_resize: int | None
    native_interpolation: str
    paper_in: float
    paper_ae_es: float
    paper_ae_diverse: float
    recommended_batch_size: int


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "resnet50",
        "ResNet-50",
        "ResNet",
        "torchvision",
        "torchvision.models.resnet50",
        "ResNet50_Weights.IMAGENET1K_V1 / resnet50-0676ba61.pth",
        "ImageNet-1K supervised",
        1000,
        224,
        256,
        "bilinear",
        86.0,
        32.1,
        17.6,
        128,
    ),
    ModelSpec(
        "resnet50_deepaugment_augmix",
        "ResNet-50 + DeepAugment + AugMix",
        "ResNet",
        "hendrycks/imagenet-r",
        "torchvision.models.resnet50 + external state_dict",
        "deepaugment_and_augmix.pth.tar (Google Drive 1QKmc_p6-qDkh51WvsaS9HKFv8bX5jLnP)",
        "Paper says IN-21K; checkpoint publisher says ImageNet classifier (conflict)",
        1000,
        224,
        256,
        "bilinear",
        87.0,
        53.2,
        36.2,
        128,
    ),
    ModelSpec(
        "resnet152",
        "ResNet-152",
        "ResNet",
        "torchvision",
        "torchvision.models.resnet152",
        "ResNet152_Weights.IMAGENET1K_V1 / resnet152-394f9c45.pth",
        "ImageNet-1K supervised",
        1000,
        224,
        256,
        "bilinear",
        87.8,
        41.1,
        21.9,
        64,
    ),
    ModelSpec(
        "efficientnet_b0",
        "EfficientNet-B0",
        "EfficientNet",
        "torchvision",
        "torchvision.models.efficientnet_b0",
        "EfficientNet_B0_Weights.IMAGENET1K_V1 / efficientnet_b0_rwightman-7f5810bc.pth",
        "ImageNet-1K supervised",
        1000,
        224,
        256,
        "bicubic",
        88.2,
        51.8,
        21.8,
        128,
    ),
    ModelSpec(
        "efficientnet_b3",
        "EfficientNet-B3",
        "EfficientNet",
        "torchvision",
        "torchvision.models.efficientnet_b3",
        "EfficientNet_B3_Weights.IMAGENET1K_V1 / efficientnet_b3_rwightman-b3899882.pth",
        "ImageNet-1K supervised",
        1000,
        300,
        320,
        "bicubic",
        88.1,
        62.0,
        33.6,
        64,
    ),
    ModelSpec(
        "swin_v2_t",
        "SwinV2-T",
        "Swin Transformer V2",
        "torchvision",
        "torchvision.models.swin_v2_t",
        "Swin_V2_T_Weights.IMAGENET1K_V1 / swin_v2_t-b137f0e2.pth",
        "ImageNet-1K supervised",
        1000,
        256,
        260,
        "bicubic",
        90.6,
        54.1,
        26.5,
        64,
    ),
    ModelSpec(
        "swin_v2_s",
        "SwinV2-S",
        "Swin Transformer V2",
        "torchvision",
        "torchvision.models.swin_v2_s",
        "Swin_V2_S_Weights.IMAGENET1K_V1 / swin_v2_s-637d8ceb.pth",
        "ImageNet-1K supervised",
        1000,
        256,
        260,
        "bicubic",
        91.7,
        59.9,
        30.8,
        32,
    ),
    ModelSpec(
        "swin_v2_b",
        "SwinV2-B",
        "Swin Transformer V2",
        "torchvision",
        "torchvision.models.swin_v2_b",
        "Swin_V2_B_Weights.IMAGENET1K_V1 / swin_v2_b-781e5279.pth",
        "ImageNet-1K supervised",
        1000,
        256,
        272,
        "bicubic",
        91.9,
        60.0,
        30.8,
        24,
    ),
    ModelSpec(
        "openclip_b",
        "OpenCLIP-b",
        "Vision Transformer / OpenCLIP",
        "timm",
        "vit_base_patch16_clip_224.laion2b_ft_in1k",
        "timm pretrained weights for vit_base_patch16_clip_224.laion2b_ft_in1k",
        "LAION-2B contrastive pretraining, then ImageNet-1K fine-tuning",
        1000,
        224,
        224,
        "bicubic",
        94.3,
        66.1,
        38.8,
        64,
    ),
    ModelSpec(
        "openclip_h",
        "OpenCLIP-h",
        "Vision Transformer / OpenCLIP",
        "timm",
        "vit_huge_patch14_clip_224.laion2b_ft_in1k",
        "timm pretrained weights for vit_huge_patch14_clip_224.laion2b_ft_in1k",
        "LAION-2B contrastive pretraining, then ImageNet-1K fine-tuning",
        1000,
        224,
        224,
        "bicubic",
        94.9,
        79.0,
        45.5,
        8,
    ),
    ModelSpec(
        "dinov2_b",
        "DINOv2-b",
        "DINOv2 ViT-B/14",
        "facebookresearch/dinov2 PyTorch Hub",
        "dinov2_vitb14_lc (4-layer linear classifier)",
        "dinov2_vitb14_pretrain.pth + dinov2_vitb14_linear4_head.pth",
        "LVD-142M self-supervised backbone + ImageNet-1K linear head",
        1000,
        518,
        None,
        "bicubic",
        93.6,
        74.5,
        44.5,
        32,
    ),
    ModelSpec(
        "dinov2_g",
        "DINOv2-g",
        "DINOv2 ViT-g/14",
        "facebookresearch/dinov2 PyTorch Hub",
        "dinov2_vitg14_lc (4-layer linear classifier)",
        "dinov2_vitg14_pretrain.pth + dinov2_vitg14_linear4_head.pth",
        "LVD-142M self-supervised backbone + ImageNet-1K linear head",
        1000,
        518,
        None,
        "bicubic",
        94.7,
        84.3,
        62.8,
        2,
    ),
)

MODEL_BY_KEY: Mapping[str, ModelSpec] = {spec.key: spec for spec in MODEL_SPECS}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return project_root().parent


def evidence_path() -> Path:
    return project_root() / "evidence" / "model_evidence.json"


def load_evidence() -> Dict[str, Any]:
    with evidence_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_model_keys(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return [spec.key for spec in MODEL_SPECS]
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(keys) - set(MODEL_BY_KEY))
    if unknown:
        raise ValueError(f"Unknown model key(s): {unknown}")
    return keys


def parse_dataset_names(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(DATASET_NAMES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"Unknown dataset name(s): {unknown}")
    return names


def paper_value(spec: ModelSpec, dataset_name: str) -> float:
    if dataset_name == DATASET_IN:
        return spec.paper_in
    if dataset_name == DATASET_AE_ES:
        return spec.paper_ae_es
    if dataset_name == DATASET_AE_DIVERSE:
        return spec.paper_ae_diverse
    raise KeyError(dataset_name)


def iter_specs(keys: Iterable[str]) -> Iterable[ModelSpec]:
    for key in keys:
        yield MODEL_BY_KEY[key]
