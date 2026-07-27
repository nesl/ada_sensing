from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

from .protocol import DINOV2_HUB_REF, MODEL_BY_KEY, ModelSpec


def configure_model_cache(checkpoint_dir: Path) -> Dict[str, str]:
    """Keep every framework cache inside the reproducibility workspace."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch_hub = checkpoint_dir / "torch_hub"
    huggingface = checkpoint_dir / "huggingface"
    torch_hub.mkdir(parents=True, exist_ok=True)
    huggingface.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_hub))
    os.environ["TORCH_HOME"] = str(torch_hub)
    os.environ["HF_HOME"] = str(huggingface)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(huggingface / "hub")
    return {
        "torch_hub_cache": str(torch_hub.resolve()),
        "huggingface_cache": str(huggingface.resolve()),
    }


def _torchvision_model(spec: ModelSpec) -> nn.Module:
    from torchvision import models

    constructors = {
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        ),
        "efficientnet_b3": (
            models.efficientnet_b3,
            models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        ),
        "swin_v2_t": (models.swin_v2_t, models.Swin_V2_T_Weights.IMAGENET1K_V1),
        "swin_v2_s": (models.swin_v2_s, models.Swin_V2_S_Weights.IMAGENET1K_V1),
        "swin_v2_b": (models.swin_v2_b, models.Swin_V2_B_Weights.IMAGENET1K_V1),
    }
    constructor, weights = constructors[spec.key]
    return constructor(weights=weights)


def _torchvision_checkpoint(spec: ModelSpec) -> Path | None:
    filename = spec.checkpoint.rsplit("/", maxsplit=1)[-1].strip()
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    return path if path.is_file() else None


def _deepaugment_model(checkpoint: Path) -> nn.Module:
    from torchvision import models

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"DeepAugment+AugMix checkpoint missing: {checkpoint}. "
            "Run scripts/download_checkpoints.sh first."
        )
    model = models.resnet50(weights=None)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    cleaned = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return model


def _timm_model(spec: ModelSpec) -> nn.Module:
    import timm

    return timm.create_model(spec.identifier, pretrained=True)


def _dinov2_model(spec: ModelSpec, hub_ref: str) -> nn.Module:
    hub_name = "dinov2_vitb14_lc" if spec.key == "dinov2_b" else "dinov2_vitg14_lc"
    repository = f"facebookresearch/dinov2:{hub_ref}"
    return torch.hub.load(repository, hub_name, pretrained=True, trust_repo=True)


def load_model(
    key: str,
    checkpoint_dir: Path,
    dinov2_hub_ref: str = DINOV2_HUB_REF,
) -> Tuple[nn.Module, Dict[str, str]]:
    cache_paths = configure_model_cache(checkpoint_dir)
    spec = MODEL_BY_KEY[key]
    if key == "resnet50_deepaugment_augmix":
        checkpoint = checkpoint_dir / "deepaugment_and_augmix.pth.tar"
        model = _deepaugment_model(checkpoint)
        resolved_checkpoint = str(checkpoint.resolve())
        sha256 = sha256_file(checkpoint)
    elif spec.source == "torchvision":
        model = _torchvision_model(spec)
        checkpoint = _torchvision_checkpoint(spec)
        resolved_checkpoint = str(checkpoint.resolve()) if checkpoint else spec.checkpoint
        sha256 = sha256_file(checkpoint) if checkpoint else "framework-cache-file-not-found"
    elif spec.source == "timm":
        model = _timm_model(spec)
        resolved_checkpoint = spec.checkpoint
        sha256 = "not-resolved-from-framework-cache"
    elif spec.key.startswith("dinov2_"):
        model = _dinov2_model(spec, dinov2_hub_ref)
        resolved_checkpoint = spec.checkpoint
        sha256 = "not-resolved-from-torch-hub-cache"
    else:
        raise KeyError(key)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    provenance = {
        "model_key": key,
        "requested_identifier": spec.identifier,
        "requested_checkpoint": spec.checkpoint,
        "resolved_checkpoint": resolved_checkpoint,
        "checkpoint_sha256": sha256,
        "dinov2_hub_ref": dinov2_hub_ref if key.startswith("dinov2_") else "",
        "torch_version": torch.__version__,
        "torchvision_version": package_version("torchvision"),
        "timm_version": package_version("timm"),
        **cache_paths,
    }
    return model, provenance


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
