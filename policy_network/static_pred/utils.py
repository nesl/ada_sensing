# data_preprocess.py
"""
Data loading and preprocessing utilities for the Manifest Lens project.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

import timm
from torchvision import transforms

# # class ManifestLensDataset(Dataset):
#     """
#     Returns one sample with all its candidates (paths + metadata) and GT label.
#     The selection (CSA) happens in eval_lens.py.
#     """
#     def __init__(self, manifest_path: str):
#         with open(manifest_path, "r") as f:
#             self.items: List[Dict[str, Any]] = json.load(f)

#     def __len__(self) -> int:
#         return len(self.items)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         return self.items[idx]

def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_timm_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)
    return model

def imagenet_preprocess(image_size: int = 224):
    # Standard ImageNet eval preprocessing
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
