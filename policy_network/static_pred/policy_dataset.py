import json
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from utils import load_image_rgb


class PolicyDataset(Dataset):
    """
    Expected json format: a list of dicts, each containing at least
    - "sample_id"
    - "baseline_path"
    - "best_option_id"

    Example item:
    {
        "sample_id": "l1__n01443537__ILSVRC2012_val_00000994",
        "baseline_path": "/path/to/baseline.jpg",
        "best_option_id": 13,
        ...
    }
    """
    def __init__(self, json_path: str, transform=None):
        with open(json_path, "r") as f:
            self.items: List[Dict[str, Any]] = json.load(f)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        img = load_image_rgb(item["baseline_path"])
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(int(item["best_option_id"]), dtype=torch.long)

        return {
            "image": img,
            "target": target,
            "sample_id": item["sample_id"],
        }