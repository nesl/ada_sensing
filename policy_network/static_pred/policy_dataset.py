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
    def __init__(
        self,
        json_path: str,
        transform=None,
    ):
        with open(json_path, "r") as f:
            self.items: List[Dict[str, Any]] = json.load(f)

        self.transform = transform
        self.has_soft_targets = any("soft_target" in item for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        image_path = item["baseline_path"]
        img = load_image_rgb(image_path)
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(int(item["best_option_id"]), dtype=torch.long)
        record = {
            "image": img,
            "target": target,
            "sample_id": item["sample_id"],
            "input_path": image_path,
        }
        if "soft_target" in item:
            record["soft_target"] = torch.tensor(item["soft_target"], dtype=torch.float32)
        return record
