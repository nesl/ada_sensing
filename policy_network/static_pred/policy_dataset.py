import json
import random
from typing import Any, Dict, List, Optional

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

    When `manifest_path` is provided, the dataset can also sample the input image
    from all candidates of the same physical scene instead of always using the
    fixed baseline image.
    """
    def __init__(
        self,
        json_path: str,
        transform=None,
        manifest_path: Optional[str] = None,
        input_sampling: str = "baseline",
        fixed_option_id: Optional[int] = None,
    ):
        with open(json_path, "r") as f:
            self.items: List[Dict[str, Any]] = json.load(f)

        self.transform = transform
        self.input_sampling = input_sampling
        self.fixed_option_id = fixed_option_id
        self.manifest_by_id: Optional[Dict[str, Dict[str, Any]]] = None

        if self.input_sampling not in {"baseline", "random_candidate", "fixed_option"}:
            raise ValueError(
                f"Unsupported input_sampling={input_sampling}. "
                "Expected 'baseline', 'random_candidate', or 'fixed_option'."
            )

        if manifest_path is not None:
            with open(manifest_path, "r") as f:
                manifest_items: List[Dict[str, Any]] = json.load(f)
            self.manifest_by_id = {
                str(item["id"]): item for item in manifest_items
            }

        if self.input_sampling in {"random_candidate", "fixed_option"} and self.manifest_by_id is None:
            raise ValueError(
                "manifest_path is required when input_sampling uses manifest candidates."
            )

        if self.input_sampling == "fixed_option" and self.fixed_option_id is None:
            raise ValueError("fixed_option_id is required when input_sampling='fixed_option'.")

        self.has_soft_targets = any("soft_target" in item for item in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_input_path(self, item: Dict[str, Any]) -> str:
        if self.input_sampling == "baseline":
            return item["baseline_path"]

        manifest_item = self.manifest_by_id[str(item["sample_id"])]
        candidates = manifest_item["candidates"]
        if self.input_sampling == "random_candidate":
            chosen_candidate = random.choice(candidates)
            return chosen_candidate["path"]

        for candidate in candidates:
            if int(candidate["option_id"]) == int(self.fixed_option_id):
                return candidate["path"]

        raise KeyError(
            f"sample_id={item['sample_id']} does not contain option_id={self.fixed_option_id}"
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        image_path = self._resolve_input_path(item)
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
