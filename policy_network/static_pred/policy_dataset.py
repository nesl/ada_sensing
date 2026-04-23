import hashlib
import json
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
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
        manifest_path: Optional[str] = None,
        input_mode: str = "single",
        env_option_id: Optional[int] = None,
        input_variant: str = "real",
        noise_seed: int = 0,
    ):
        with open(json_path, "r") as f:
            self.items: List[Dict[str, Any]] = json.load(f)

        self.transform = transform
        self.has_soft_targets = any("soft_target" in item for item in self.items)
        self.input_mode = input_mode
        self.env_option_id = env_option_id
        self.input_variant = input_variant
        self.noise_seed = noise_seed
        self.env_path_by_sample_id: Dict[str, str] = {}

        if self.input_mode not in {"single", "dual"}:
            raise ValueError(f"Unsupported input_mode={self.input_mode}")
        if self.input_variant not in {"real", "random_noise_per_sample"}:
            raise ValueError(f"Unsupported input_variant={self.input_variant}")

        if self.input_mode == "dual":
            if manifest_path is None:
                raise ValueError("manifest_path is required when input_mode='dual'.")
            if env_option_id is None:
                raise ValueError("env_option_id is required when input_mode='dual'.")
            with open(manifest_path, "r") as f:
                manifest_items: List[Dict[str, Any]] = json.load(f)
            for manifest_item in manifest_items:
                sample_id = str(manifest_item["id"])
                env_path = None
                for candidate in manifest_item["candidates"]:
                    if int(candidate["option_id"]) == int(env_option_id):
                        env_path = candidate["path"]
                        break
                if env_path is None:
                    raise KeyError(
                        f"Could not find option_id={env_option_id} for sample_id={sample_id}."
                    )
                self.env_path_by_sample_id[sample_id] = env_path

    def _seed_for_sample(self, sample_id: Any) -> int:
        seed_material = f"{self.noise_seed}:{sample_id}".encode("utf-8")
        digest = hashlib.sha256(seed_material).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _make_random_noise_image(self, sample_id: Any, image: Image.Image) -> Image.Image:
        width, height = image.size
        generator = torch.Generator()
        #generator.manual_seed(self._seed_for_sample(sample_id))
        noise = torch.randint(
            low=0,
            high=256,
            size=(height, width, 3),
            generator=generator,
            dtype=torch.uint8,
        )
        return Image.fromarray(noise.numpy(), mode="RGB")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        image_path = item["baseline_path"]
        img = load_image_rgb(image_path)
        if self.input_variant == "random_noise_per_sample":
            img = self._make_random_noise_image(item["sample_id"], img)
        if self.transform is not None:
            img = self.transform(img)

        input_paths: List[str] = [image_path]
        if self.input_mode == "dual":
            sample_id = str(item["sample_id"])
            env_image_path = self.env_path_by_sample_id[sample_id]
            env_img = load_image_rgb(env_image_path)
            if self.transform is not None:
                env_img = self.transform(env_img)
            image_tensor = torch.stack([img, env_img], dim=0)
            input_paths.append(env_image_path)
        else:
            image_tensor = img

        target = torch.tensor(int(item["best_option_id"]), dtype=torch.long)
        record = {
            "image": image_tensor,
            "target": target,
            "sample_id": item["sample_id"],
            "input_path": image_path,
            "input_paths": input_paths,
            "input_mode": self.input_mode,
        }
        if "soft_target" in item:
            record["soft_target"] = torch.tensor(item["soft_target"], dtype=torch.float32)
        return record
