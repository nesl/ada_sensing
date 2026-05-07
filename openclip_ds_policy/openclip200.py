from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


DEFAULT_PROMPT_TEMPLATE = "a photo of a {class_name}."


def load_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def parse_class_id(sample_id: str) -> str:
    parts = str(sample_id).split("__")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse sample_id={sample_id}")
    return parts[1]


def parse_group_id(sample_id: str) -> str:
    parts = str(sample_id).split("__")
    if len(parts) >= 3:
        return "__".join(parts[1:])
    return str(sample_id)


def load_split_sample_ids(data_json: str | Path) -> set[str]:
    return {str(record["sample_id"]) for record in load_json(data_json)}


def filter_manifest_items(items: Sequence[Dict[str, Any]], data_json: str | Path | None) -> List[Dict[str, Any]]:
    if data_json is None:
        return list(items)
    allowed = load_split_sample_ids(data_json)
    return [item for item in items if str(item["id"]) in allowed]


def get_subset_label_ids(items: Sequence[Dict[str, Any]]) -> List[int]:
    return sorted({int(item["label"]) for item in items})


def load_class_names(class_index_json: str | Path, label_ids: Sequence[int]) -> List[str]:
    class_index = load_json(class_index_json)
    return [
        str(class_index[str(label_id)][1]).replace("_", " ")
        for label_id in label_ids
    ]


def resolve_ae_path(root: str | Path, sample_id: str, candidate_path: str) -> str:
    parts = str(sample_id).split("__")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse sample_id={sample_id}")
    env, class_id, stem = parts[0], parts[1], "__".join(parts[2:])
    suffix = Path(candidate_path).suffix
    path = (
        Path(root)
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "auto_exposure"
        / env
        / "param_1"
        / class_id
        / f"{stem}{suffix}"
    )
    if not path.exists():
        raise FileNotFoundError(f"AE image not found for sample_id={sample_id}: {path}")
    return str(path)


def build_option_name_map(items: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    option_id_to_name: Dict[int, str] = {}
    for item in items:
        for candidate in item["candidates"]:
            option_id = int(candidate["option_id"])
            option_name = str(candidate.get("meta", {}).get("option_name", ""))
            previous = option_id_to_name.get(option_id)
            if previous is not None and previous != option_name:
                raise ValueError(
                    f"option_id={option_id} maps to both {previous} and {option_name}"
                )
            option_id_to_name[option_id] = option_name
    return option_id_to_name


def summarize_binary_hits(hits: Sequence[int]) -> Dict[str, Any]:
    correct = int(sum(hits))
    total = len(hits)
    return {
        "correct": correct,
        "total": total,
        "acc": 100.0 * correct / max(1, total),
    }


def summarize_float_hits(values: Sequence[float]) -> Dict[str, Any]:
    total = len(values)
    return {
        "mean_acc": 100.0 * float(sum(values)) / max(1, total),
        "num_samples": total,
    }


class OpenCLIP200Classifier:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        class_index_json: str | Path,
        label_ids: Sequence[int],
        device: torch.device,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "Missing open_clip. Install open_clip_torch in the lens env first."
            ) from exc

        self.label_ids = [int(label_id) for label_id in label_ids]
        self.label_to_subset_index = {
            label_id: idx for idx, label_id in enumerate(self.label_ids)
        }
        self.class_names = load_class_names(class_index_json, self.label_ids)
        self.prompt_template = prompt_template
        self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_features = self._build_text_features()

    def _build_text_features(self) -> torch.Tensor:
        prompts = [
            self.prompt_template.format(class_name=class_name)
            for class_name in self.class_names
        ]
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def gt_subset_index(self, raw_label_id: int) -> int:
        return self.label_to_subset_index[int(raw_label_id)]

    def preprocess_images(self, images: Iterable[Any]) -> torch.Tensor:
        return torch.stack([self.preprocess(image) for image in images], dim=0)

    @torch.no_grad()
    def similarity_from_images(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(images.to(self.device, non_blocking=True))
        image_features = F.normalize(image_features, dim=-1)
        return image_features @ self.text_features.t()

    def pred_raw_labels(self, similarity: torch.Tensor) -> torch.Tensor:
        subset_preds = torch.argmax(similarity, dim=-1).detach().cpu()
        label_ids = torch.tensor(self.label_ids, dtype=torch.long)
        return label_ids[subset_preds]
