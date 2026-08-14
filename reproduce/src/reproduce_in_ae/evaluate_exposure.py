from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .datasets import (
    DatasetRoots,
    SettingImageFolder,
    build_dataset,
    default_roots,
    paper_transform,
)
from .evaluate import output_indices
from .exposure import (
    EXPOSURE_MODES,
    FIXED_EV_VALUES,
    TARGET_LUMINANCE_VALUES,
    ExposureMetadata,
    ExposureSpec,
    apply_exposure,
    load_luminance_index,
)
from .models import load_model, parameter_count
from .protocol import (
    DATASET_AE_DIVERSE,
    DATASET_AE_ES,
    DINOV2_HUB_REF,
    PAPER_CROP_SIZE,
    PAPER_RESIZE_SIZE,
    iter_specs,
    parse_model_keys,
    workspace_root,
)


AE_DATASETS = (DATASET_AE_ES, DATASET_AE_DIVERSE)
AE_DATASET_TOTALS = {
    DATASET_AE_ES: 10_000,
    DATASET_AE_DIVERSE: 30_000,
}


def parse_args() -> argparse.Namespace:
    defaults = default_roots(workspace_root())
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate digital-exposure sweeps on the two AE datasets."
    )
    parser.add_argument("--models", default="all")
    parser.add_argument(
        "--datasets",
        default="all",
        help="all or comma-separated ae_imagenet_es,ae_imagenet_es_diverse",
    )
    parser.add_argument(
        "--modes",
        default="all",
        help=f"all or comma-separated values from {EXPOSURE_MODES}",
    )
    parser.add_argument("--fixed-ev-values", type=float, nargs="*")
    parser.add_argument("--target-luminance-values", type=float, nargs="*")
    parser.add_argument("--in-root", type=Path, default=defaults.in_root)
    parser.add_argument("--ae-es-root", type=Path, default=defaults.ae_es_root)
    parser.add_argument(
        "--ae-diverse-root", type=Path, default=defaults.ae_diverse_root
    )
    parser.add_argument(
        "--luminance-csv",
        type=Path,
        default=project / "results" / "ae_luminance" / "per_image.csv",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=project / "checkpoints"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project / "results" / "ae_exposure_raw",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2481757)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dinov2-hub-ref", default=DINOV2_HUB_REF)
    return parser.parse_args()


def parse_datasets(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(AE_DATASETS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(AE_DATASETS))
    if unknown:
        raise ValueError(f"Unknown AE dataset(s): {unknown}")
    return names


def parse_modes(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(EXPOSURE_MODES)
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(modes) - set(EXPOSURE_MODES))
    if unknown:
        raise ValueError(f"Unknown exposure mode(s): {unknown}")
    return modes


def build_specs(args: argparse.Namespace) -> list[ExposureSpec]:
    modes = parse_modes(args.modes)
    specs: list[ExposureSpec] = []
    if "fixed_ev" in modes:
        values = (
            args.fixed_ev_values
            if args.fixed_ev_values is not None
            else FIXED_EV_VALUES
        )
        specs.extend(ExposureSpec("fixed_ev", value) for value in values)
    if "target_mean_luminance" in modes:
        values = (
            args.target_luminance_values
            if args.target_luminance_values is not None
            else TARGET_LUMINANCE_VALUES
        )
        specs.extend(
            ExposureSpec("target_mean_luminance", value) for value in values
        )
    tags = [spec.tag for spec in specs]
    if len(tags) != len(set(tags)):
        raise ValueError("Exposure sweep contains duplicate values")
    return specs


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ExposureDataset(Dataset):
    def __init__(
        self,
        name: str,
        base: SettingImageFolder,
        spec: ExposureSpec,
        luminance_index: Mapping[str, Mapping[str, float]],
    ) -> None:
        self.base = base
        self.name = name
        self.spec = spec
        self.luminance_index = luminance_index
        self.transform = paper_transform()
        self.classes = self.base.classes
        self.setting_roots = self.base.setting_roots

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, target, setting, path_string = self.base[index]
        path = str(Path(path_string).resolve())
        try:
            original = self.luminance_index[path]
        except KeyError as error:
            raise KeyError(
                f"{path} is absent from the full-image luminance index"
            ) from error
        adjusted, metadata = apply_exposure(
            image=image,
            spec=self.spec,
            current_mean_luminance=float(original["mean_luminance"]),
            original_metrics=original,
        )
        return self.transform(adjusted), target, setting, path, metadata


def collate_exposure(batch):
    images, targets, settings, paths, metadata = zip(*batch)
    return (
        torch.stack(images),
        torch.tensor(targets, dtype=torch.long),
        list(settings),
        list(paths),
        list(metadata),
    )


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz_dump(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def summarize_float(values: Sequence[float]) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def summarize_exposure(metadata: Sequence[ExposureMetadata]) -> Dict[str, Any]:
    fields = (
        "gain",
        "effective_ev",
        "original_mean_luminance",
        "achieved_mean_luminance",
        "near_black_fraction",
        "near_white_fraction",
        "any_channel_zero_fraction",
        "any_channel_saturated_fraction",
    )
    return {
        field: summarize_float([float(getattr(row, field)) for row in metadata])
        for field in fields
    }


def evaluate_one(
    model: torch.nn.Module,
    model_key: str,
    dataset_name: str,
    dataset: ExposureDataset,
    spec: ExposureSpec,
    device: torch.device,
    batch_size: int,
    workers: int,
    class_index_json: Path,
    max_samples: int,
    prediction_path: Path,
) -> Dict[str, Any]:
    full_size = len(dataset)
    evaluated_dataset: Dataset = dataset
    if max_samples:
        evaluated_dataset = Subset(dataset, range(min(max_samples, full_size)))
    indices = torch.tensor(
        output_indices(dataset.classes, class_index_json),
        dtype=torch.long,
        device=device,
    )
    loader = DataLoader(
        evaluated_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_exposure,
    )
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_hits: list[bool] = []
    all_metadata: list[ExposureMetadata] = []
    setting_correct: Dict[str, int] = defaultdict(int)
    setting_total: Dict[str, int] = defaultdict(int)
    path_digest = hashlib.sha256()
    start = time.time()
    with torch.inference_mode():
        for images, targets, settings, paths, metadata in tqdm(
            loader, desc=f"{model_key}:{dataset_name}:{spec.tag}"
        ):
            images = images.to(device, non_blocking=True)
            targets_device = targets.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim != 2 or logits.shape[1] != 1000:
                raise ValueError(
                    f"{model_key} returned shape {tuple(logits.shape)}; "
                    "expected [N, 1000]"
                )
            predictions = logits.index_select(1, indices).argmax(dim=1)
            hits = predictions.eq(targets_device)
            targets_list = targets.tolist()
            predictions_list = predictions.detach().cpu().tolist()
            hits_list = hits.detach().cpu().tolist()
            all_targets.extend(int(value) for value in targets_list)
            all_predictions.extend(int(value) for value in predictions_list)
            all_hits.extend(bool(value) for value in hits_list)
            all_metadata.extend(metadata)
            for setting, hit in zip(settings, hits_list):
                setting_correct[setting] += int(hit)
                setting_total[setting] += 1
            for path in paths:
                path_digest.update(str(Path(path).resolve()).encode("utf-8"))
                path_digest.update(b"\n")

    targets_array = np.asarray(all_targets, dtype=np.uint16)
    predictions_array = np.asarray(all_predictions, dtype=np.uint16)
    hits_array = np.asarray(all_hits, dtype=np.bool_)
    atomic_npz_dump(
        prediction_path,
        targets=targets_array,
        predictions=predictions_array,
        hits=hits_array,
    )
    correct = int(hits_array.sum())
    total = int(hits_array.size)
    per_setting = {
        setting: {
            "correct": setting_correct[setting],
            "total": setting_total[setting],
            "accuracy": 100.0
            * setting_correct[setting]
            / max(1, setting_total[setting]),
        }
        for setting in sorted(setting_total)
    }
    return {
        "model": model_key,
        "dataset": dataset_name,
        "exposure": {"mode": spec.mode, "value": spec.value, "tag": spec.tag},
        "correct": correct,
        "total": total,
        "full_dataset_total": full_size,
        "micro_accuracy": 100.0 * correct / max(1, total),
        "macro_setting_accuracy": sum(
            row["accuracy"] for row in per_setting.values()
        )
        / max(1, len(per_setting)),
        "per_setting": per_setting,
        "exposure_statistics": summarize_exposure(all_metadata),
        "prediction_file": str(prediction_path.resolve()),
        "prediction_arrays": {
            "targets": "uint16",
            "predictions": "uint16 closed-200-way indices",
            "hits": "bool",
        },
        "path_order_sha256": path_digest.hexdigest(),
        "elapsed_seconds": time.time() - start,
        "is_smoke_test": bool(max_samples),
    }


def result_paths(
    output_dir: Path,
    model_key: str,
    dataset_name: str,
    spec: ExposureSpec,
    max_samples: int,
) -> tuple[Path, Path]:
    smoke = f"__smoke_{max_samples}" if max_samples else ""
    stem = f"{model_key}__{dataset_name}__{spec.tag}{smoke}"
    return output_dir / f"{stem}.json", output_dir / f"{stem}.npz"


def result_is_complete(
    json_path: Path,
    prediction_path: Path,
    expected_total: int,
) -> bool:
    if not json_path.is_file() or not prediction_path.is_file():
        return False
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        with np.load(prediction_path) as arrays:
            total = int(arrays["hits"].size)
            return (
                int(payload["total"]) == total
                and total == expected_total
                and arrays["targets"].size == total
                and arrays["predictions"].size == total
            )
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return False


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; pass --device cpu for a smoke test"
        )
    device = torch.device(args.device)
    roots = DatasetRoots(args.in_root, args.ae_es_root, args.ae_diverse_root)
    model_keys = parse_model_keys(args.models)
    dataset_names = parse_datasets(args.datasets)
    exposure_specs = build_specs(args)
    if not args.luminance_csv.is_file():
        raise FileNotFoundError(
            f"Missing {args.luminance_csv}; run "
            "python -m reproduce_in_ae.analyze_luminance first"
        )
    luminance_index = load_luminance_index(args.luminance_csv)
    class_index_json = (
        workspace_root()
        / "data"
        / "ImageNet-ES-Diverse"
        / "imagenet_class_index.json"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_datasets = {
        dataset_name: build_dataset(
            dataset_name, roots, transform=lambda image: image
        )
        for dataset_name in dataset_names
    }

    for model_spec in iter_specs(model_keys):
        pending: list[tuple[str, ExposureSpec, Path, Path]] = []
        for dataset_name in dataset_names:
            for exposure_spec in exposure_specs:
                json_path, prediction_path = result_paths(
                    args.output_dir,
                    model_spec.key,
                    dataset_name,
                    exposure_spec,
                    args.max_samples,
                )
                if args.overwrite or not result_is_complete(
                    json_path,
                    prediction_path,
                    min(
                        args.max_samples or AE_DATASET_TOTALS[dataset_name],
                        AE_DATASET_TOTALS[dataset_name],
                    ),
                ):
                    pending.append(
                        (dataset_name, exposure_spec, json_path, prediction_path)
                    )
        if not pending:
            print(f"Skip {model_spec.key}: all requested exposure results exist")
            continue

        model, provenance = load_model(
            model_spec.key,
            checkpoint_dir=args.checkpoint_dir,
            dinov2_hub_ref=args.dinov2_hub_ref,
        )
        model = model.to(device)
        provenance["parameter_count"] = str(parameter_count(model))
        provenance["device"] = str(device)
        provenance["platform"] = platform.platform()
        batch_size = args.batch_size or model_spec.recommended_batch_size

        for dataset_name, exposure_spec, json_path, prediction_path in pending:
            dataset = ExposureDataset(
                name=dataset_name,
                base=base_datasets[dataset_name],
                spec=exposure_spec,
                luminance_index=luminance_index,
            )
            result = evaluate_one(
                model=model,
                model_key=model_spec.key,
                dataset_name=dataset_name,
                dataset=dataset,
                spec=exposure_spec,
                device=device,
                batch_size=batch_size,
                workers=args.workers,
                class_index_json=class_index_json,
                max_samples=args.max_samples,
                prediction_path=prediction_path,
            )
            result["model_provenance"] = provenance
            result["dataset_roots"] = [
                {"setting": setting, "root": str(root.resolve())}
                for setting, root in dataset.setting_roots
            ]
            result["protocol"] = {
                "digital_exposure": (
                    "full decoded JPEG -> inverse sRGB -> uniform linear RGB gain "
                    "-> clip [0,1] -> sRGB uint8; fixed 0 EV bypasses conversion"
                ),
                "target_gain": (
                    "target_mean_luminance / max(original_full_image_mean_Y, 1e-6)"
                ),
                "luminance_definition": (
                    "linear Rec.709 Y=0.2126R+0.7152G+0.0722B"
                ),
                "model_resize": PAPER_RESIZE_SIZE,
                "model_crop": PAPER_CROP_SIZE,
                "model_preprocessing": (
                    "PIL bilinear short-edge resize 256, center crop 224, "
                    "ToTensor, ImageNet mean/std normalization"
                ),
                "evaluation_label_space": (
                    "closed 200-way Tiny-ImageNet subset; slice 1000 logits "
                    "to sorted dataset WNIDs before argmax"
                ),
                "weights_frozen": True,
                "fp32": True,
                "batch_size": batch_size,
                "workers": args.workers,
                "seed": args.seed,
                "luminance_csv": str(args.luminance_csv.resolve()),
            }
            atomic_json_dump(result, json_path)
            print(
                f"{model_spec.key} {dataset_name} {exposure_spec.tag}: "
                f"{result['micro_accuracy']:.4f}% -> {json_path}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
