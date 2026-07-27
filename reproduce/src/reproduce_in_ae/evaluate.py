from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .datasets import (
    DatasetRoots,
    build_dataset,
    collate_settings,
    default_roots,
)
from .models import load_model, parameter_count
from .protocol import (
    DATASET_NAMES,
    DINOV2_HUB_REF,
    MODEL_BY_KEY,
    PAPER_CROP_SIZE,
    PAPER_RESIZE_SIZE,
    iter_specs,
    paper_value,
    parse_dataset_names,
    parse_model_keys,
    workspace_root,
)


def parse_args() -> argparse.Namespace:
    defaults = default_roots(workspace_root())
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Reproduce Table 1 IN and AE top-1 accuracies."
    )
    parser.add_argument("--models", default="all", help="all or comma-separated model keys")
    parser.add_argument(
        "--datasets",
        default="all",
        help=f"all or comma-separated values from {DATASET_NAMES}",
    )
    parser.add_argument("--in-root", type=Path, default=defaults.in_root)
    parser.add_argument("--ae-es-root", type=Path, default=defaults.ae_es_root)
    parser.add_argument("--ae-diverse-root", type=Path, default=defaults.ae_diverse_root)
    parser.add_argument("--checkpoint-dir", type=Path, default=project / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=project / "results" / "raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 uses the per-model conservative batch size from the registry.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2481757)
    parser.add_argument("--max-samples", type=int, default=0, help="Smoke-test only")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based deterministic sample shard index.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of interleaved sample shards (1 disables sharding).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dinov2-hub-ref", default=DINOV2_HUB_REF)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def output_indices(classes: Iterable[str], class_index_json: Path) -> list[int]:
    with class_index_json.open("r", encoding="utf-8") as handle:
        class_index = json.load(handle)
    wnid_to_index = {entry[0]: int(index) for index, entry in class_index.items()}
    missing = sorted(set(classes) - set(wnid_to_index))
    if missing:
        raise ValueError(f"Dataset classes absent from ImageNet class index: {missing}")
    return [wnid_to_index[wnid] for wnid in classes]


def evaluate_dataset(
    model: torch.nn.Module,
    dataset,
    model_key: str,
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    class_index_json: Path,
    max_samples: int,
    shard_index: int = 0,
    shard_count: int = 1,
) -> Dict[str, Any]:
    full_size = len(dataset)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"Invalid shard {shard_index}/{shard_count}; require "
            "shard_count >= 1 and 0 <= shard_index < shard_count"
        )
    evaluated_dataset = (
        dataset
        if shard_count == 1
        else Subset(dataset, range(shard_index, full_size, shard_count))
    )
    if max_samples:
        evaluated_dataset = Subset(
            evaluated_dataset, range(min(max_samples, len(evaluated_dataset)))
        )

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
        collate_fn=collate_settings,
    )
    overall_correct = 0
    overall_total = 0
    setting_correct: Dict[str, int] = defaultdict(int)
    setting_total: Dict[str, int] = defaultdict(int)
    start = time.time()

    with torch.inference_mode():
        for images, targets, settings, _paths in tqdm(
            loader, desc=f"{model_key}:{dataset_name}"
        ):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.ndim != 2 or logits.shape[1] != 1000:
                raise ValueError(
                    f"{model_key} returned shape {tuple(logits.shape)}; expected [N, 1000]"
                )
            closed_logits = logits.index_select(1, indices)
            predictions = closed_logits.argmax(dim=1)
            hits = predictions.eq(targets).detach().cpu().tolist()
            overall_correct += int(sum(hits))
            overall_total += len(hits)
            for setting, hit in zip(settings, hits):
                setting_correct[setting] += int(hit)
                setting_total[setting] += 1

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
    macro_accuracy = sum(row["accuracy"] for row in per_setting.values()) / max(
        1, len(per_setting)
    )
    micro_accuracy = 100.0 * overall_correct / max(1, overall_total)
    return {
        "model": model_key,
        "dataset": dataset_name,
        "dataset_roots": [
            {"setting": setting, "root": str(root.resolve())}
            for setting, root in dataset.setting_roots
        ],
        "correct": overall_correct,
        "total": overall_total,
        "full_dataset_total": full_size,
        "micro_accuracy": micro_accuracy,
        "macro_setting_accuracy": macro_accuracy,
        "paper_value": paper_value(MODEL_BY_KEY[model_key], dataset_name),
        "paper_rounding_match": round(micro_accuracy, 1)
        == paper_value(MODEL_BY_KEY[model_key], dataset_name),
        "evaluation_label_space": (
            "closed 200-way Tiny-ImageNet subset; slice 1000 logits to the "
            "200 sorted dataset WNIDs before argmax"
        ),
        "aggregation": (
            "micro over all images; macro over equal-sized environment/AE-shot "
            "settings is also reported"
        ),
        "per_setting": per_setting,
        "elapsed_seconds": time.time() - start,
        "is_smoke_test": bool(max_samples),
        "shard": {"index": shard_index, "count": shard_count},
    }


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            f"Invalid shard {args.shard_index}/{args.shard_count}; require "
            "shard_count >= 1 and 0 <= shard_index < shard_count"
        )
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Restore the NVIDIA driver or pass "
            "--device cpu for small smoke tests."
        )
    device = torch.device(args.device)
    roots = DatasetRoots(args.in_root, args.ae_es_root, args.ae_diverse_root)
    model_keys = parse_model_keys(args.models)
    dataset_names = parse_dataset_names(args.datasets)
    class_index_json = (
        workspace_root() / "data" / "ImageNet-ES-Diverse" / "imagenet_class_index.json"
    )

    for spec in iter_specs(model_keys):
        def result_path(name: str) -> Path:
            shard_suffix = (
                f"__shard_{args.shard_index}_of_{args.shard_count}"
                if args.shard_count > 1
                else ""
            )
            smoke_suffix = f"__smoke_{args.max_samples}" if args.max_samples else ""
            suffix = shard_suffix + smoke_suffix
            return args.output_dir / f"{spec.key}__{name}{suffix}.json"

        pending = [
            name
            for name in dataset_names
            if args.overwrite
            or not result_path(name).exists()
        ]
        if not pending:
            print(f"Skip {spec.key}: all requested result files exist")
            continue
        model, provenance = load_model(
            spec.key,
            checkpoint_dir=args.checkpoint_dir,
            dinov2_hub_ref=args.dinov2_hub_ref,
        )
        model = model.to(device)
        provenance["parameter_count"] = str(parameter_count(model))
        provenance["device"] = str(device)
        provenance["platform"] = platform.platform()
        batch_size = args.batch_size or spec.recommended_batch_size

        for dataset_name in pending:
            dataset = build_dataset(dataset_name, roots)
            result = evaluate_dataset(
                model=model,
                dataset=dataset,
                model_key=spec.key,
                dataset_name=dataset_name,
                device=device,
                batch_size=batch_size,
                workers=args.workers,
                class_index_json=class_index_json,
                max_samples=args.max_samples,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
            result["model_provenance"] = provenance
            result["protocol"] = {
                "resize": PAPER_RESIZE_SIZE,
                "crop": PAPER_CROP_SIZE,
                "aspect_ratio": "preserve on short-edge resize, then center crop",
                "interpolation": "PIL bilinear",
                "pixel_pipeline": (
                    "RGB uint8 -> ToTensor [0,1] -> Normalize "
                    "mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)"
                ),
                "weights_frozen": True,
                "target_domain_adaptation": False,
                "batch_size": batch_size,
                "workers": args.workers,
                "seed": args.seed,
                "expected_setting_counts": {
                    key: row["total"] for key, row in result["per_setting"].items()
                },
            }
            output_path = result_path(dataset_name)
            atomic_json_dump(result, output_path)
            print(
                f"{spec.paper_name} {dataset_name}: "
                f"{result['micro_accuracy']:.4f}% "
                f"(paper {result['paper_value']:.1f}%) -> {output_path}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
