from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .protocol import workspace_root


LIGHT_IDS = ("b010", "b200", "b500", "b700", "b1000")
AE_SHOTS = (1, 2, 3)
REQUIRED_LABEL_FIELDS = (
    "sample_id",
    "original_path",
    "wnid",
    "source_relative_path",
)


def default_labels_csv() -> Path:
    return workspace_root() / "data" / "replication" / "manual_dataset" / "labels.csv"


def default_dpi600_roi_root() -> Path:
    return workspace_root() / "data" / "replication" / "dpi600_roi"


def default_replicated_roi_root() -> Path:
    return workspace_root() / "data" / "replication" / "replicated_capture_roi"


def default_imagenet_es_auto_root() -> Path:
    return (
        workspace_root()
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "auto_exposure"
    )


def default_output_dir() -> Path:
    return workspace_root() / "replicate_result" / "comparison" / "ae_visual"


@dataclass(frozen=True)
class ExportedImage:
    source: Path
    output: Path


@dataclass(frozen=True)
class ExportResult:
    sample_id: str
    light_id: str
    ae_shot: int
    reference_light_id: str
    images: tuple[ExportedImage, ...]


def load_sample(labels_csv: Path, sample_id: str) -> Mapping[str, str]:
    labels_csv = Path(labels_csv).resolve()
    if not labels_csv.is_file():
        raise FileNotFoundError(f"labels CSV does not exist: {labels_csv}")

    with labels_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            field
            for field in REQUIRED_LABEL_FIELDS
            if field not in (reader.fieldnames or ())
        ]
        if missing:
            raise ValueError(f"labels CSV is missing required fields: {missing}")
        matches = [dict(row) for row in reader if row.get("sample_id") == sample_id]

    if not matches:
        raise ValueError(f"sample_id is not present in labels CSV: {sample_id}")
    if len(matches) != 1:
        raise ValueError(f"labels CSV contains duplicate sample_id: {sample_id}")
    return matches[0]


def _export_plan(
    metadata: Mapping[str, str],
    *,
    labels_csv: Path,
    dpi600_roi_root: Path,
    replicated_roi_root: Path,
    imagenet_es_auto_root: Path,
    output_dir: Path,
    light_id: str,
    ae_shot: int,
    reference_light_id: str,
) -> tuple[ExportedImage, ...]:
    sample_id = metadata["sample_id"]
    shot_id = f"ae_{ae_shot:02d}"
    reference_parameter = f"param_{ae_shot}"
    reference_relative = (
        Path(metadata["wnid"]) / Path(metadata["source_relative_path"]).name
    )

    def capture_path(root: Path, zoom_id: str) -> Path:
        return root / sample_id / zoom_id / light_id / "ae" / f"{shot_id}.jpg"

    sources_and_names = (
        (
            labels_csv.parent / metadata["original_path"],
            "1_original_imagenet_no_resize.JPEG",
        ),
        (
            capture_path(dpi600_roi_root, "z001"),
            f"2_dpi600_roi_zoom1_{light_id}_{shot_id}.jpg",
        ),
        (
            capture_path(dpi600_roi_root, "z002"),
            f"3_dpi600_roi_zoom2_{light_id}_{shot_id}.jpg",
        ),
        (
            capture_path(replicated_roi_root, "z001"),
            f"4_replicated_roi_zoom1_{light_id}_{shot_id}.jpg",
        ),
        (
            capture_path(replicated_roi_root, "z002"),
            f"5_replicated_roi_zoom2_{light_id}_{shot_id}.jpg",
        ),
        (
            imagenet_es_auto_root
            / reference_light_id
            / reference_parameter
            / reference_relative,
            f"6_imagenet_es_diverse_test_{reference_light_id}_{shot_id}.JPEG",
        ),
    )
    return tuple(
        ExportedImage(source=Path(source).resolve(), output=output_dir / name)
        for source, name in sources_and_names
    )


def export_ae_visual(
    sample_id: str,
    output_dir: Path,
    *,
    labels_csv: Path | None = None,
    dpi600_roi_root: Path | None = None,
    replicated_roi_root: Path | None = None,
    imagenet_es_auto_root: Path | None = None,
    light_id: str = "b200",
    ae_shot: int = 2,
    reference_light_id: str = "l1",
    overwrite: bool = False,
) -> ExportResult:
    if light_id not in LIGHT_IDS:
        raise ValueError(f"light_id must be one of {LIGHT_IDS}, got {light_id!r}")
    if ae_shot not in AE_SHOTS:
        raise ValueError(f"ae_shot must be one of {AE_SHOTS}, got {ae_shot!r}")
    if (
        not reference_light_id
        or reference_light_id in {".", ".."}
        or "/" in reference_light_id
        or "\\" in reference_light_id
    ):
        raise ValueError("reference_light_id must be a single non-empty path component")

    labels_path = Path(labels_csv or default_labels_csv()).resolve()
    output_path = Path(output_dir).resolve()
    metadata = load_sample(labels_path, sample_id)
    images = _export_plan(
        metadata,
        labels_csv=labels_path,
        dpi600_roi_root=Path(dpi600_roi_root or default_dpi600_roi_root()).resolve(),
        replicated_roi_root=Path(
            replicated_roi_root or default_replicated_roi_root()
        ).resolve(),
        imagenet_es_auto_root=Path(
            imagenet_es_auto_root or default_imagenet_es_auto_root()
        ).resolve(),
        output_dir=output_path,
        light_id=light_id,
        ae_shot=ae_shot,
        reference_light_id=reference_light_id,
    )

    missing_sources = [image.source for image in images if not image.source.is_file()]
    if missing_sources:
        formatted = "\n".join(f"  - {path}" for path in missing_sources)
        raise FileNotFoundError(
            f"The six-image comparison is incomplete for {sample_id}:\n{formatted}"
        )

    collisions = [image.output for image in images if image.output.exists()]
    if collisions and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(f"Output already exists; pass --overwrite:\n{formatted}")

    output_path.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for image in images:
            handle = tempfile.NamedTemporaryFile(
                prefix=f".{image.output.stem}-",
                suffix=image.output.suffix,
                dir=output_path,
                delete=False,
            )
            temporary = Path(handle.name)
            handle.close()
            shutil.copyfile(image.source, temporary)
            temporary_paths.append((temporary, image.output))
        for temporary, target in temporary_paths:
            os.replace(temporary, target)
            target.chmod(0o644)
    finally:
        for temporary, _target in temporary_paths:
            if temporary.exists():
                temporary.unlink()

    return ExportResult(
        sample_id=sample_id,
        light_id=light_id,
        ae_shot=ae_shot,
        reference_light_id=reference_light_id,
        images=images,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export six byte-identical images for an AE visual comparison: clean "
            "ImageNet, two 600-PPI recapture ROIs, two fixed-width recapture ROIs, "
            "and the corresponding ImageNet-ES-Diverse AE image."
        )
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--light-id", choices=LIGHT_IDS, default="b200")
    parser.add_argument("--ae-shot", type=int, choices=AE_SHOTS, default=2)
    parser.add_argument("--reference-light-id", default="l1")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--labels-csv", type=Path, default=default_labels_csv())
    parser.add_argument("--dpi600-roi-root", type=Path, default=default_dpi600_roi_root())
    parser.add_argument(
        "--replicated-roi-root", type=Path, default=default_replicated_roi_root()
    )
    parser.add_argument(
        "--imagenet-es-auto-root", type=Path, default=default_imagenet_es_auto_root()
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = export_ae_visual(
        args.sample_id,
        args.output_dir,
        labels_csv=args.labels_csv,
        dpi600_roi_root=args.dpi600_roi_root,
        replicated_roi_root=args.replicated_roi_root,
        imagenet_es_auto_root=args.imagenet_es_auto_root,
        light_id=args.light_id,
        ae_shot=args.ae_shot,
        reference_light_id=args.reference_light_id,
        overwrite=args.overwrite,
    )
    print(
        f"Exported {len(result.images)} images for {result.sample_id} "
        f"({result.light_id}, ae_{result.ae_shot:02d}, {result.reference_light_id})"
    )
    for image in result.images:
        print(f"  {image.output} <- {image.source}")


if __name__ == "__main__":
    main()
