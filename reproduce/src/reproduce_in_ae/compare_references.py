from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

from .protocol import workspace_root

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(root: Path) -> Dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def class_hashes(file_manifest: Dict[str, str]) -> Dict[str, list[str]]:
    grouped: Dict[str, list[str]] = defaultdict(list)
    for relative_path, digest in file_manifest.items():
        class_name = relative_path.split("/", maxsplit=1)[0]
        grouped[class_name].append(digest)
    return {name: sorted(hashes) for name, hashes in sorted(grouped.items())}


def compare(original: Path, diverse: Path) -> dict:
    original_manifest = manifest(original)
    diverse_manifest = manifest(diverse)
    original_keys = set(original_manifest)
    diverse_keys = set(diverse_manifest)
    common = original_keys & diverse_keys
    changed = sorted(
        key for key in common if original_manifest[key] != diverse_manifest[key]
    )
    original_by_class = class_hashes(original_manifest)
    diverse_by_class = class_hashes(diverse_manifest)
    return {
        "original_root": str(original.resolve()),
        "diverse_root": str(diverse.resolve()),
        "original_files": len(original_manifest),
        "diverse_files": len(diverse_manifest),
        "same_relative_paths": original_keys == diverse_keys,
        "same_bytes_at_relative_paths": (
            original_keys == diverse_keys and not changed
        ),
        "same_content_multiset_per_class": original_by_class == diverse_by_class,
        "only_original_paths": sorted(original_keys - diverse_keys),
        "only_diverse_paths": sorted(diverse_keys - original_keys),
        "changed_paths": changed,
    }


def main() -> None:
    workspace = workspace_root()
    parser = argparse.ArgumentParser(
        description="Hash-compare original and Diverse clean reference sets."
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=workspace
        / "data"
        / "ImageNet-ES"
        / "es-test"
        / "sampled_tin_no_resize2",
    )
    parser.add_argument(
        "--diverse",
        type=Path,
        default=workspace
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "sampled_tin_no_resize2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "evidence"
        / "reference_comparison.json",
    )
    args = parser.parse_args()
    result = compare(args.original, args.diverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
