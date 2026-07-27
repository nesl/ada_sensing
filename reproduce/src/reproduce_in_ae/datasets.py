from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from .protocol import (
    DATASET_AE_DIVERSE,
    DATASET_AE_ES,
    DATASET_IN,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PAPER_CROP_SIZE,
    PAPER_RESIZE_SIZE,
)


@dataclass(frozen=True)
class DatasetRoots:
    in_root: Path
    ae_es_root: Path
    ae_diverse_root: Path


def paper_transform() -> Compose:
    """Exact transform hard-coded by the official ImageNet-ES evaluator."""
    return Compose(
        [
            Resize(PAPER_RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(PAPER_CROP_SIZE),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def rgb_loader(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class SettingImageFolder(Dataset):
    def __init__(
        self,
        setting_roots: Sequence[Tuple[str, Path]],
        transform: Callable,
    ) -> None:
        if not setting_roots:
            raise ValueError("No dataset settings were supplied")
        self.setting_roots = [(setting, Path(root)) for setting, root in setting_roots]
        self.datasets: List[Tuple[str, ImageFolder]] = []
        self.offsets: List[int] = []
        total = 0
        expected_classes: List[str] | None = None
        for setting, root in setting_roots:
            dataset = ImageFolder(root=str(root), transform=transform, loader=rgb_loader)
            if expected_classes is None:
                expected_classes = dataset.classes
            elif dataset.classes != expected_classes:
                raise ValueError(
                    f"Class mismatch in {root}: expected {expected_classes[:3]}..., "
                    f"found {dataset.classes[:3]}..."
                )
            self.offsets.append(total)
            self.datasets.append((setting, dataset))
            total += len(dataset)
        self.classes = expected_classes or []
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        for dataset_index in range(len(self.datasets) - 1, -1, -1):
            offset = self.offsets[dataset_index]
            if index >= offset:
                setting, dataset = self.datasets[dataset_index]
                image, target = dataset[index - offset]
                path = dataset.samples[index - offset][0]
                return image, target, setting, path
        raise IndexError(index)


def _ae_settings(root: Path) -> List[Tuple[str, Path]]:
    settings: List[Tuple[str, Path]] = []
    if not root.is_dir():
        raise FileNotFoundError(f"AE root does not exist: {root}")
    for environment in sorted(path for path in root.iterdir() if path.is_dir()):
        for shot in sorted(path for path in environment.iterdir() if path.is_dir()):
            settings.append((f"{environment.name}/{shot.name}", shot))
    return settings


def build_dataset(
    name: str,
    roots: DatasetRoots,
    transform: Callable | None = None,
) -> SettingImageFolder:
    transform = transform or paper_transform()
    if name == DATASET_IN:
        return SettingImageFolder([("in/reference", roots.in_root)], transform)
    if name == DATASET_AE_ES:
        return SettingImageFolder(_ae_settings(roots.ae_es_root), transform)
    if name == DATASET_AE_DIVERSE:
        return SettingImageFolder(_ae_settings(roots.ae_diverse_root), transform)
    raise KeyError(name)


def default_roots(workspace: Path) -> DatasetRoots:
    return DatasetRoots(
        in_root=workspace
        / "data"
        / "ImageNet-ES"
        / "es-test"
        / "sampled_tin_no_resize2",
        ae_es_root=workspace
        / "data"
        / "ImageNet-ES"
        / "es-test"
        / "auto_exposure",
        ae_diverse_root=workspace
        / "data"
        / "ImageNet-ES-Diverse"
        / "es-diverse-test"
        / "auto_exposure",
    )


def collate_settings(batch):
    images, targets, settings, paths = zip(*batch)
    import torch

    return torch.stack(images), torch.tensor(targets), list(settings), list(paths)


def setting_counts(dataset: SettingImageFolder) -> Dict[str, int]:
    return {setting: len(child) for setting, child in dataset.datasets}
