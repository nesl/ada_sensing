# lens_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch
import torch.nn.functional as F

@dataclass
class Candidate:
    option_id: int
    path: str
    meta: Dict[str, Any]

def visit_confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    VisiT quality indicator in the paper: model confidence = max softmax(logits).
    logits: [B, C]
    returns: [B] in [0,1]
    """
    probs = F.softmax(logits, dim=-1)
    conf, _ = probs.max(dim=-1)
    return conf

@torch.no_grad()
def lens_select_best(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> Tuple[int, float, torch.Tensor]:
    """
    images: [K, 3, H, W] candidates for ONE sample (already preprocessed)
    returns: (best_idx, best_conf, best_logits)
    """
    model.eval()
    images = images.to(device, non_blocking=True)
    logits = model(images)  # [K, C]
    conf = visit_confidence_from_logits(logits)  # [K]
    best_idx = int(torch.argmax(conf).item())
    return best_idx, float(conf[best_idx].item()), logits[best_idx]
