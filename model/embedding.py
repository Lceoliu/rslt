from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch

from .parts_gcn import PARTS_DEFAULT
from .visual_encoder import VisualEncoder

__all__ = ["build_visual_encoder", "compute_embedding_from_parts"]


def _ensure_parts(parts: Sequence[str]) -> Sequence[str]:
    if "fullbody" in parts:
        return parts
    ordered = list(parts) + ["fullbody"]
    deduped: list[str] = []
    seen = set()
    for name in ordered:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def build_visual_encoder(cfg: Dict[str, Any], llm_dim: int) -> VisualEncoder:
    mcfg = cfg.get("model", {})
    parts = _ensure_parts(mcfg.get("parts", PARTS_DEFAULT))
    uni_cfg = mcfg.get("uni_gcn", {})
    sampling_stride = int(mcfg.get("temporal_sampling_stride", 2))
    tokens_per_chunk = int(mcfg.get("tokens_per_chunk", 0))

    encoder = VisualEncoder(
        parts=parts,
        drop_conf=bool(mcfg.get("drop_conf", True)),
        gcn_embed_dim=int(mcfg.get("part_embed_dim", 256)),
        gcn_proj_dim=int(uni_cfg.get("proj_dim", 64)),
        gcn_temporal_kernel=int(uni_cfg.get("temporal_kernel", 5)),
        gcn_adaptive=bool(uni_cfg.get("adaptive", True)),
        gcn_dropout=float(uni_cfg.get("dropout", 0.0)),
        tokens_per_chunk=tokens_per_chunk,
        llm_dim=int(llm_dim),
        sampling_stride=sampling_stride,
    )
    return encoder


@torch.no_grad()
def compute_embedding_from_parts(parts: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> np.ndarray:
    raise NotImplementedError(
        "Legacy embedding API is not supported with the new pipeline."
    )
