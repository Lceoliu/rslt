from typing import Dict, Any

import numpy as np
import torch

from .parts_gcn import MultiPartGCNModel, PARTS_DEFAULT


def build_model_from_config(cfg: Dict[str, Any]) -> MultiPartGCNModel:
    mcfg = cfg.get("model", {})
    parts = mcfg.get("parts", PARTS_DEFAULT)
    # Ensure fullbody is included
    if 'fullbody' not in parts:
        parts = list(dict.fromkeys(list(parts) + ['fullbody']))
    backbone = mcfg.get("backbone", "aagcn")
    part_embed_dim = int(mcfg.get("part_embed_dim", 256))
    out_embed_dim = int(mcfg.get("embed_dim", 512))
    drop_conf = bool(mcfg.get("drop_conf", True))
    fusion = mcfg.get("fusion", "concat_mlp")
    # Preset length for per-chunk tokens: prefer llm.num_prefix_tokens if available
    llm_cfg = cfg.get("llm", {})
    preset_len = int(mcfg.get("preset_len", llm_cfg.get("num_prefix_tokens", 1)))
    return MultiPartGCNModel(
        parts=parts,
        backbone=backbone,
        part_embed_dim=part_embed_dim,
        out_embed_dim=out_embed_dim,
        drop_conf=drop_conf,
        fusion=fusion,
        preset_len=preset_len,
    )


@torch.no_grad()
def compute_embedding_from_parts(parts: Dict[str, np.ndarray], cfg: Dict[str, Any]) -> np.ndarray:
    model = build_model_from_config(cfg)
    model.eval()
    z = model(parts)  # [1, D]
    return z.cpu().numpy()
