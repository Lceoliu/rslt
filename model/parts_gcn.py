from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from dataset.transform import NormalizeProcessor
from model.backbones.aagcn_minimal import AAGCNBackbone, build_adjacency_from_numpy
from model.backbones.stgcn_minimal import STGCNBackbone
from model.fusion import ConcatMLPFusion, AttentionFusion


PARTS_DEFAULT = ["body", "face", "left_hand", "right_hand"]


def np_to_tensor_kpts(arr: np.ndarray, drop_conf: bool = True) -> torch.Tensor:
    """Convert (T, V, C=3) numpy to torch [1, C_used, T, V]."""
    assert arr.ndim == 3
    if drop_conf:
        arr = arr[..., :2]  # keep x,y only
    C_used = arr.shape[-1]
    arr = np.transpose(arr, (2, 0, 1))  # (C, T, V)
    return torch.from_numpy(arr.astype("float32")).unsqueeze(0)  # (1, C, T, V)


class MultiPartGCNModel(nn.Module):
    def __init__(
        self,
        parts: Optional[List[str]] = None,
        backbone: str = "aagcn",  # or "stgcn"
        part_embed_dim: int = 256,
        out_embed_dim: int = 512,
        drop_conf: bool = True,
        fusion: str = "concat_mlp",  # or "attention"
    ) -> None:
        super().__init__()
        self.parts = parts or PARTS_DEFAULT
        self.drop_conf = drop_conf

        # Build per-part adjacency via NormalizeProcessor (static)
        proc = NormalizeProcessor()
        A_parts_np = proc.gen_adjacency_matrix(normalize=False, split_part=True)

        self.backbones = nn.ModuleDict()
        in_channels = 2 if drop_conf else 3
        part_dims = []
        for part in self.parts:
            A_np = A_parts_np[part]
            A = build_adjacency_from_numpy(A_np)
            if backbone.lower() == "stgcn":
                self.backbones[part] = STGCNBackbone(in_channels=in_channels, A=A, embed_dim=part_embed_dim)
            else:
                self.backbones[part] = AAGCNBackbone(in_channels=in_channels, A=A, embed_dim=part_embed_dim)
            part_dims.append(part_embed_dim)

        if fusion == "attention":
            self.fusion = AttentionFusion(part_dims, out_embed_dim, d_model=min(256, part_embed_dim))
        else:
            self.fusion = ConcatMLPFusion(part_dims, out_embed_dim, hidden_dim=max(512, out_embed_dim))

    def forward(self, parts_kpts: Dict[str, np.ndarray | torch.Tensor]) -> torch.Tensor:
        feats = []
        for part in self.parts:
            k = parts_kpts[part]
            if isinstance(k, np.ndarray):
                x = np_to_tensor_kpts(k, drop_conf=self.drop_conf)
            else:
                # torch Tensor assumed shape [T, V, C]
                if self.drop_conf and k.shape[-1] >= 3:
                    k = k[..., :2]
                x = k.permute(2, 0, 1).unsqueeze(0).contiguous()
            x = self.backbones[part](x)
            feats.append(x)
        z = self.fusion([f for f in feats])
        return z  # [N=1, out_embed_dim]

