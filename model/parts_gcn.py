from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..dataset.transform import NormalizeProcessor
from .backbones.aagcn_minimal import AAGCNBackbone, build_adjacency_from_numpy
from .backbones.stgcn_minimal import STGCNBackbone
from .fusion import ConcatMLPFusion, AttentionFusion


PARTS_DEFAULT = ["body", "face", "left_hand", "right_hand", "fullbody"]


def _np_or_torch_to_nctv(
    x: np.ndarray | torch.Tensor, drop_conf: bool, device: torch.device
) -> torch.Tensor:
    """Accept numpy/torch (T,V,C) or (B,T,V,C) -> tensor (B,C_used,T,V) on device."""
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 3:
            arr = arr[None, ...]  # (1,T,V,C)
        assert arr.ndim == 4
        if drop_conf:
            arr = arr[..., :2]
        arr = arr.astype('float32')
        B, T, V, C = arr.shape
        arr = np.transpose(arr, (0, 3, 1, 2))  # (B,C,T,V)
        return torch.from_numpy(arr).to(device)
    else:
        ten = x
        if ten.dim() == 3:
            ten = ten.unsqueeze(0)
        assert ten.dim() == 4
        if drop_conf and ten.size(-1) >= 3:
            ten = ten[..., :2]
        B, T, V, C = ten.shape
        ten = ten.permute(0, 3, 1, 2).contiguous()  # (B,C,T,V)
        return ten.to(device)


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
        device = next(self.parameters()).device
        feats = []
        B_ref: Optional[int] = None
        for part in self.parts:
            k = parts_kpts[part]
            x = _np_or_torch_to_nctv(k, drop_conf=self.drop_conf, device=device)
            if B_ref is None:
                B_ref = x.size(0)
            else:
                assert x.size(0) == B_ref, "All parts must share the same batch size"
            x = self.backbones[part](x)  # [B, Dp]
            feats.append(x)
        z = self.fusion(feats)  # [B, D]
        return z
