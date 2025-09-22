from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.transform import NormalizeProcessor
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


class ChunkTransformerEncoder(nn.Module):
    """Lightweight Transformer encoder to map variable-length sequence [B, T, D]
    to fixed-length tokens [B, P, D_out]. Uses sinusoidal positional encoding and
    simple linear interpolation to exactly P tokens after contextualization.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        preset_len: int = 1,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.preset_len = int(max(1, preset_len))
        d_model = d_out
        if dim_feedforward is None:
            dim_feedforward = max(256, 4 * d_model)
        self.in_proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    @staticmethod
    def _positional_encoding(B: int, T: int, D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Sin-cos positional encoding
        position = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype) * (-np.log(10000.0) / max(1, D)))
        pe = torch.zeros(T, D, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).expand(B, -1, -1)  # [B, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in] -> [B, P, D_out]
        B, T, Din = x.shape
        x = self.in_proj(x)
        D = x.size(-1)
        pe = self._positional_encoding(B, T, D, x.device, x.dtype)
        x = x + pe
        y = self.encoder(x)  # [B, T, D]
        if self.preset_len == T:
            return y
        # Downsample/upsample to preset_len via linear interpolation along time
        yT = y.transpose(1, 2)  # [B, D, T]
        yT2 = F.interpolate(yT, size=self.preset_len, mode='linear', align_corners=False)
        out = yT2.transpose(1, 2).contiguous()  # [B, P, D]
        return out


class MultiPartGCNModel(nn.Module):

    def __init__(
        self,
        parts: Optional[List[str]] = None,
        backbone: str = "aagcn",  # or "stgcn"
        part_embed_dim: int = 256,
        out_embed_dim: int = 512,
        drop_conf: bool = True,
        fusion: str = "attention",  # "attention" "concat_mlp"
        preset_len: int = 1,
    ) -> None:
        super().__init__()
        self.parts = parts or PARTS_DEFAULT
        self.drop_conf = drop_conf
        self.preset_len = int(max(1, preset_len))

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

        # Chunk-level sequence encoder to fixed number of tokens per chunk
        self.chunk_encoder = ChunkTransformerEncoder(
            d_in=out_embed_dim,
            d_out=out_embed_dim,
            preset_len=self.preset_len,
            nhead=8,
            num_layers=4,
            dim_feedforward=max(256, 4 * out_embed_dim),
            dropout=0.1,
        )

    def forward(self, parts_kpts: Dict[str, np.ndarray | torch.Tensor], pose_len: Optional[torch.Tensor | np.ndarray] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        feats = []
        B_ref: Optional[int] = None
        # Build optional mask from pose_len
        mask: Optional[torch.Tensor] = None
        if pose_len is not None:
            if isinstance(pose_len, np.ndarray):
                pose_len = torch.from_numpy(pose_len)
            pose_len = pose_len.to(device)
        for part in self.parts:
            k = parts_kpts[part]
            x = _np_or_torch_to_nctv(k, drop_conf=self.drop_conf, device=device)
            if B_ref is None:
                B_ref = x.size(0)
            else:
                assert x.size(0) == B_ref, "All parts must share the same batch size"
            # Create/update mask matching current T
            if pose_len is not None:
                T = x.size(2)
                m = torch.arange(T, device=device).unsqueeze(0).expand(B_ref, -1)
                mask = (m < pose_len.view(-1, 1)).to(x.dtype)
            x = self.backbones[part](x, mask=mask)  # [B, Dp]
            feats.append(x)
        z = self.fusion(feats)  # [B, D]
        return z

    @torch.no_grad()
    def _infer_common_T(self, parts_kpts: Dict[str, np.ndarray | torch.Tensor]) -> int:
        # Guess T from any part
        for v in parts_kpts.values():
            if isinstance(v, np.ndarray):
                return int(v.shape[1] if v.ndim == 4 else v.shape[0])
            else:
                return int(v.shape[1] if v.dim() == 4 else v.shape[0])
        return 0

    def encode_chunks(
        self,
        parts_kpts: Dict[str, np.ndarray | torch.Tensor],
        pose_len: Optional[torch.Tensor | np.ndarray],
        window: int,
        stride: int,
        drop_last: bool = True,
    ) -> torch.Tensor:
        """Encode sliding-window chunks to a sequence of embeddings.

        Returns: Tensor[B, N_chunks, P, D]
        """
        device = next(self.parameters()).device
        # Normalize pose_len
        if pose_len is not None and isinstance(pose_len, np.ndarray):
            pose_len = torch.from_numpy(pose_len)
        if pose_len is not None:
            pose_len = pose_len.to(device)

        # Determine common number of chunks based on min valid length (ensure at least 1)
        if pose_len is not None:
            min_len = int(torch.as_tensor(pose_len).min().item())
        else:
            min_len = self._infer_common_T(parts_kpts)

        starts: list[int] = []
        if drop_last:
            if min_len >= window:
                starts = list(range(0, min_len - window + 1, max(1, stride)))
            else:
                starts = [0]
        else:
            # Allow trailing incomplete; we will mask paddings
            starts = list(range(0, max(1, min_len), max(1, stride)))

        # Helper to slice time range [s, s+window)
        def _slice_time(x: np.ndarray | torch.Tensor, s: int, w: int):
            if isinstance(x, np.ndarray):
                if x.ndim == 4:  # [B,T,V,C]
                    return x[:, s : s + w]
                return x[s : s + w]
            else:
                if x.dim() == 4:
                    return x[:, s : s + w]
                return x[s : s + w]

        feats_seq = []  # list of [B, P, D]
        B_ref: Optional[int] = None
        for s in starts:
            # Build per-chunk features then fuse
            chunk_feats_seq: List[torch.Tensor] = []  # each [B, t', Dp] for every part
            chunk_mask: Optional[torch.Tensor] = None  # [B, t_raw]
            for part in self.parts:
                k_all = parts_kpts[part]  # [B, T, V, C] or [T, V, C]
                k_chunk = _slice_time(k_all, s, window)  # [B,w,V,C] or [w,V,C]
                x = _np_or_torch_to_nctv(
                    k_chunk, drop_conf=self.drop_conf, device=device
                )  # [B, C, w, V]
                if B_ref is None:
                    B_ref = x.size(0)
                # per-chunk mask
                if pose_len is not None:
                    t = x.size(2)  # <= window (may be shorter if np slice)
                    m = (
                        torch.arange(t, device=device).unsqueeze(0).expand(B_ref, -1)
                    )  # [B, w]
                    chunk_mask = (m + s < pose_len.view(-1, 1)).to(x.dtype)
                # Get per-time features for this chunk
                y_seq = self.backbones[part](x, mask=chunk_mask, return_seq=True)  # [B, t', Dp]
                chunk_feats_seq.append(y_seq)
            # Fuse across parts per time step by flattening time into batch
            t_prime = min(f.shape[1] for f in chunk_feats_seq)
            # Align time lengths if needed (should be same; safeguard by min crop)
            chunk_feats_seq = [f[:, :t_prime] for f in chunk_feats_seq]
            # Flatten time into batch
            flat_parts = [f.reshape(-1, f.size(-1)) for f in chunk_feats_seq]  # list of [B*t', Dp]
            z_flat = self.fusion(flat_parts)  # [B*t', D]
            z_seq = z_flat.view(B_ref, t_prime, -1)  # [B, t', D]
            # Encode to fixed P tokens
            z_tok = self.chunk_encoder(z_seq)  # [B, P, D]
            feats_seq.append(z_tok)

        z_seq = torch.stack(feats_seq, dim=1)  # [B, N_chunks, P, D]
        return z_seq
