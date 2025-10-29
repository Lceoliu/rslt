from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .backbones.uni_gcn_part import UniGCNPartBackbone

__all__ = ["MultiPartGCNModel", "PARTS_DEFAULT"]

_DEFAULT_PARTS: Tuple[str, ...] = (
    "body",
    "face",
    "left_hand",
    "right_hand",
    "fullbody",
)

PARTS_DEFAULT: Tuple[str, ...] = _DEFAULT_PARTS


def _slice_pose_by_part(
    pose: torch.Tensor, part_lens: Sequence[int]
) -> Sequence[torch.Tensor]:
    if sum(part_lens) != pose.size(2):
        raise ValueError(
            "Sum of part_lens must equal the joint dimension of pose."
        )
    return pose.split(tuple(int(l) for l in part_lens), dim=2)


def _build_masks(
    pose_len: Optional[torch.Tensor],
    last_chunk_valid_len: Optional[torch.Tensor],
    *,
    batch: int,
    num_chunks: int,
    chunk_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if pose_len is None:
        return None, None, None
    if pose_len.dim() != 1 or pose_len.numel() != batch:
        raise ValueError("pose_len must be 1D with length equal to batch size.")
    pose_len = pose_len.to(device=device, dtype=torch.long)
    chunk_ids = torch.arange(num_chunks, device=device)
    chunk_mask = chunk_ids.unsqueeze(0) < pose_len.unsqueeze(1)  # [B, N]
    
    # Expand chunk mask to frame level
    frame_mask_bool = chunk_mask.view(-1, 1).expand(-1, chunk_len).clone()  # [B*N, T]

    # Refine mask for the last valid chunk of each sample
    if last_chunk_valid_len is not None:
        if last_chunk_valid_len.dim() != 1 or last_chunk_valid_len.numel() != batch:
            raise ValueError("last_chunk_valid_len must be 1D with length equal to batch size.")
        last_chunk_valid_len = last_chunk_valid_len.to(device=device, dtype=torch.long)
        for i in range(batch):
            # Index of the last valid chunk for sample i
            last_valid_chunk_idx = pose_len[i] - 1
            if last_valid_chunk_idx >= 0:
                valid_frames = last_chunk_valid_len[i]
                # Get the index in the flattened BN dimension
                flat_idx = i * num_chunks + last_valid_chunk_idx
                # Set the padding part of the mask for this specific chunk to False
                frame_mask_bool[flat_idx, valid_frames:] = False

    frame_mask = frame_mask_bool.to(dtype)
    return frame_mask, frame_mask_bool, chunk_mask


class MultiPartGCNModel(nn.Module):
    """Encode chunked multi-part poses with Uni-GCN backbones.

    Args:
        parts: Ordered list of body part names matching the joint layout.
        drop_conf: Whether to drop the confidence channel before encoding.
        embed_dim: Output feature dimensionality of each part-specific backbone.
        proj_dim: Projection dimensionality inside Uni-GCN.
        temporal_kernel: Temporal kernel size for Uni-GCN blocks.
        adaptive: Whether Uni-GCN uses adaptive adjacency.
        dropout: Dropout applied after Uni-GCN projection.
    """

    def __init__(
        self,
        *,
        parts: Optional[Sequence[str]] = None,
        drop_conf: bool = True,
        embed_dim: int = 256,
        proj_dim: int = 64,
        temporal_kernel: int = 5,
        adaptive: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.parts: Tuple[str, ...] = tuple(parts) if parts else _DEFAULT_PARTS
        self.drop_conf = drop_conf
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.temporal_kernel = temporal_kernel
        self.adaptive = adaptive
        self.dropout = dropout

        self._in_channels: Optional[int] = None
        self.backbones = nn.ModuleDict()

    @property
    def is_initialized(self) -> bool:
        return len(self.backbones) > 0

    def _ensure_backbones(
        self,
        adjacency: Dict[str, torch.Tensor],
        in_channels: int,
        device: torch.device,
    ) -> None:
        if not adjacency:
            raise ValueError("Adjacency matrices are required to initialise backbones.")
        missing = [part for part in self.parts if part not in adjacency]
        if missing:
            raise KeyError(f"Missing adjacency for parts: {missing}")
        if self._in_channels is not None and self._in_channels != in_channels:
            raise ValueError(
                "Cannot change input channel count once backbones are initialised."
            )
        if self.backbones:
            return
        for part in self.parts:
            adj = adjacency[part]
            if not isinstance(adj, torch.Tensor):
                adj = torch.as_tensor(adj, dtype=torch.float32, device=device)
            else:
                adj = adj.detach().to(device=device, dtype=torch.float32)
            backbone = UniGCNPartBackbone(
                in_channels=in_channels,
                adjacency=adj,
                proj_dim=self.proj_dim,
                embed_dim=self.embed_dim,
                adaptive=self.adaptive,
                temporal_kernel_size=self.temporal_kernel,
                dropout=self.dropout,
            ).to(device)
            self.backbones[part] = backbone
        self._in_channels = in_channels

    def initialize_backbones(
        self,
        adjacency: Dict[str, torch.Tensor] | Dict[str, Any],
        *,
        in_channels: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:

        if in_channels is None:
            in_channels = 2 if self.drop_conf else 3

        if device is None:
            param = next(self.parameters(), None)
            device = param.device if param is not None else torch.device('cpu')

        prepared: Dict[str, torch.Tensor] = {}
        for part in self.parts:
            if part not in adjacency:
                raise KeyError(
                    f"Missing adjacency for part '{part}' during initialization."
                )
            adj = adjacency[part]
            if not isinstance(adj, torch.Tensor):
                adj = torch.as_tensor(adj, dtype=torch.bfloat16, device=device)
            prepared[part] = adj.detach().to(device=device, dtype=torch.bfloat16)
        print(
            f"Prepared adjacency matrices for parts: {list(prepared.keys())} \n device: {device}, dtype: {prepared[part].dtype}, in_channels: {in_channels}"
        )
        self._ensure_backbones(prepared, in_channels, device)

    def forward(
        self,
        pose: torch.Tensor,
        *,
        part_lens: Sequence[int],
        valid_mask: Optional[torch.Tensor] = None,
        adjacency: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run Uni-GCN encoders over chunked poses.

        Args:
            pose: Tensor shaped [B, chunk_len, sum_K, C].
            part_lens: Joint counts per part matching ``self.parts``.
            valid_mask: Optional bool mask [B, chunk_len] for valid frames.
            adjacency: Part-wise adjacency matrices used on the first call.

        Returns:
            features: Tensor [B, P, chunk_len, D].
        """

        if len(part_lens) != len(self.parts):
            raise ValueError("part_lens length must match the configured parts.")

        batch_size, chunk_len, total_joints, channels = pose.shape
        if sum(part_lens) != total_joints:
            raise ValueError("part_lens must sum to the joint dimension of pose.")

        channels_used = channels
        if self.drop_conf:
            if channels < 2:
                raise ValueError("drop_conf=True requires at least two channels.")
            channels_used = min(2, channels)

        if not self.backbones:
            if adjacency is None:
                raise ValueError(
                    "Adjacency matrices must be provided before the first forward call."
                )
            self._ensure_backbones(adjacency, channels_used, pose.device)
        elif adjacency is not None:
            self._ensure_backbones(adjacency, channels_used, pose.device)

        flat_pose = pose.reshape(batch_size, chunk_len, total_joints, channels)
        part_poses = _slice_pose_by_part(flat_pose, part_lens)

        outputs = []

        for part_name, part_pose in zip(self.parts, part_poses):
            if self.drop_conf and channels >= 2:
                part_pose = part_pose[..., :channels_used]
            x = part_pose.permute(0, 3, 1, 2).contiguous()  # [B, C, T, K]
            feats = self.backbones[part_name](
                x,
                mask=valid_mask,
                return_seq=True,
            )  # [B, T, D]
            outputs.append(feats)

        features = torch.stack(outputs, dim=1)  # [B, P, T, D]
        return features
