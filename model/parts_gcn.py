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
        enable_body_fusion: Whether to fuse body features into other parts (UniSign-style).
        share_hand_params: Whether left_hand and right_hand share parameters.
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
        enable_body_fusion: bool = True,
        share_hand_params: bool = True,
    ) -> None:
        super().__init__()
        self.parts: Tuple[str, ...] = tuple(parts) if parts else _DEFAULT_PARTS
        self.drop_conf = drop_conf
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.temporal_kernel = temporal_kernel
        self.adaptive = adaptive
        self.dropout = dropout
        self.enable_body_fusion = enable_body_fusion
        self.share_hand_params = share_hand_params

        self._in_channels: Optional[int] = None
        self.backbones = nn.ModuleDict()

        # Body keypoint indices for fusion (COCO-17 format)
        # 0: nose, 9: left_wrist, 10: right_wrist
        self.body_keypoint_map = {
            'left_hand': 9,   # left wrist
            'right_hand': 10,  # right wrist
            'face': 0,         # nose/neck
        }

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

        # Track which parts have been created (for parameter sharing)
        created_parts = set()

        for part in self.parts:
            # Skip left_hand if sharing with right_hand (create right_hand first)
            if self.share_hand_params and part == 'left_hand' and 'right_hand' in self.parts:
                if 'right_hand' not in created_parts:
                    continue  # Will be handled when we process right_hand

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
            created_parts.add(part)

            # If this is right_hand and we share params, also assign to left_hand
            if self.share_hand_params and part == 'right_hand' and 'left_hand' in self.parts:
                print(f"[MultiPartGCN] Sharing parameters: left_hand <-> right_hand")
                self.backbones['left_hand'] = self.backbones['right_hand']
                created_parts.add('left_hand')

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
        pose_len: Optional[torch.Tensor] = None,
        last_chunk_valid_len: Optional[torch.Tensor] = None,
        adjacency: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run Uni-GCN encoders over chunked poses.

        Args:
            pose: Tensor shaped [B, N_chunk, chunk_len, sum_K, C].
            part_lens: Joint counts per part matching ``self.parts``.
            pose_len: Optional valid chunk counts per sample ``[B]``.
            last_chunk_valid_len: Optional valid frame counts for the last chunk ``[B]``.
            adjacency: Part-wise adjacency matrices used on the first call.

        Returns:
            features: Tensor [B*N_chunk, P, chunk_len, D].
            frame_mask: Optional bool mask [B*N_chunk, chunk_len] for valid frames.
            chunk_mask: Optional bool mask [B, N_chunk] for valid chunks.
        """

        if pose.dim() != 5:
            raise ValueError(
                "pose must have shape [B, N_chunk, chunk_len, sum_K, C]."
            )
        if len(part_lens) != len(self.parts):
            raise ValueError("part_lens length must match the configured parts.")

        batch_size, num_chunks, chunk_len, total_joints, channels = pose.shape
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

        frame_mask_float, frame_mask_bool, chunk_mask = _build_masks(
            pose_len,
            last_chunk_valid_len,
            batch=batch_size,
            num_chunks=num_chunks,
            chunk_len=chunk_len,
            device=pose.device,
            dtype=pose.dtype,
        )

        flat_pose = pose.reshape(
            batch_size * num_chunks, chunk_len, total_joints, channels
        )
        part_poses = _slice_pose_by_part(flat_pose, part_lens)

        # === Phase 1: Spatial GCN for all parts (UniSign-style) ===
        spatial_features = {}
        mask_for_backbone = frame_mask_float

        for part_name, part_pose in zip(self.parts, part_poses):
            if self.drop_conf and channels >= 2:
                part_pose = part_pose[..., :channels_used]
            x = part_pose.permute(0, 3, 1, 2).contiguous()  # [B*N, C, T, K]

            # Execute only spatial GCN
            spatial_feat = self.backbones[part_name].forward_spatial(
                x, mask=mask_for_backbone
            )  # [B*N, C_spatial, T, V]
            spatial_features[part_name] = spatial_feat

        # === Phase 2: Body-to-Part Fusion (UniSign-style) ===
        if self.enable_body_fusion and 'body' in self.parts:
            body_spatial_feat = spatial_features['body']  # [B*N, C_spatial, T, V_body]

            for part_name in self.parts:
                if part_name == 'body' or part_name == 'fullbody':
                    continue

                # Get the corresponding body keypoint index for this part
                body_kp_idx = self.body_keypoint_map.get(part_name)
                if body_kp_idx is not None:
                    # Extract the specific body keypoint feature
                    # body_spatial_feat: [B*N, C_spatial, T, V_body]
                    # Extract keypoint at index body_kp_idx: [B*N, C_spatial, T, 1]
                    body_node_feat = body_spatial_feat[:, :, :, body_kp_idx:body_kp_idx+1]

                    # Add to the part's spatial features (broadcast across all keypoints)
                    # This follows UniSign's approach: gcn_feat + body_feat[..., idx][..., None].detach()
                    spatial_features[part_name] = spatial_features[part_name] + body_node_feat.detach()

        # === Phase 3: Temporal GCN and pooling for all parts ===
        outputs = []
        for part_name in self.parts:
            spatial_feat = spatial_features[part_name]

            # Execute temporal GCN and pooling
            feats = self.backbones[part_name].forward_temporal(
                spatial_feat,
                mask=mask_for_backbone,
                return_seq=True,
            )  # [B*N, T, D]

            if feats.dtype != pose.dtype:
                feats = feats.to(pose.dtype)
            outputs.append(feats)

        features = torch.stack(outputs, dim=1)  # [B*N, P, T, D]
        return features, frame_mask_bool, chunk_mask
