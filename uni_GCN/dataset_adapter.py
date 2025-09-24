"""
Dataset adapter for integrating Uni-GCN with custom dataset processing pipeline.

This module provides adapters to seamlessly integrate the Uni-GCN ST-GCN models
with the MyDataset data loading pipeline that uses COCO_Wholebody format.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

from .uni_stgcn import UniSTGCN, MultiPartSTGCN
from .keypoint_formats import get_keypoint_format, create_mydataset_compatible_format
from .format_converter import KeypointConverter


class DatasetSTGCN(nn.Module):
    """
    ST-GCN model adapter for MyDataset pipeline

    This class wraps UniSTGCN to work directly with the output format from
    MyDataset and my_collate_fn, which provides data in the format:
    {
        'pose': {
            'body': (B, T, K, C),
            'face': (B, T, K, C),
            'left_hand': (B, T, K, C),
            'right_hand': (B, T, K, C)
        },
        'pose_len': (B,),
        'adjacency_matrix': {part: Tensor},
        ...
    }
    """

    def __init__(
        self,
        parts: List[str] = ['body', 'left_hand', 'right_hand'],
        hidden_dim: int = 64,
        graph_strategy: str = 'distance',
        adaptive_graph: bool = True,
        max_hop: int = 1,
        output_pooling: str = 'mean',
        use_pose_length_mask: bool = True,
        temporal_pooling: str = 'mean',  # 'mean', 'max', 'last', 'attention'
        body_info_path: str = None  # Path to MyDataset body_info.json
    ):
        """
        Args:
            parts: List of body parts to use from dataset
            hidden_dim: Hidden dimension for ST-GCN
            graph_strategy: Graph adjacency strategy
            adaptive_graph: Whether to use learnable adjacency matrices
            max_hop: Maximum hop distance for graph connections
            output_pooling: Vertex pooling method ('mean', 'max', 'none')
            use_pose_length_mask: Whether to apply masking based on actual sequence lengths
            temporal_pooling: How to pool temporal dimension ('mean', 'max', 'last', 'attention')
            body_info_path: Path to MyDataset body_info.json for format compatibility
        """
        super().__init__()

        self.parts = parts
        self.use_pose_length_mask = use_pose_length_mask
        self.temporal_pooling = temporal_pooling

        # Create MyDataset-compatible COCO_Wholebody format
        if body_info_path:
            self.keypoint_format = create_mydataset_compatible_format(body_info_path)
        else:
            # Use default MyDataset settings (discarded keypoints [13,14,15,16])
            self.keypoint_format = create_mydataset_compatible_format()

        # Create ST-GCN model with MyDataset-compatible COCO_Wholebody format
        stgcn_parts = []
        for part in parts:
            if part in ['left_hand', 'right_hand', 'body', 'face']:
                # Use format name string instead of object to avoid PyTorch ModuleDict key issues
                stgcn_parts.append((self.keypoint_format.name, part))
            else:
                # Fallback to legacy naming
                legacy_name = self._map_to_legacy_name(part)
                stgcn_parts.append(legacy_name)

        self.stgcn = UniSTGCN(
            parts=stgcn_parts,
            keypoint_format=self.keypoint_format,  # Use MyDataset-compatible format
            input_dim=3,
            hidden_dim=hidden_dim,
            graph_strategy=graph_strategy,
            adaptive_graph=adaptive_graph,
            max_hop=max_hop,
            output_pooling=output_pooling
        )

        # Temporal pooling layer
        self.feature_dim = self.stgcn.get_output_dim()
        if temporal_pooling == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

    def _map_to_legacy_name(self, part_name: str) -> str:
        """Map dataset part names to legacy ST-GCN part names"""
        mapping = {
            'left_hand': 'left',
            'right_hand': 'right',
            'face': 'face_all'
        }
        return mapping.get(part_name, part_name)

    def _create_sequence_mask(self, batch_size: int, max_seq_len: int,
                            seq_lengths: torch.Tensor) -> torch.Tensor:
        """Create mask for variable length sequences"""
        # seq_lengths: (B,)
        # return: (B, T) where True means valid position
        mask = torch.arange(max_seq_len, device=seq_lengths.device).expand(
            batch_size, max_seq_len
        ) < seq_lengths.unsqueeze(1)
        return mask

    def forward(self, batch: Dict, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass using batch from MyDataset

        Args:
            batch: Batch dictionary from my_collate_fn containing:
                - 'pose': {part: (B, T, K, C)} - pose data for each part
                - 'pose_len': (B,) - actual sequence lengths
                - 'adjacency_matrix': {part: Tensor} - adjacency matrices (optional)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - 'features': (B, feature_dim) - final pooled features
                - 'sequence_features': (B, T, feature_dim) - sequence-level features (optional)
                - 'intermediate_features': dict - intermediate ST-GCN features (optional)
        """
        pose_data = batch['pose']
        pose_lengths = batch.get('pose_len', None)

        # Prepare input for ST-GCN
        stgcn_input = {}
        for i, part in enumerate(self.parts):
            if part not in pose_data:
                raise ValueError(f"Part '{part}' not found in pose data. Available: {list(pose_data.keys())}")

            part_key = self.stgcn.part_keys[i]
            stgcn_input[part_key] = pose_data[part]  # (B, T, K, C)

        # Forward through ST-GCN
        if return_features:
            sequence_features, intermediate_features = self.stgcn(
                stgcn_input, return_features=True
            )
        else:
            sequence_features = self.stgcn(stgcn_input)  # (B, T, feature_dim)
            intermediate_features = None

        batch_size, seq_len, feature_dim = sequence_features.shape

        # Apply sequence masking if needed
        if self.use_pose_length_mask and pose_lengths is not None:
            mask = self._create_sequence_mask(batch_size, seq_len, pose_lengths)
            # Zero out features for padded positions
            sequence_features = sequence_features * mask.unsqueeze(-1).float()

        # Temporal pooling
        if self.temporal_pooling == 'mean':
            if self.use_pose_length_mask and pose_lengths is not None:
                # Masked average
                valid_lengths = pose_lengths.float().clamp(min=1).unsqueeze(-1)
                pooled_features = sequence_features.sum(dim=1) / valid_lengths
            else:
                pooled_features = sequence_features.mean(dim=1)

        elif self.temporal_pooling == 'max':
            if self.use_pose_length_mask and pose_lengths is not None:
                # Set padded positions to very negative values before max pooling
                mask = self._create_sequence_mask(batch_size, seq_len, pose_lengths)
                masked_features = sequence_features.masked_fill(
                    ~mask.unsqueeze(-1), float('-inf')
                )
                pooled_features = masked_features.max(dim=1)[0]
            else:
                pooled_features = sequence_features.max(dim=1)[0]

        elif self.temporal_pooling == 'last':
            if self.use_pose_length_mask and pose_lengths is not None:
                # Use the last valid frame for each sequence
                indices = (pose_lengths - 1).clamp(min=0)
                pooled_features = sequence_features[torch.arange(batch_size), indices]
            else:
                pooled_features = sequence_features[:, -1]  # Last frame

        elif self.temporal_pooling == 'attention':
            # Self-attention pooling
            if self.use_pose_length_mask and pose_lengths is not None:
                # Create attention mask
                attn_mask = ~self._create_sequence_mask(batch_size, seq_len, pose_lengths)
                # Use sequence_features as query, key, and value
                attended_features, _ = self.temporal_attention(
                    sequence_features, sequence_features, sequence_features,
                    key_padding_mask=attn_mask
                )
                # Average over sequence dimension
                pooled_features = attended_features.mean(dim=1)
            else:
                attended_features, _ = self.temporal_attention(
                    sequence_features, sequence_features, sequence_features
                )
                pooled_features = attended_features.mean(dim=1)
        else:
            raise ValueError(f"Unknown temporal pooling method: {self.temporal_pooling}")

        # Prepare output
        output = {
            'features': pooled_features,  # (B, feature_dim)
            'sequence_features': sequence_features  # (B, T, feature_dim)
        }

        if return_features and intermediate_features is not None:
            output['intermediate_features'] = intermediate_features

        return output

    def get_output_dim(self) -> int:
        """Get the output feature dimension"""
        return self.feature_dim


class MultiTaskSTGCN(nn.Module):
    """
    Multi-task ST-GCN model for sign language recognition, translation, etc.

    This extends DatasetSTGCN with task-specific heads for different outputs.
    """

    def __init__(
        self,
        parts: List[str] = ['body', 'left_hand', 'right_hand'],
        hidden_dim: int = 64,
        tasks: Dict[str, int] = None,  # task_name -> num_classes/vocab_size
        body_info_path: str = None,  # Path to MyDataset body_info.json
        **stgcn_kwargs
    ):
        """
        Args:
            parts: Body parts to use
            hidden_dim: Hidden dimension
            tasks: Dictionary of task names and their output dimensions
                   e.g., {'recognition': 1000, 'translation': 5000}
        """
        super().__init__()

        self.tasks = tasks or {}

        # Base ST-GCN
        self.stgcn = DatasetSTGCN(
            parts=parts,
            hidden_dim=hidden_dim,
            body_info_path=body_info_path,
            **stgcn_kwargs
        )

        feature_dim = self.stgcn.get_output_dim()

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, output_dim in self.tasks.items():
            if task_name in ['recognition', 'classification']:
                # Classification head
                self.task_heads[task_name] = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(feature_dim, output_dim)
                )
            elif task_name in ['translation', 'generation']:
                # For sequence generation tasks, we might want different architectures
                self.task_heads[task_name] = nn.Linear(feature_dim, output_dim)
            else:
                # Generic linear head
                self.task_heads[task_name] = nn.Linear(feature_dim, output_dim)

    def forward(self, batch: Dict, tasks: Optional[List[str]] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs

        Args:
            batch: Batch from dataset
            tasks: List of tasks to compute outputs for. If None, compute all tasks.
            return_features: Whether to return ST-GCN features

        Returns:
            Dictionary with task outputs and optionally features
        """
        # Get ST-GCN features
        stgcn_output = self.stgcn(batch, return_features=return_features)
        features = stgcn_output['features']  # (B, feature_dim)

        # Compute task-specific outputs
        outputs = {}
        if return_features:
            outputs.update(stgcn_output)

        task_list = tasks or list(self.tasks.keys())
        for task_name in task_list:
            if task_name in self.task_heads:
                outputs[task_name] = self.task_heads[task_name](features)

        return outputs


# Convenience functions for common configurations
def create_sign_recognition_model(
    num_classes: int,
    parts: List[str] = ['body', 'left_hand', 'right_hand'],
    hidden_dim: int = 128,
    body_info_path: str = None,
    **kwargs
) -> MultiTaskSTGCN:
    """Create a model for isolated sign language recognition

    Args:
        num_classes: Number of sign classes
        parts: Body parts to use
        hidden_dim: Hidden dimension
        body_info_path: Path to MyDataset body_info.json for compatibility
        **kwargs: Additional arguments for MultiTaskSTGCN
    """
    # Set default temporal pooling if not provided
    kwargs.setdefault('temporal_pooling', 'mean')

    return MultiTaskSTGCN(
        parts=parts,
        hidden_dim=hidden_dim,
        tasks={'recognition': num_classes},
        body_info_path=body_info_path,
        **kwargs
    )


def create_sign_translation_model(
    vocab_size: int,
    parts: List[str] = ['body', 'left_hand', 'right_hand', 'face'],
    hidden_dim: int = 128,
    body_info_path: str = None,
    **kwargs
) -> DatasetSTGCN:
    """Create a model for sign language translation (encoder part)"""
    # Set default temporal pooling if not provided
    kwargs.setdefault('temporal_pooling', 'attention')  # Better for sequence-to-sequence

    return DatasetSTGCN(
        parts=parts,
        hidden_dim=hidden_dim,
        body_info_path=body_info_path,
        **kwargs
    )


def create_multi_task_model(
    num_recognition_classes: int,
    vocab_size: int,
    parts: List[str] = ['body', 'left_hand', 'right_hand'],
    hidden_dim: int = 128,
    body_info_path: str = None,
    **kwargs
) -> MultiTaskSTGCN:
    """Create a multi-task model for both recognition and translation"""
    # Set default temporal pooling if not provided
    kwargs.setdefault('temporal_pooling', 'attention')

    return MultiTaskSTGCN(
        parts=parts,
        hidden_dim=hidden_dim,
        tasks={
            'recognition': num_recognition_classes,
            'translation': vocab_size
        },
        body_info_path=body_info_path,
        **kwargs
    )


class SignLanguageTrainer:
    """
    Complete training pipeline for sign language tasks using Uni-GCN
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.model = model.to(device)
        self.device = device

        # Import optimizer here to avoid dependency issues
        import torch.optim as optim
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()

        # Move batch to device
        batch = self._move_batch_to_device(batch)

        # Forward pass
        outputs = self.model(batch)

        # Compute losses based on available tasks
        losses = {}
        total_loss = 0

        if 'recognition' in outputs:
            # For recognition task, we need ground truth labels
            # This would come from your dataset annotation
            if 'labels' in batch:  # Assuming labels are provided
                rec_loss = self.criterion(outputs['recognition'], batch['labels'])
                losses['recognition'] = rec_loss
                total_loss += rec_loss

        if 'translation' in outputs:
            # For translation task, we need target sequences
            if 'target_ids' in batch:  # Assuming target token IDs are provided
                trans_loss = self.criterion(
                    outputs['translation'].view(-1, outputs['translation'].size(-1)),
                    batch['target_ids'].view(-1)
                )
                losses['translation'] = trans_loss
                total_loss += trans_loss

        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        losses['total'] = total_loss
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def validate_step(self, batch: Dict) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        with torch.no_grad():
            batch = self._move_batch_to_device(batch)
            outputs = self.model(batch)

            # Compute validation metrics
            metrics = {}
            if 'recognition' in outputs and 'labels' in batch:
                predictions = outputs['recognition'].argmax(dim=-1)
                accuracy = (predictions == batch['labels']).float().mean()
                metrics['accuracy'] = accuracy.item()

            return metrics

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                # Handle nested dictionaries (like pose data)
                device_batch[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value
        return device_batch