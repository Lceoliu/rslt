"""
Unified ST-GCN module for multi-part skeletal modeling.

This module provides a high-level interface for processing different body parts
(body, hands, face) using spatio-temporal graph convolutional networks.
"""

import torch
import torch.nn as nn
from .graph import Graph
from .stgcn_block import get_stgcn_chain
from .keypoint_formats import KeypointFormat, get_keypoint_format
from .format_converter import KeypointConverter


class UniSTGCN(nn.Module):
    """Unified Spatio-Temporal Graph Convolutional Network

    A unified model for processing multiple body parts with ST-GCN.
    Supports multiple keypoint formats and custom part configurations.

    Args:
        parts (list): List of body parts to model
            - Legacy: ['body', 'left', 'right', 'face_all']
            - With format: [('coco_wholebody', 'body'), ('coco_wholebody', 'left_hand'), ...]
        input_dim (int): Input feature dimension (default: 3 for x,y,confidence)
        hidden_dim (int): Hidden feature dimension after projection
        graph_strategy (str): Graph adjacency strategy ('uniform', 'distance', 'spatial')
        adaptive_graph (bool): Whether to use learnable adjacency matrices
        max_hop (int): Maximum hop distance for graph connections
        output_pooling (str): Output pooling method ('mean', 'max', 'none')
        keypoint_format (str or KeypointFormat): Keypoint format specification
        auto_convert (bool): Automatically convert input keypoints to expected format
    """

    def __init__(
        self,
        parts=['body', 'left', 'right', 'face_all'],
        input_dim=3,
        hidden_dim=64,
        graph_strategy='distance',
        adaptive_graph=True,
        max_hop=1,
        output_pooling='mean',
        keypoint_format=None,
        auto_convert=False
    ):
        super(UniSTGCN, self).__init__()

        self.parts = parts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_pooling = output_pooling
        self.auto_convert = auto_convert

        # Setup keypoint format
        self.keypoint_format = None
        self.converter = None
        if keypoint_format is not None:
            if isinstance(keypoint_format, str):
                self.keypoint_format = get_keypoint_format(keypoint_format)
            else:
                self.keypoint_format = keypoint_format
            self.converter = KeypointConverter(self.keypoint_format)

        # Build graphs and networks for each body part
        self.graphs = {}
        self.proj_layers = nn.ModuleDict()
        self.spatial_gcns = nn.ModuleDict()
        self.temporal_gcns = nn.ModuleDict()
        self.part_keys = []  # Store part keys for forward pass

        for i, part in enumerate(self.parts):
            # Handle different part specifications
            if isinstance(part, tuple):
                # Format-specific part: (format_name, part_name)
                part_key = f"{part[0]}_{part[1]}"
                layout_spec = part
                if self.keypoint_format is None:
                    self.keypoint_format = get_keypoint_format(part[0])
                    self.converter = KeypointConverter(self.keypoint_format)
            else:
                # Legacy part name
                part_key = part
                if self.keypoint_format is not None:
                    layout_spec = (self.keypoint_format.name, part)
                else:
                    layout_spec = part

            self.part_keys.append(part_key)

            # Create graph structure
            graph = Graph(
                layout=layout_spec,
                strategy=graph_strategy,
                max_hop=max_hop,
                keypoint_format=self.keypoint_format
            )
            A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
            self.graphs[part_key] = A

            # Input projection layer
            self.proj_layers[part_key] = nn.Linear(input_dim, hidden_dim)

            # Spatial ST-GCN chain
            spatial_kernel_size = (1, A.size(0))
            spatial_gcn, spatial_out_dim = get_stgcn_chain(
                hidden_dim, 'spatial', spatial_kernel_size, A.clone(), adaptive_graph
            )
            self.spatial_gcns[part_key] = spatial_gcn

            # Temporal ST-GCN chain
            temporal_kernel_size = (5, A.size(0))  # Temporal kernel size = 5
            temporal_gcn, temporal_out_dim = get_stgcn_chain(
                spatial_out_dim, 'temporal', temporal_kernel_size, A.clone(), adaptive_graph
            )
            self.temporal_gcns[part_key] = temporal_gcn

        # Handle weight sharing for hands (support various naming conventions)
        self._setup_weight_sharing()

        self.final_dim = 256 * len(self.parts)  # Output dimension

    def _setup_weight_sharing(self):
        """Setup weight sharing between similar parts (e.g., left/right hands)"""
        # Find hand pairs to share weights
        hand_pairs = [
            ('left', 'right'),
            ('left_hand', 'right_hand'),
            ('coco_wholebody_left_hand', 'coco_wholebody_right_hand'),
            ('mediapipe_holistic_left_hand', 'mediapipe_holistic_right_hand')
        ]

        for left_key, right_key in hand_pairs:
            if left_key in self.part_keys and right_key in self.part_keys:
                # Share weights from right to left
                self.spatial_gcns[left_key] = self.spatial_gcns[right_key]
                self.temporal_gcns[left_key] = self.temporal_gcns[right_key]
                self.proj_layers[left_key] = self.proj_layers[right_key]

    def forward(self, keypoints_input, return_features=False):
        """Forward pass through the unified ST-GCN

        Args:
            keypoints_input: Either a dict with keypoints for each body part or
                           full keypoint tensor for auto-conversion
                - Dict format: {'body': Tensor(B,T,V,C), 'left': Tensor(B,T,V,C), ...}
                - Tensor format: Tensor(B,T,total_points,C) for auto-conversion
            return_features (bool): Whether to return intermediate features

        Returns:
            torch.Tensor: Concatenated features from all body parts (B, T, final_dim)
            dict (optional): Intermediate features if return_features=True
        """
        # Handle input format conversion
        if isinstance(keypoints_input, torch.Tensor) and self.auto_convert:
            if self.converter is None:
                raise ValueError("Auto-conversion enabled but no keypoint format specified")

            # Extract required parts from full keypoint tensor
            keypoints_dict = {}
            for i, part in enumerate(self.parts):
                if isinstance(part, tuple):
                    _, part_name = part
                    part_key = self.part_keys[i]
                else:
                    part_name = part
                    part_key = part

                part_keypoints = self.converter.extract_part(keypoints_input, part_name)
                keypoints_dict[part_key] = part_keypoints
        else:
            keypoints_dict = keypoints_input

        part_features = []
        intermediate_features = {} if return_features else None
        body_feat = None

        for i, part in enumerate(self.parts):
            part_key = self.part_keys[i]

            if part_key not in keypoints_dict:
                # Try alternative key names for backward compatibility
                alt_keys = [part if isinstance(part, str) else part[1],
                           str(part), part_key.split('_')[-1]]

                found_key = None
                for alt_key in alt_keys:
                    if alt_key in keypoints_dict:
                        found_key = alt_key
                        break

                if found_key is None:
                    raise ValueError(f"Missing keypoints for body part: {part_key}. "
                                   f"Available keys: {list(keypoints_dict.keys())}")

                x = keypoints_dict[found_key]
            else:
                x = keypoints_dict[part_key]

            # Input: (B, T, V, C) -> (B, C, T, V)
            batch_size, seq_len, num_joints, feat_dim = x.shape

            # Project input features
            x = self.proj_layers[part_key](x)  # (B, T, V, hidden_dim)
            x = x.permute(0, 3, 1, 2)  # (B, hidden_dim, T, V)

            if return_features:
                intermediate_features[f'{part_key}_projected'] = x

            # Spatial ST-GCN
            x = self.spatial_gcns[part_key](x)  # (B, 256, T, V)

            # Store body features for part interaction
            if self._is_body_part(part, part_key):
                body_feat = x

            # Part interaction: enhance hand/face features with body context
            elif body_feat is not None:
                x = self._apply_part_interaction(x, body_feat, part, part_key)

            if return_features:
                intermediate_features[f'{part_key}_spatial'] = x

            # Temporal ST-GCN
            x = self.temporal_gcns[part_key](x)  # (B, 256, T, V)

            if return_features:
                intermediate_features[f'{part_key}_temporal'] = x

            # Pooling across vertices
            if self.output_pooling == 'mean':
                pooled_feat = x.mean(dim=-1)  # (B, 256, T)
            elif self.output_pooling == 'max':
                pooled_feat = x.max(dim=-1)[0]  # (B, 256, T)
            else:  # no pooling
                pooled_feat = x  # (B, 256, T, V)

            pooled_feat = pooled_feat.transpose(1, 2)  # (B, T, 256)
            part_features.append(pooled_feat)

            if return_features:
                intermediate_features[f'{part_key}_pooled'] = pooled_feat

        # Concatenate features from all parts
        combined_features = torch.cat(part_features, dim=-1)  # (B, T, final_dim)

        if return_features:
            intermediate_features['combined'] = combined_features
            return combined_features, intermediate_features

        return combined_features

    def _is_body_part(self, part, part_key):
        """Check if a part represents the main body"""
        if isinstance(part, tuple):
            return part[1] == 'body'
        else:
            return part == 'body' or 'body' in part_key

    def _apply_part_interaction(self, x, body_feat, part, part_key):
        """Apply interaction between body and other parts"""
        if isinstance(part, tuple):
            part_name = part[1]
        else:
            part_name = part

        # Enhance hand/face features with body context
        if 'left' in part_name and 'hand' in part_name:
            # Left hand: add left wrist context
            try:
                x = x + body_feat[..., -2].unsqueeze(-1).detach()
            except:
                # Fallback if body doesn't have expected structure
                pass
        elif 'right' in part_name and 'hand' in part_name:
            # Right hand: add right wrist context
            try:
                x = x + body_feat[..., -1].unsqueeze(-1).detach()
            except:
                pass
        elif part_name in ['left']:
            # Legacy left hand
            try:
                x = x + body_feat[..., -2].unsqueeze(-1).detach()
            except:
                pass
        elif part_name in ['right']:
            # Legacy right hand
            try:
                x = x + body_feat[..., -1].unsqueeze(-1).detach()
            except:
                pass
        elif 'face' in part_name:
            # Face: add head context
            try:
                x = x + body_feat[..., 0].unsqueeze(-1).detach()
            except:
                pass

        return x

    def get_output_dim(self):
        """Get the output feature dimension"""
        return self.final_dim

    def get_part_info(self):
        """Get information about each body part"""
        info = {}
        for part in self.parts:
            A = self.graphs[part]
            info[part] = {
                'num_joints': A.shape[-1],
                'adjacency_partitions': A.shape[0],
                'output_dim': 256
            }
        return info


class MultiPartSTGCN(nn.Module):
    """Simplified interface for common multi-part configurations"""

    def __init__(self, config='full', **kwargs):
        """
        Args:
            config (str): Predefined configuration
                - 'full': All parts (body, left, right, face_all)
                - 'upper': Upper body (body, left, right)
                - 'hands': Hands only (left, right)
                - 'body_only': Body only
        """
        super(MultiPartSTGCN, self).__init__()

        if config == 'full':
            parts = ['body', 'left', 'right', 'face_all']
        elif config == 'upper':
            parts = ['body', 'left', 'right']
        elif config == 'hands':
            parts = ['left', 'right']
        elif config == 'body_only':
            parts = ['body']
        else:
            raise ValueError(f"Unknown config: {config}")

        self.stgcn = UniSTGCN(parts=parts, **kwargs)

    def forward(self, keypoints_dict, **kwargs):
        return self.stgcn(keypoints_dict, **kwargs)

    def get_output_dim(self):
        return self.stgcn.get_output_dim()

    def get_part_info(self):
        return self.stgcn.get_part_info()