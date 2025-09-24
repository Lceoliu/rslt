"""
Keypoint format conversion utilities.

This module provides tools to convert between different keypoint formats
and extract specific body parts from full-body keypoint data.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from .keypoint_formats import KeypointFormat, get_keypoint_format


class KeypointConverter:
    """Converter between different keypoint formats"""

    def __init__(self, source_format: Union[str, KeypointFormat],
                 target_format: Union[str, KeypointFormat] = None):
        """
        Args:
            source_format: Source keypoint format name or object
            target_format: Target keypoint format name or object (optional)
        """
        if isinstance(source_format, str):
            self.source_format = get_keypoint_format(source_format)
        else:
            self.source_format = source_format

        if target_format is not None:
            if isinstance(target_format, str):
                self.target_format = get_keypoint_format(target_format)
            else:
                self.target_format = target_format
        else:
            self.target_format = None

    def extract_part(self, keypoints: Union[torch.Tensor, np.ndarray],
                     part_name: str) -> Union[torch.Tensor, np.ndarray]:
        """Extract specific body part keypoints from full keypoint data

        Args:
            keypoints: Full keypoint tensor/array of shape (..., total_points, features)
            part_name: Name of body part to extract

        Returns:
            Extracted part keypoints of shape (..., part_points, features)
        """
        part_indices = self.source_format.get_part_indices(part_name)
        if not part_indices:
            raise ValueError(f"Part '{part_name}' not found in format '{self.source_format.name}'")

        return keypoints[..., part_indices, :]

    def extract_multiple_parts(self, keypoints: Union[torch.Tensor, np.ndarray],
                              part_names: List[str]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Extract multiple body parts from full keypoint data

        Args:
            keypoints: Full keypoint tensor/array
            part_names: List of part names to extract

        Returns:
            Dictionary mapping part names to extracted keypoints
        """
        result = {}
        for part_name in part_names:
            result[part_name] = self.extract_part(keypoints, part_name)
        return result

    def normalize_part(self, part_keypoints: Union[torch.Tensor, np.ndarray],
                       part_name: str, method: str = 'center') -> Union[torch.Tensor, np.ndarray]:
        """Normalize part keypoints relative to center point

        Args:
            part_keypoints: Part keypoints of shape (..., num_joints, features)
            part_name: Name of the body part
            method: Normalization method ('center', 'bbox', 'none')

        Returns:
            Normalized keypoints
        """
        if method == 'none':
            return part_keypoints

        if method == 'center':
            center_idx = self.source_format.get_part_center(part_name)
            if isinstance(part_keypoints, torch.Tensor):
                center_point = part_keypoints[..., center_idx:center_idx+1, :2]
                normalized = part_keypoints.clone()
                normalized[..., :2] = normalized[..., :2] - center_point
            else:
                center_point = part_keypoints[..., center_idx:center_idx+1, :2]
                normalized = part_keypoints.copy()
                normalized[..., :2] = normalized[..., :2] - center_point
            return normalized

        elif method == 'bbox':
            coords = part_keypoints[..., :2]  # x, y coordinates
            if isinstance(coords, torch.Tensor):
                min_coords = torch.min(coords, dim=-2, keepdim=True)[0]
                max_coords = torch.max(coords, dim=-2, keepdim=True)[0]
                center = (min_coords + max_coords) / 2
                scale = torch.max(max_coords - min_coords, dim=-1, keepdim=True)[0]

                normalized = part_keypoints.clone()
                normalized[..., :2] = (coords - center) / (scale + 1e-8)
            else:
                min_coords = np.min(coords, axis=-2, keepdims=True)
                max_coords = np.max(coords, axis=-2, keepdims=True)
                center = (min_coords + max_coords) / 2
                scale = np.max(max_coords - min_coords, axis=-1, keepdims=True)

                normalized = part_keypoints.copy()
                normalized[..., :2] = (coords - center) / (scale + 1e-8)

            return normalized

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def convert_format(self, keypoints: Union[torch.Tensor, np.ndarray],
                       part_mapping: Dict[str, str]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Convert keypoints from source to target format

        Args:
            keypoints: Source keypoints
            part_mapping: Mapping from target part names to source part names

        Returns:
            Dictionary of converted keypoints for each target part
        """
        if self.target_format is None:
            raise ValueError("Target format not specified")

        result = {}
        for target_part, source_part in part_mapping.items():
            # Extract source part
            source_keypoints = self.extract_part(keypoints, source_part)

            # Get expected size for target part
            target_size = self.target_format.get_part_size(target_part)
            source_size = source_keypoints.shape[-2]

            if source_size == target_size:
                # Direct mapping
                result[target_part] = source_keypoints
            elif source_size > target_size:
                # Subsample (take first N points)
                result[target_part] = source_keypoints[..., :target_size, :]
            else:
                # Pad with zeros or duplicate last point
                if isinstance(keypoints, torch.Tensor):
                    pad_shape = list(source_keypoints.shape)
                    pad_shape[-2] = target_size - source_size
                    padding = torch.zeros(pad_shape, dtype=keypoints.dtype, device=keypoints.device)
                    result[target_part] = torch.cat([source_keypoints, padding], dim=-2)
                else:
                    pad_shape = list(source_keypoints.shape)
                    pad_shape[-2] = target_size - source_size
                    padding = np.zeros(pad_shape, dtype=keypoints.dtype)
                    result[target_part] = np.concatenate([source_keypoints, padding], axis=-2)

        return result


# Predefined conversion mappings
COCO_TO_UNISIGN_MAPPING = {
    'body': 'body',          # Map COCO body to Uni-Sign body (will be subsampled)
    'left': 'left_hand',     # COCO left hand to Uni-Sign left
    'right': 'right_hand',   # COCO right hand to Uni-Sign right
    'face_all': 'face'       # COCO face to Uni-Sign face (will be subsampled)
}

MEDIAPIPE_TO_UNISIGN_MAPPING = {
    'body': 'body',          # MediaPipe pose to Uni-Sign body
    'left': 'left_hand',     # MediaPipe left hand to Uni-Sign left
    'right': 'right_hand',   # MediaPipe right hand to Uni-Sign right
    'face_all': 'face'       # MediaPipe face to Uni-Sign face
}


def create_coco_wholebody_converter() -> KeypointConverter:
    """Create converter for COCO-Wholebody format"""
    return KeypointConverter('coco_wholebody')


def create_mediapipe_converter() -> KeypointConverter:
    """Create converter for MediaPipe Holistic format"""
    return KeypointConverter('mediapipe_holistic')


def convert_coco_to_unisign(keypoints: Union[torch.Tensor, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """Convert COCO-Wholebody keypoints to Uni-Sign format

    Args:
        keypoints: COCO-Wholebody keypoints (..., 133, 3)

    Returns:
        Dictionary with Uni-Sign format parts
    """
    converter = KeypointConverter('coco_wholebody', 'uni_sign')
    return converter.convert_format(keypoints, COCO_TO_UNISIGN_MAPPING)


def convert_mediapipe_to_unisign(keypoints: Union[torch.Tensor, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """Convert MediaPipe Holistic keypoints to Uni-Sign format

    Args:
        keypoints: MediaPipe keypoints (..., 543, 3)

    Returns:
        Dictionary with Uni-Sign format parts
    """
    converter = KeypointConverter('mediapipe_holistic', 'uni_sign')
    return converter.convert_format(keypoints, MEDIAPIPE_TO_UNISIGN_MAPPING)


class BatchConverter:
    """Batch converter for processing multiple sequences efficiently"""

    def __init__(self, converter: KeypointConverter):
        self.converter = converter

    def extract_parts_batch(self, keypoints_batch: Union[torch.Tensor, np.ndarray],
                           part_names: List[str]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Extract parts from a batch of keypoint sequences

        Args:
            keypoints_batch: Batch of keypoints (batch_size, seq_len, total_points, features)
            part_names: List of part names to extract

        Returns:
            Dictionary with extracted parts for the whole batch
        """
        result = {}
        for part_name in part_names:
            part_indices = self.converter.source_format.get_part_indices(part_name)
            if not part_indices:
                raise ValueError(f"Part '{part_name}' not found")

            result[part_name] = keypoints_batch[..., part_indices, :]

            # Apply normalization if it's hand or face
            if part_name in ['left_hand', 'right_hand', 'left', 'right', 'face', 'face_all']:
                result[part_name] = self.converter.normalize_part(
                    result[part_name], part_name, method='center'
                )

        return result


def create_part_extractor(format_name: str, part_names: List[str]):
    """Create a function to extract specific parts from keypoints

    Args:
        format_name: Keypoint format name
        part_names: List of parts to extract

    Returns:
        Function that takes keypoints and returns extracted parts
    """
    converter = KeypointConverter(format_name)
    batch_converter = BatchConverter(converter)

    def extract_parts(keypoints):
        return batch_converter.extract_parts_batch(keypoints, part_names)

    return extract_parts


# Example usage functions
def demo_coco_extraction():
    """Demonstrate COCO-Wholebody keypoint extraction"""
    # Simulate COCO-Wholebody keypoints
    batch_size, seq_len = 2, 64
    coco_keypoints = np.random.randn(batch_size, seq_len, 133, 3)

    # Create converter
    converter = create_coco_wholebody_converter()

    # Extract specific parts
    parts = converter.extract_multiple_parts(coco_keypoints, ['body', 'left_hand', 'right_hand', 'face'])

    print("COCO-Wholebody extraction demo:")
    for part_name, part_kpts in parts.items():
        print(f"  {part_name}: {part_kpts.shape}")

    # Convert to Uni-Sign format
    unisign_parts = convert_coco_to_unisign(coco_keypoints)
    print("\nConverted to Uni-Sign format:")
    for part_name, part_kpts in unisign_parts.items():
        print(f"  {part_name}: {part_kpts.shape}")


if __name__ == "__main__":
    demo_coco_extraction()