"""
Keypoint format definitions and configurations.

This module provides support for different keypoint formats including
COCO-Wholebody, MediaPipe, OpenPose, and custom formats.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class KeypointFormat:
    """Base class for keypoint format definitions

    Attributes:
        name (str): Format name
        total_points (int): Total number of keypoints
        parts (dict): Dictionary mapping part names to keypoint indices
        connections (dict): Dictionary mapping part names to connection lists
        part_centers (dict): Dictionary mapping part names to center point indices
    """

    def __init__(self, name: str):
        self.name = name
        self.total_points = 0
        self.parts = {}
        self.connections = {}
        self.part_centers = {}

    def get_part_indices(self, part_name: str) -> List[int]:
        """Get keypoint indices for a specific body part"""
        return self.parts.get(part_name, [])

    def get_part_connections(self, part_name: str) -> List[List[int]]:
        """Get connection pairs for a specific body part"""
        return self.connections.get(part_name, [])

    def get_part_center(self, part_name: str) -> int:
        """Get center keypoint index for a specific body part"""
        return self.part_centers.get(part_name, 0)

    def get_part_size(self, part_name: str) -> int:
        """Get number of keypoints for a specific body part"""
        return len(self.get_part_indices(part_name))


class COCOWholebodyFormat(KeypointFormat):
    """COCO-Wholebody format (133 keypoints)

    Format breakdown:
    - Body: 17 keypoints (COCO body format) - indices 0-16
    - Face: 68 keypoints (face landmarks) - indices 23-90
    - Left hand: 21 keypoints - indices 91-111
    - Right hand: 21 keypoints - indices 112-132
    - Foot keypoints: 17-22 (usually discarded)

    Compatible with MyDataset format that includes discarded keypoints [13,14,15,16]
    """

    def __init__(self, discarded_keypoints: List[int] = None):
        super().__init__("coco_wholebody")
        self.total_points = 133

        # Support discarded keypoints from MyDataset
        self.discarded_keypoints = discarded_keypoints or [13, 14, 15, 16]

        # Define keypoint indices for each part (matching MyDataset body_info.json)
        self.parts = {
            'body': list(range(0, 17)),                 # 0-16: body keypoints
            'face': list(range(23, 91)),               # 23-90: face keypoints (68 points)
            'left_hand': list(range(91, 112)),         # 91-111: left hand (21 points)
            'right_hand': list(range(112, 133)),       # 112-132: right hand (21 points)
        }

        # Define connections for each part (before filtering)
        self.connections = {
            'body': [
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # head-shoulder
                [5, 11], [6, 12], [5, 6],                          # shoulders
                [5, 7], [6, 8], [7, 9], [8, 10],                  # arms
                [11, 13], [12, 14], [13, 15], [14, 16]             # legs (simplified)
            ],
            'face': self._get_face_connections(),
            'left_hand': self._get_hand_connections(),
            'right_hand': self._get_hand_connections(),
            'left_foot': [[0, 1], [1, 2]],
            'right_foot': [[0, 1], [1, 2]]
        }

        # Define center points
        self.part_centers = {
            'body': 0,      # nose
            'face': 30,     # nose tip (relative to face part)
            'left_hand': 0,  # wrist (relative to hand part)
            'right_hand': 0, # wrist (relative to hand part)
            'left_foot': 1,  # ankle
            'right_foot': 1  # ankle
        }

        # Apply post-processing for discarded keypoints
        self._post_init_process()

    def _get_face_connections(self) -> List[List[int]]:
        """Generate face keypoint connections (simplified)"""
        connections = []
        # Face contour (17 points: 0-16)
        for i in range(16):
            connections.append([i, i + 1])

        # Eyebrows, eyes, nose, mouth connections (simplified)
        # This is a simplified version - full face has more complex connections
        eyebrow_left = [[17, 18], [18, 19], [19, 20], [20, 21]]
        eyebrow_right = [[22, 23], [23, 24], [24, 25], [25, 26]]

        eye_left = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36]]
        eye_right = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42]]

        nose = [[27, 28], [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35]]

        mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
                [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48]]

        connections.extend(eyebrow_left + eyebrow_right + eye_left + eye_right + nose + mouth)

        # Convert to relative indices (subtract 23 since face starts at index 23)
        return [[max(0, a-23), max(0, b-23)] for a, b in connections if a >= 23 and b >= 23]

    def _get_hand_connections(self) -> List[List[int]]:
        """Generate hand keypoint connections"""
        return [
            [0, 1], [1, 2], [2, 3], [3, 4],        # thumb
            [0, 5], [5, 6], [6, 7], [7, 8],        # index
            [0, 9], [9, 10], [10, 11], [11, 12],   # middle
            [0, 13], [13, 14], [14, 15], [15, 16], # ring
            [0, 17], [17, 18], [18, 19], [19, 20]  # pinky
        ]

    def _post_init_process(self):
        """Post-initialization processing to handle discarded keypoints"""
        # Post-process: Remove discarded keypoints and remap connections
        if self.discarded_keypoints:
            original_body = self.parts['body'].copy()
            self.parts['body'] = [i for i in self.parts['body'] if i not in self.discarded_keypoints]

            # Create mapping from original indices to new indices
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.parts['body'])}

            # Remap body connections to use new indices
            new_body_connections = []
            for conn in self.connections['body']:
                i, j = conn
                if i not in self.discarded_keypoints and j not in self.discarded_keypoints:
                    new_i = index_mapping[i]
                    new_j = index_mapping[j]
                    new_body_connections.append([new_i, new_j])

            self.connections['body'] = new_body_connections

            # Update center point mapping
            if self.part_centers['body'] in index_mapping:
                self.part_centers['body'] = index_mapping[self.part_centers['body']]
            else:
                # If original center was discarded, use first available keypoint
                self.part_centers['body'] = 0


class MediaPipeHolisticFormat(KeypointFormat):
    """MediaPipe Holistic format (543 keypoints)

    Format breakdown:
    - Pose: 33 keypoints
    - Face: 468 keypoints
    - Left hand: 21 keypoints
    - Right hand: 21 keypoints
    """

    def __init__(self):
        super().__init__("mediapipe_holistic")
        self.total_points = 543

        self.parts = {
            'body': list(range(33)),                    # 0-32: pose keypoints
            'face': list(range(33, 501)),              # 33-500: face mesh (468 points)
            'left_hand': list(range(501, 522)),        # 501-521: left hand
            'right_hand': list(range(522, 543))        # 522-542: right hand
        }

        self.connections = {
            'body': self._get_mediapipe_pose_connections(),
            'face': [],  # Face mesh has too many connections, simplified
            'left_hand': self._get_hand_connections(),
            'right_hand': self._get_hand_connections()
        }

        self.part_centers = {
            'body': 0,       # nose
            'face': 1,       # approximate face center
            'left_hand': 0,  # wrist
            'right_hand': 0  # wrist
        }

    def _get_mediapipe_pose_connections(self) -> List[List[int]]:
        """MediaPipe pose connections"""
        return [
            # Upper body
            [11, 13], [13, 15], [12, 14], [14, 16],  # arms
            [11, 12],                                  # shoulders
            [5, 11], [6, 12],                         # shoulder to body
            [5, 6],                                   # nose to shoulders (simplified)
            # Lower body
            [23, 25], [25, 27], [24, 26], [26, 28],  # legs
            [23, 24],                                 # hips
            [11, 23], [12, 24]                        # torso
        ]

    def _get_hand_connections(self) -> List[List[int]]:
        """Hand connections same as COCO format"""
        return [
            [0, 1], [1, 2], [2, 3], [3, 4],        # thumb
            [0, 5], [5, 6], [6, 7], [7, 8],        # index
            [0, 9], [9, 10], [10, 11], [11, 12],   # middle
            [0, 13], [13, 14], [14, 15], [15, 16], # ring
            [0, 17], [17, 18], [18, 19], [19, 20]  # pinky
        ]


class UniSignFormat(KeypointFormat):
    """Original Uni-Sign format (133 keypoints)

    This matches the original Uni-Sign keypoint layout
    """

    def __init__(self):
        super().__init__("uni_sign")
        self.total_points = 133

        self.parts = {
            'body': list(range(9)),                    # Simplified body (9 points)
            'left': list(range(91, 112)),              # Left hand (21 points)
            'right': list(range(112, 133)),            # Right hand (21 points)
            'face_all': list(range(23, 41))            # Face subset (18 points)
        }

        self.connections = {
            'body': [
                [0, 1], [0, 2], [0, 3], [0, 4],       # center to extremities
                [3, 5], [5, 7], [4, 6], [6, 8]        # limb connections
            ],
            'left': self._get_hand_connections(),
            'right': self._get_hand_connections(),
            'face_all': self._get_face_subset_connections()
        }

        self.part_centers = {
            'body': 0,
            'left': 0,
            'right': 0,
            'face_all': 17  # Last point as center
        }

    def _get_hand_connections(self) -> List[List[int]]:
        """Hand connections"""
        return [
            [0, 1], [1, 2], [2, 3], [3, 4],        # thumb
            [0, 5], [5, 6], [6, 7], [7, 8],        # index
            [0, 9], [9, 10], [10, 11], [11, 12],   # middle
            [0, 13], [13, 14], [14, 15], [15, 16], # ring
            [0, 17], [17, 18], [18, 19], [19, 20]  # pinky
        ]

    def _get_face_subset_connections(self) -> List[List[int]]:
        """Simplified face connections for 18-point subset"""
        return [
            # Contour
            [i, i+1] for i in range(8)
        ] + [
            # Features to center connections
            [i, 17] for i in range(9, 17)
        ]


# Registry of available formats
KEYPOINT_FORMATS = {
    'coco_wholebody': COCOWholebodyFormat,
    'mediapipe_holistic': MediaPipeHolisticFormat,
    'uni_sign': UniSignFormat
}


def get_keypoint_format(format_name: str) -> KeypointFormat:
    """Get keypoint format by name"""
    if format_name not in KEYPOINT_FORMATS:
        available = list(KEYPOINT_FORMATS.keys())
        raise ValueError(f"Unknown format '{format_name}'. Available formats: {available}")

    return KEYPOINT_FORMATS[format_name]()


def create_custom_format(
    name: str,
    parts_config: Dict[str, List[int]],
    connections_config: Dict[str, List[List[int]]],
    centers_config: Dict[str, int]
) -> KeypointFormat:
    """Create a custom keypoint format

    Args:
        name: Format name
        parts_config: Dict mapping part names to keypoint indices
        connections_config: Dict mapping part names to connection pairs
        centers_config: Dict mapping part names to center indices

    Returns:
        KeypointFormat instance
    """
    format_obj = KeypointFormat(name)
    format_obj.parts = parts_config
    format_obj.connections = connections_config
    format_obj.part_centers = centers_config
    format_obj.total_points = max(max(indices) for indices in parts_config.values()) + 1

    return format_obj


def create_mydataset_compatible_format(body_info_path: str = None) -> KeypointFormat:
    """Create COCO-Wholebody format compatible with MyDataset body_info.json

    Args:
        body_info_path: Path to body_info.json file. If None, uses default discarded keypoints.

    Returns:
        COCOWholebodyFormat configured with MyDataset settings
    """
    discarded_keypoints = [13, 14, 15, 16]  # Default from MyDataset

    if body_info_path:
        import json
        try:
            with open(body_info_path, 'r') as f:
                body_info = json.load(f)
            if 'COCO_Wholebody' in body_info:
                discarded_keypoints = body_info['COCO_Wholebody'].get('DISCARDED_KEYPOINTS', [13, 14, 15, 16])
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load body_info.json ({e}), using default discarded keypoints")

    return COCOWholebodyFormat(discarded_keypoints=discarded_keypoints)