"""
Uni-GCN: Spatio-Temporal Graph Convolutional Networks for Sign Language Understanding

This module provides standalone ST-GCN implementation with support for multiple
keypoint formats including COCO-Wholebody, MediaPipe, and custom formats.
"""

from .graph import Graph
from .stgcn_block import GCN_unit, STGCN_block, STGCNChain, get_stgcn_chain
from .uni_stgcn import UniSTGCN, MultiPartSTGCN
from .keypoint_formats import (
    KeypointFormat,
    COCOWholebodyFormat,
    MediaPipeHolisticFormat,
    UniSignFormat,
    get_keypoint_format,
    create_custom_format,
    create_mydataset_compatible_format,
    KEYPOINT_FORMATS
)
from .format_converter import (
    KeypointConverter,
    BatchConverter,
    create_coco_wholebody_converter,
    create_mediapipe_converter,
    convert_coco_to_unisign,
    convert_mediapipe_to_unisign,
    create_part_extractor
)
from .dataset_adapter import (
    DatasetSTGCN,
    MultiTaskSTGCN,
    SignLanguageTrainer,
    create_sign_recognition_model,
    create_sign_translation_model,
    create_multi_task_model
)

__version__ = "2.1.0"
__all__ = [
    # Core ST-GCN components
    "Graph",
    "GCN_unit",
    "STGCN_block",
    "STGCNChain",
    "get_stgcn_chain",

    # Main models
    "UniSTGCN",
    "MultiPartSTGCN",

    # Dataset integration
    "DatasetSTGCN",
    "MultiTaskSTGCN",
    "SignLanguageTrainer",
    "create_sign_recognition_model",
    "create_sign_translation_model",
    "create_multi_task_model",

    # Keypoint formats
    "KeypointFormat",
    "COCOWholebodyFormat",
    "MediaPipeHolisticFormat",
    "UniSignFormat",
    "get_keypoint_format",
    "create_custom_format",
    "create_mydataset_compatible_format",
    "KEYPOINT_FORMATS",

    # Format conversion
    "KeypointConverter",
    "BatchConverter",
    "create_coco_wholebody_converter",
    "create_mediapipe_converter",
    "convert_coco_to_unisign",
    "convert_mediapipe_to_unisign",
    "create_part_extractor"
]