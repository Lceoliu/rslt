"""
Graph construction utilities for skeletal data modeling.

This module provides graph topology definitions for different body parts
including body, hands, and face keypoints. Supports multiple keypoint formats
including COCO-Wholebody, MediaPipe, and custom formats.
"""

import torch
import numpy as np
from .keypoint_formats import KeypointFormat, get_keypoint_format


class Graph:
    """Graph structure for modeling skeletal keypoints

    Supports different body part layouts and partitioning strategies for
    spatio-temporal graph convolutional networks. Can work with multiple
    keypoint formats including COCO-Wholebody, MediaPipe, and custom formats.

    Args:
        layout (string or tuple): Body part layout specification
            - If string: Legacy format ('body', 'left', 'right', 'face_all')
            - If tuple: (format_name, part_name) e.g., ('coco_wholebody', 'body')
        strategy (string): Adjacency matrix partitioning strategy
            - 'uniform': Uniform labeling
            - 'distance': Distance partitioning
            - 'spatial': Spatial configuration
        max_hop (int): Maximum distance between connected nodes
        dilation (int): Controls spacing between kernel points
        keypoint_format (KeypointFormat, optional): Custom keypoint format object
    """

    def __init__(self, layout='body', strategy='uniform', max_hop=1, dilation=1,
                 keypoint_format=None):
        self.max_hop = max_hop
        self.dilation = dilation
        self.keypoint_format = keypoint_format

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return f"Graph(layout={self.layout}, nodes={self.num_node}, strategy={self.strategy})"

    def get_edge(self, layout):
        """Define edge connections based on body part layout"""
        self.layout = layout

        # Handle new format specification
        if isinstance(layout, tuple):
            format_name, part_name = layout
            if self.keypoint_format is None:
                self.keypoint_format = get_keypoint_format(format_name)
            self._build_from_format(part_name)
            return

        # Handle legacy format
        self._build_legacy_format(layout)

    def _build_from_format(self, part_name):
        """Build graph from keypoint format specification"""
        if self.keypoint_format is None:
            raise ValueError("keypoint_format must be provided for format-based layout")

        # Get part information
        part_size = self.keypoint_format.get_part_size(part_name)
        connections = self.keypoint_format.get_part_connections(part_name)
        center = self.keypoint_format.get_part_center(part_name)

        if part_size == 0:
            raise ValueError(f"Part '{part_name}' not found in format '{self.keypoint_format.name}'")

        # Set graph properties
        self.num_node = part_size
        self.center = center

        # Build edge list
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = connections

        self.edge = self_link + neighbor_link
        self.part_name = part_name

    def _build_legacy_format(self, layout):
        """Build graph using legacy layout specification"""
        if layout in ['left', 'right']:
            # Hand keypoints: 21 joints
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]

            # Hand skeleton connections
            neighbor_1base = [
                [0, 1], [1, 2], [2, 3], [3, 4],    # thumb
                [0, 5], [5, 6], [6, 7], [7, 8],    # index
                [0, 9], [9, 10], [10, 11], [11, 12],  # middle
                [0, 13], [13, 14], [14, 15], [15, 16],  # ring
                [0, 17], [17, 18], [18, 19], [19, 20]   # pinky
            ]
            self.edge = self_link + neighbor_1base
            self.center = 0

        elif layout == 'body':
            # Body keypoints: 9 joints
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]

            # Body skeleton connections
            neighbor_1base = [
                [0, 1], [0, 2], [0, 3], [0, 4],  # head to shoulders/hips
                [3, 5], [5, 7],  # left arm
                [4, 6], [6, 8],  # right arm
            ]
            self.edge = self_link + neighbor_1base
            self.center = 0

        elif layout == 'face_all':
            # Face keypoints: 18 joints (9 contour + 8 features + 1 center)
            self.num_node = 9 + 8 + 1  # 18 total
            self_link = [(i, i) for i in range(self.num_node)]

            # Face skeleton connections
            neighbor_1base = (
                [[i, i + 1] for i in range(8)] +  # contour chain
                [[i, i + 1] for i in range(9, 16)] +  # feature chain
                [[16, 9]] +  # connect feature to contour
                [[17, i] for i in range(17)]  # center connects to all
            )
            self.edge = self_link + neighbor_1base
            self.center = self.num_node - 1

        else:
            raise ValueError(f"Unsupported layout: {layout}")

        self.part_name = layout

    def get_adjacency(self, strategy):
        """Generate adjacency matrix based on partitioning strategy"""
        self.strategy = strategy
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))

        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A

        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A

        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))

                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (self.hop_dis[j, self.center] ==
                                self.hop_dis[i, self.center]):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (self.hop_dis[j, self.center] >
                                  self.hop_dis[i, self.center]):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]

                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)

            A = np.stack(A)
            self.A = A

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")


def get_hop_distance(num_node, edge, max_hop=1):
    """Calculate hop distances between all pairs of nodes"""
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # Compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0

    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    return hop_dis


def normalize_digraph(A):
    """Normalize adjacency matrix for directed graph"""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))

    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)

    AD = np.dot(A, Dn)
    return AD