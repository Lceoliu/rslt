from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def to_stgcn_input(fullbody_kpts: np.ndarray, channels: Iterable[int] = (0, 1, 2)) -> torch.Tensor:
    """Convert (T, V, 3) keypoints to (N, C, T, V) tensor for ST-GCN/AAGCN.

    Args:
        fullbody_kpts: numpy array of shape (T, V, 3) with columns [x, y, conf].
        channels: which columns to include as channels (default x,y,conf -> 0,1,2).

    Returns:
        torch.Tensor with shape (1, C, T, V), dtype float32.
    """
    assert fullbody_kpts.ndim == 3 and fullbody_kpts.shape[-1] >= max(channels) + 1
    T, V, _ = fullbody_kpts.shape
    C = len(tuple(channels))
    arr = fullbody_kpts[..., list(channels)]  # (T, V, C)
    arr = np.transpose(arr, (2, 0, 1))  # (C, T, V)
    x = torch.from_numpy(arr.astype('float32')).unsqueeze(0)  # (1, C, T, V)
    return x

