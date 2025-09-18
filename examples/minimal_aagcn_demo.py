"""Minimal demo: use AAGCNBackbone with COCO-WholeBody adjacency.

This avoids installing mmcv/mmengine. Requires only torch and numpy.
"""
import sys
import types
import numpy as np
import torch

# Optional: stub heavy optional deps used by NormalizeProcessor import
for mod in ("av", "matplotlib", "matplotlib.pyplot"):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)
if hasattr(sys.modules["matplotlib"], "use") is False:
    sys.modules["matplotlib"].use = lambda *a, **k: None

from dataset.transform import NormalizeProcessor
from model.backbones.aagcn_minimal import AAGCNBackbone, build_adjacency_from_numpy
from model.utils.pose_pack import to_stgcn_input


def main():
    proc = NormalizeProcessor()
    A_np = proc.gen_adjacency_matrix(normalize=False)  # global adjacency (kept joints)
    A = build_adjacency_from_numpy(A_np)

    # Fake pose sequence: T frames, V joints, channels [x,y,conf]
    V = len(proc.all_indices)
    T = 20
    kpts = np.random.randn(T, V, 3).astype("float32")
    kpts[..., 2] = 1.0  # conf

    x = to_stgcn_input(kpts)  # (1, 3, T, V)

    model = AAGCNBackbone(in_channels=3, A=A, embed_dim=256)
    with torch.no_grad():
        emb = model(x)
    print("Embedding shape:", emb.shape)


if __name__ == "__main__":
    main()

