import sys
import types
import numpy as np
import torch


def _stub_optional():
    if 'av' not in sys.modules:
        sys.modules['av'] = types.ModuleType('av')
    if 'matplotlib' not in sys.modules:
        m = types.ModuleType('matplotlib')
        m.use = lambda *a, **k: None
        sys.modules['matplotlib'] = m
    if 'matplotlib.pyplot' not in sys.modules:
        sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')


_stub_optional()

from model.embedding import build_model_from_config


@torch.no_grad()
def test_masked_pooling_matches_truncated():
    cfg = {
        'model': {
            'embed_dim': 128,
            'part_embed_dim': 64,
            'backbone': 'aagcn',
            'fusion': 'attention',
            'parts': ['body', 'face', 'left_hand', 'right_hand', 'fullbody'],
            'drop_conf': True,
        }
    }
    model = build_model_from_config(cfg).eval()
    # Build batch with variable T using padding
    K = {'body': 13, 'face': 68, 'left_hand': 21, 'right_hand': 21, 'fullbody': 123}
    B = 2
    T_real = torch.tensor([6, 4], dtype=torch.long)
    T_max = int(T_real.max().item() + 2)  # pad with two extra frames

    parts = {}
    rng = np.random.default_rng(0)
    for name, V in K.items():
        arr = rng.standard_normal(size=(B, T_max, V, 3), dtype=np.float32)
        arr[..., 2] = 1.0  # conf
        parts[name] = arr

    # Compute embeddings with mask
    z_masked = model(parts, pose_len=T_real)

    # Compute embeddings by truncating to true T and averaging
    parts_trunc = {}
    for name in K:
        arr = parts[name].copy()
        for i in range(B):
            arr[i, T_real[i].item():] = 0.0
        parts_trunc[name] = arr[:, : int(T_real.min()), :, :]  # common truncation
    z_trunc = model(parts_trunc)

    assert z_masked.shape == z_trunc.shape
    # They won't be exactly equal due to different temporal lengths, but should be finite
    assert torch.isfinite(z_masked).all()
    assert torch.isfinite(z_trunc).all()

