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

from ..model.embedding import build_model_from_config


def synth_parts(T=8, B=None):
    # Sizes per part after discards: body=13, face=68, left=21, right=21, fullbody=123
    K = {
        'body': 13,
        'face': 68,
        'left_hand': 21,
        'right_hand': 21,
        'fullbody': 13 + 68 + 21 + 21,
    }
    parts = {}
    for k, v in K.items():
        if B is None:
            arr = np.random.randn(T, v, 3).astype('float32')
            arr[..., 2] = np.random.rand(T, v).astype('float32')  # conf
        else:
            arr = np.random.randn(B, T, v, 3).astype('float32')
            arr[..., 2] = np.random.rand(B, T, v).astype('float32')
        parts[k] = arr
    return parts


def _cfg(backbone='aagcn', fusion='concat_mlp', D=512, Pd=128, drop_conf=True):
    cfg = {
        'model': {
            'embed_dim': D,
            'part_embed_dim': Pd,
            'backbone': backbone,
            'fusion': fusion,
            'parts': ['body', 'face', 'left_hand', 'right_hand', 'fullbody'],
            'drop_conf': drop_conf,
        }
    }
    return cfg


@torch.no_grad()
def test_aagcn_fusion_concat_shapes():
    model = build_model_from_config(
        _cfg(backbone='aagcn', fusion='concat_mlp', D=512, Pd=128)
    )
    model.eval()
    z = model(synth_parts())
    assert z.shape == (1, 512)
    assert torch.isfinite(z).all()


@torch.no_grad()
def test_stgcn_fusion_attention_shapes():
    model = build_model_from_config(
        _cfg(backbone='stgcn', fusion='attention', D=256, Pd=128)
    )
    model.eval()
    z = model(synth_parts(T=6))
    assert z.shape == (1, 256)
    assert torch.isfinite(z).all()


@torch.no_grad()
def test_batch_attention_fullbody():
    model = build_model_from_config(
        _cfg(backbone='aagcn', fusion='attention', D=320, Pd=128)
    )
    model.eval()
    parts = synth_parts(T=5, B=3)
    z = model(parts)
    assert z.shape == (3, 320)
    assert torch.isfinite(z).all()
