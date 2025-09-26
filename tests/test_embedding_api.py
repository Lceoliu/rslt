import sys
import types
import numpy as np
import pytest

from ..model.embedding import build_visual_encoder, compute_embedding_from_parts


def _stub_optional() -> None:
    if 'av' not in sys.modules:
        sys.modules['av'] = types.ModuleType('av')
    if 'matplotlib' not in sys.modules:
        mod = types.ModuleType('matplotlib')
        mod.use = lambda *args, **kwargs: None
        sys.modules['matplotlib'] = mod
    if 'matplotlib.pyplot' not in sys.modules:
        sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')


_stub_optional()


def test_build_visual_encoder_appends_fullbody() -> None:
    cfg = {'model': {'parts': ['body', 'face'], 'drop_conf': False}}
    encoder = build_visual_encoder(cfg, llm_dim=128)
    assert encoder.parts == ('body', 'face', 'fullbody')
    assert encoder.multipart.drop_conf is False
    assert encoder.transformer.model_dim == 128


def test_build_visual_encoder_defaults() -> None:
    encoder = build_visual_encoder({'model': {}}, llm_dim=64)
    assert encoder.parts[-1] == 'fullbody'
    assert encoder.multipart.embed_dim == 256
    assert encoder.transformer.num_tokens == 4


def test_compute_embedding_from_parts_raises() -> None:
    parts = {'body': np.zeros((2, 3, 3), dtype='float32')}
    with pytest.raises(NotImplementedError):
        compute_embedding_from_parts(parts, {'model': {}})
