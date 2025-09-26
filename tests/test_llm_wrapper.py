import sys
import types
import importlib.machinery
import importlib.util
from unittest import mock


def _stub_optional() -> None:
    if 'av' not in sys.modules:
        mod = types.ModuleType('av')
        mod.__spec__ = importlib.machinery.ModuleSpec('av', loader=None)
        sys.modules['av'] = mod
    if 'matplotlib' not in sys.modules:
        mod = types.ModuleType('matplotlib')
        mod.use = lambda *args, **kwargs: None
        sys.modules['matplotlib'] = mod
    if 'matplotlib.pyplot' not in sys.modules:
        sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')


_stub_optional()

import torch

from model.LLM_wrapper import LLMWithVisualPrefix


def test_llm_forward_produces_loss() -> None:
    def _fake_find_spec(name, *args, **kwargs):
        if name == 'av':
            return importlib.machinery.ModuleSpec('av', loader=None)
        return original_find_spec(name, *args, **kwargs)

    original_find_spec = importlib.util.find_spec
    with mock.patch('importlib.util.find_spec', side_effect=_fake_find_spec):
        llm = LLMWithVisualPrefix(
            'hf-internal-testing/tiny-random-gpt2',
            max_text_len=8,
            gradient_checkpointing=False,
            freeze_lm=False,
        )
    dtype = llm.model.get_input_embeddings().weight.dtype
    device = llm.device
    chunk_tokens = torch.randn(1, 2, 3, llm.hidden_size, device=device, dtype=dtype, requires_grad=True)
    token_mask = torch.tensor(
        [[[True, True, True], [True, False, False]]],
        device=device,
    )
    loss = llm(chunk_tokens, token_mask, ["hello world"])
    assert loss.shape == ()
    loss.backward()
    assert chunk_tokens.grad is not None
