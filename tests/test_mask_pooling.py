import sys
import types
import torch

from model.parts_gcn import MultiPartGCNModel, PARTS_DEFAULT


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


def _make_pose(batch: int, chunks: int, length: int, part_lens: list[int]) -> torch.Tensor:
    joints = sum(part_lens)
    pose = torch.randn(batch, chunks, length, joints, 3, dtype=torch.float32)
    pose[..., 2] = torch.rand(batch, chunks, length, joints)
    return pose


def _make_adjacency(part_lens: list[int]) -> dict[str, torch.Tensor]:
    return {
        name: torch.eye(size, dtype=torch.float32)
        for name, size in zip(PARTS_DEFAULT[: len(part_lens)], part_lens)
    }


@torch.no_grad()
def test_masked_frames_zero_out_features() -> None:
    part_lens = [5, 10, 3, 3, 21]
    model = MultiPartGCNModel(embed_dim=16, proj_dim=8, temporal_kernel=3).eval()
    adjacency = _make_adjacency(part_lens)
    pose = _make_pose(batch=2, chunks=2, length=4, part_lens=part_lens)
    pose_len = torch.tensor([2, 1], dtype=torch.long)

    feats, frame_mask, chunk_mask = model(
        pose,
        part_lens=part_lens,
        pose_len=pose_len,
        adjacency=adjacency,
    )
    feats = feats.view(2, 2, len(part_lens), 4, -1)
    assert chunk_mask[1, 1].item() is False
    assert torch.allclose(feats[1, 1], torch.zeros_like(feats[1, 1]))


