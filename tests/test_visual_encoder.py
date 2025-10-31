import torch

from model.visual_encoder import VisualEncoder
from model.parts_gcn import PARTS_DEFAULT


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


def test_visual_encoder_shapes_and_masks() -> None:
    part_lens = [5, 10, 3, 3, 21]
    encoder = VisualEncoder(
        gcn_embed_dim=16,
        gcn_proj_dim=8,
        tokens_per_chunk=0,
        llm_dim=24,
        sampling_stride=2,
    )
    pose = _make_pose(batch=2, chunks=3, length=4, part_lens=part_lens)
    pose_len = torch.tensor([3, 2], dtype=torch.long)
    adjacency = _make_adjacency(part_lens)

    tokens, token_mask, chunk_mask = encoder(
        pose,
        part_lens=part_lens,
        pose_len=pose_len,
        adjacency=adjacency,
    )

    assert tokens.shape == (2, 3, 2, 24)
    assert token_mask.shape == (2, 3, 2)
    assert chunk_mask.shape == (2, 3)
    # Last chunk of second sample should be padding
    assert token_mask[1, 2].tolist() == [False, False]
    assert torch.all(tokens[1, 2] == 0)


def test_visual_encoder_no_pose_len() -> None:
    part_lens = [5, 10, 3, 3, 21]
    encoder = VisualEncoder(
        gcn_embed_dim=8,
        gcn_proj_dim=4,
        tokens_per_chunk=0,
        llm_dim=12,
        sampling_stride=2,
    )
    pose = _make_pose(batch=1, chunks=2, length=4, part_lens=part_lens)
    adjacency = _make_adjacency(part_lens)

    tokens, token_mask, chunk_mask = encoder(
        pose,
        part_lens=part_lens,
        adjacency=adjacency,
    )

    assert tokens.shape == (1, 2, 2, 12)
    assert torch.all(token_mask)
    assert torch.all(chunk_mask)
