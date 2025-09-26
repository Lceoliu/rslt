import torch

from model.parts_gcn import MultiPartGCNModel, PARTS_DEFAULT


def _make_pose(batch: int, chunks: int, length: int, part_lens: list[int]) -> torch.Tensor:
    joints = sum(part_lens)
    pose = torch.randn(batch, chunks, length, joints, 3, dtype=torch.float32)
    pose[..., 2] = torch.rand(batch, chunks, length, joints)
    return pose


def _make_adjacency(part_lens: list[int]) -> dict[str, torch.Tensor]:
    adj = {}
    for name, size in zip(PARTS_DEFAULT[: len(part_lens)], part_lens):
        adj[name] = torch.eye(size, dtype=torch.float32)
    return adj


def test_forward_shapes_and_masks() -> None:
    part_lens = [5, 10, 3, 3, 21]
    model = MultiPartGCNModel(embed_dim=32, proj_dim=16, temporal_kernel=3)
    pose = _make_pose(batch=2, chunks=3, length=4, part_lens=part_lens)
    pose_len = torch.tensor([3, 2], dtype=torch.long)
    adjacency = _make_adjacency(part_lens)

    feats, frame_mask, chunk_mask = model(
        pose,
        part_lens=part_lens,
        pose_len=pose_len,
        adjacency=adjacency,
    )

    assert feats.shape == (2 * 3, len(part_lens), 4, 32)
    assert frame_mask is not None and frame_mask.shape == (2 * 3, 4)
    assert chunk_mask is not None and chunk_mask.shape == (2, 3)
    # The final sample has only two valid chunks.
    assert chunk_mask[1].tolist() == [True, True, False]


def test_forward_without_pose_len() -> None:
    part_lens = [5, 10, 3, 3, 21]
    model = MultiPartGCNModel(embed_dim=16, proj_dim=8, temporal_kernel=3)
    pose = _make_pose(batch=1, chunks=2, length=5, part_lens=part_lens)
    adjacency = _make_adjacency(part_lens)

    feats, frame_mask, chunk_mask = model(
        pose,
        part_lens=part_lens,
        adjacency=adjacency,
    )

    assert feats.shape == (2, len(part_lens), 5, 16)
    assert frame_mask is None
    assert chunk_mask is None

    # Second pass reuses cached backbones without adjacency.
    feats2, _, _ = model(pose, part_lens=part_lens)
    torch.testing.assert_close(feats, feats2)
