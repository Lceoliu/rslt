import sys
import types
import numpy as np


def _ensure_optional_deps():
    """Stub optional heavy deps so importing dataset.transform works in lean envs."""
    if 'av' not in sys.modules:
        sys.modules['av'] = types.ModuleType('av')
    if 'matplotlib' not in sys.modules:
        m = types.ModuleType('matplotlib')
        m.use = lambda *args, **kwargs: None
        sys.modules['matplotlib'] = m
    if 'matplotlib.pyplot' not in sys.modules:
        sys.modules['matplotlib.pyplot'] = types.ModuleType('matplotlib.pyplot')


try:
    from ..dataset.transform import NormalizeProcessor
except ModuleNotFoundError:
    _ensure_optional_deps()
    from dataset.transform import NormalizeProcessor


def test_indices_map_offsets_due_to_discard():
    p = NormalizeProcessor()
    # Expected kept counts per part given BODY_PARTS_INTERVALS and DISCARDED_KEYPOINTS
    kept_body = [i for i in range(0, 17) if i not in p.discarded_keypoints]
    kept_face = list(range(23, 91))
    kept_left = list(range(91, 112))
    kept_right = list(range(112, 133))

    assert len(kept_body) == 13
    assert len(kept_face) == 68
    assert len(kept_left) == 21
    assert len(kept_right) == 21

    # Offsets from compaction after discarding and gaps (17..22)
    assert p.indices_map[0] == 0
    assert p.indices_map[12] == 12
    assert p.indices_map[23] == len(kept_body)  # 13
    assert p.indices_map[91] == len(kept_body) + len(kept_face)  # 81
    assert p.indices_map[112] == len(kept_body) + len(kept_face) + len(kept_left)  # 102

    # Discarded indices should not exist in the compact index map
    for d in p.discarded_keypoints:
        assert d not in p.indices_map


def test_global_adjacency_symmetry_and_sample_edges():
    p = NormalizeProcessor()
    A = p.gen_adjacency_matrix()

    # Shape equals kept nodes across all parts
    kept_total = (
        len([i for i in range(0, 17) if i not in p.discarded_keypoints])
        + (91 - 23)
        + (112 - 91)
        + (133 - 112)
    )
    assert A.shape == (kept_total, kept_total)

    # Symmetric (undirected)
    assert np.array_equal(A, A.T)

    idx = p.indices_map
    # Body edge (both kept)
    assert A[idx[5], idx[6]] == 1.0 and A[idx[6], idx[5]] == 1.0
    # Face edge
    assert A[idx[23], idx[24]] == 1.0 and A[idx[24], idx[23]] == 1.0
    # Left hand edge
    assert A[idx[91], idx[92]] == 1.0 and A[idx[92], idx[91]] == 1.0
    # Right hand edge
    assert A[idx[112], idx[113]] == 1.0 and A[idx[113], idx[112]] == 1.0

    # Edges touching discarded nodes should not be representable
    for d in p.discarded_keypoints:
        assert d not in idx


def test_split_part_matrices_shapes_and_consistency():
    p = NormalizeProcessor()
    A_global = p.gen_adjacency_matrix()
    parts = p.gen_adjacency_matrix(split_part=True)

    # Expect per-part matrices including 'fullbody'
    for key in ['body', 'face', 'left_hand', 'right_hand', 'fullbody']:
        assert key in parts

    # Shapes per part after discarding
    assert parts['body'].shape == (13, 13)
    assert parts['face'].shape == (68, 68)
    assert parts['left_hand'].shape == (21, 21)
    assert parts['right_hand'].shape == (21, 21)
    assert parts['fullbody'].shape == A_global.shape

    # Fullbody should match the global adjacency
    assert np.array_equal(parts['fullbody'], A_global)

    # Validate a couple edges inside each part using local (part) indexing
    # Body local indices are the kept absolute indices in [0, 17)
    kept_body_abs = [i for i in range(0, 17) if i not in p.discarded_keypoints]
    body_pos = {abs_i: j for j, abs_i in enumerate(kept_body_abs)}
    assert parts['body'][body_pos[5], body_pos[6]] == 1.0

    # Face has no discards; local index = abs - 23
    assert parts['face'][23 - 23, 24 - 23] == 1.0

    # Left hand: local index = abs - 91
    assert parts['left_hand'][91 - 91, 92 - 91] == 1.0

    # Right hand: local index = abs - 112
    assert parts['right_hand'][112 - 112, 113 - 112] == 1.0


def test_enabled_parts_excludes_fullbody():
    p = NormalizeProcessor(enabled_parts=['body', 'face'])
    assert p.body_parts == ['body', 'face']
    assert p.add_fullbody_channel is False
    parts = p.gen_adjacency_matrix(split_part=True)
    assert list(parts.keys()) == ['body', 'face']
    assert 'fullbody' not in parts
