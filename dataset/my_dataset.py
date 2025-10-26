__all__ = ['MyDataset', 'create_dataloader']

import torch
import numpy as np
import json
import time
import os
import pickle
import random

from .transform import NormalizeProcessor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Literal, List, Tuple
from scipy.interpolate import interp1d
from functools import lru_cache

from functools import lru_cache
import torch.distributed as dist

class MyDataset(Dataset):

    def __init__(
        self,
        config: Dict,
        transform: NormalizeProcessor,
        split: Literal['train', 'val', 'test'] = 'train',
        verbose: bool = False,
    ):
        self.config = config.get('dataset', config)  # Handle nested or flat config
        self.transform = transform
        self.split = split
        self.verbose = verbose
        self.dtype = config.get('dtype', 'bfloat16')

        self._memmap_files: List[np.memmap] = []
        self._annotations: Dict[str, Dict] = {}
        self.data_ids: List[str] = []

        self._load_data()
        self._split_dataset()

        self.min_reserved_ratio = self.config.get('min_reserved_ratio', 0.6)
        self._base_seed = int(self.config.get('seed', 3407))
        self._epoch = 0
        self.pad_last = self.config.get('pad_last', True)
        self.window = self.config.get('window', 32)
        self.stride = self.config.get('stride', 16)
        assert (
            0.0 < self.min_reserved_ratio <= 1.0
        ), "min_reserved_ratio must be in (0.0, 1.0]"

    def _load_data(self):
        data_dirs_cfg = self.config.get('data_dirs', self.config.get('data_dir'))
        if isinstance(data_dirs_cfg, str):
            if self.config.get('have_sub_dirs', False):
                data_dirs_cfg = [p for p in Path(data_dirs_cfg).glob('*') if p.is_dir()]
            else:
                data_dirs_cfg = [data_dirs_cfg]
        elif isinstance(data_dirs_cfg, list):
            assert all(
                isinstance(d, str) for d in data_dirs_cfg
            ), "'data_dirs' must be a list of strings."
            if self.config.get('have_sub_dirs', False):
                expanded_dirs = []
                for d in data_dirs_cfg:
                    expanded_dirs.extend(list(Path(d).glob('*')))
                data_dirs_cfg = expanded_dirs
        else:
            raise ValueError("'data_dirs' must be a string or a list of strings.")
        if not data_dirs_cfg:
            raise ValueError("'data_dirs' must be specified in the dataset config.")

        self.data_dirs = [Path(d) for d in data_dirs_cfg]
        self.data_dirs = sorted(self.data_dirs)
        print(f"Loading dataset from {len(self.data_dirs)} directories...")
        print("\nData directories:\n", self.data_dirs)
        shard_name_to_index = {p.name: i for i, p in enumerate(self.data_dirs)}

        # 1. Load all memmap files first
        for i, data_dir in enumerate(self.data_dirs):
            if not data_dir.exists():
                print(f"[Warning] Data directory {data_dir} does not exist. Skipping.")
                continue
            memmap_paths = list(data_dir.glob('*.dat')) + list(data_dir.glob('*.bin'))
            if not memmap_paths:
                print(f"[Warning] No .dat or .bin file found in {data_dir}. Skipping.")
                continue
            meta_path = data_dir / 'meta.json'
            if not meta_path.exists():
                meta_path = data_dir / 'meta.pkl'
            if not meta_path.exists():
                meta_paths = list(data_dir.glob('*.json')) + list(
                    data_dir.glob('*.pkl')
                )
                if meta_paths:
                    meta_path = meta_paths[0]
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found in {data_dir}. Expected '.json' or '.pkl'."
                )

            if meta_path.suffix == '.pkl':
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
            else:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            if not {'dtype', 'shape'}.issubset(meta):
                raise ValueError(
                    f"Metadata file {meta_path} is missing required keys 'dtype' and 'shape'. Got: {list(meta.keys())}"
                )
            dtype = np.dtype(meta['dtype'])
            shape = tuple(meta['shape'])
            self._memmap_files.append(
                np.memmap(memmap_paths[0], dtype=dtype, mode='r', shape=shape)
            )
            if self.verbose:
                print(f"Loaded memmap shard {i} from {data_dir} with shape {shape}")

        # 2. Load annotations (auto-detecting format)
        sharded_annotations = {}
        for i, data_dir in enumerate(self.data_dirs):
            annotation_path = data_dir / 'annotation.json'
            if not annotation_path.exists():
                possible_paths = list(data_dir.glob('*.json'))
                if possible_paths:
                    annotation_path = possible_paths[0]
            if annotation_path.exists():
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    shard_ann = json.load(f)
                for k, v in shard_ann.items():
                    assert isinstance(
                        v, dict
                    ), f"Annotation {annotation_path} entry for {k} is not a dict, but {type(v)}"
                    v['shard_index'] = i
                sharded_annotations.update(shard_ann)

        if sharded_annotations:
            print(
                f"Detected and loaded sharded annotations from {len(self.data_dirs)} directories."
            )
            self._annotations = sharded_annotations
        else:
            try:
                common_path = Path(os.path.commonpath([str(p) for p in self.data_dirs]))
            except ValueError:
                common_path = self.data_dirs[0].parent

            central_ann_path = common_path / 'annotation.json'
            print(
                f"No sharded annotations found. Trying to load centralized annotation from {central_ann_path}"
            )
            if not central_ann_path.exists():
                raise FileNotFoundError(
                    f"Dataset loading failed: No sharded annotation.json found, and the centralized annotation file was not found at {central_ann_path}"
                )

            with open(central_ann_path, 'r', encoding='utf-8') as f:
                central_ann = json.load(f)

            for k, v in central_ann.items():
                shard_name = v.get('shard')
                assert (
                    shard_name is not None
                ), f"Centralized annotation entry '{k}' is missing the required 'shard' key."
                shard_index = shard_name_to_index.get(shard_name)
                assert (
                    shard_index is not None
                ), f"Shard name '{shard_name}' for entry '{k}' does not match any directory name in data_dirs: {list(shard_name_to_index.keys())}"
                v['shard_index'] = shard_index
            self._annotations = central_ann
            print(
                f"Successfully loaded centralized annotations for {len(self._annotations)} samples."
            )

        self.data_ids = list(self._annotations.keys())
        random.shuffle(self.data_ids)
        print(f"Total unique samples loaded across all shards: {len(self.data_ids)}")

    def _split_dataset(self):
        # Use the first data directory as the location for the split file
        common_path = Path(os.path.commonpath([str(p) for p in self.data_dirs]))
        split_dir = common_path
        split_file = split_dir / 'split.json'

        # If split file exists, just use it
        if split_file.exists():
            with open(split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            self.data_ids = split_data.get(self.split, [])
            print(
                f"Dataset split '{self.split}': {len(self.data_ids)} samples loaded from {split_file}."
            )
            return

        # If not, only the main process should create it
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        if is_main_process:
            print(f"Split file not found. Creating new split file at {split_file}...")
            all_ids = self.data_ids  # Already shuffled from _load_data
            train_cnt = max(1, int(len(all_ids) * 0.98))
            val_cnt = int(len(all_ids) * 0.01)
            train_ids = all_ids[:train_cnt]
            val_ids = all_ids[train_cnt : train_cnt + val_cnt]
            test_ids = all_ids[train_cnt + val_cnt :]
            split_data = {"train": train_ids, "val": val_ids, "test": test_ids}

            # Atomic write
            tmp_file = split_file.with_suffix('.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False)
            os.replace(tmp_file, split_file)
            print(
                f"Created new split file with {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test samples."
            )

        if dist.is_initialized():
            dist.barrier()  # Wait for main process to write the file before others read it

        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        self.data_ids = split_data.get(self.split, [])
        print(
            f"Dataset split '{self.split}': {len(self.data_ids)} samples loaded from {split_file}."
        )

    def speed_augment(self, data: np.ndarray, speed_factor: float) -> np.ndarray:
        '''
        通过时间重采样来增强数据速度。

        :param data: 输入的骨架数据，形状为 (T, 133, 3)。T是时间帧数。
        :param speed_factor: 速度因子。> 1.0 表示加速，< 1.0 表示减速。
        :return: 经过速度增强后的新数据，形状为 (T_new, 133, 3)。
        '''
        if speed_factor <= 0:
            raise ValueError("Speed factor must be positive.")
        if speed_factor == 1.0:
            return data.copy()

        original_len = data.shape[0]
        if original_len < 2:
            return data.copy()
        new_len = int(round(original_len / speed_factor))
        if new_len < 1:  # 至少要有一帧
            new_len = 1

        original_time_axis = np.arange(original_len)
        new_time_axis = np.linspace(0, original_len - 1, new_len)

        interpolator = interp1d(
            original_time_axis,
            data,
            axis=0,
            kind='linear',
            bounds_error=False,
            fill_value=(data[0], data[-1]),  # 使用第一帧和最后一帧的数据来填充越界值
        )

        new_data = interpolator(new_time_axis)
        return new_data.astype(data.dtype)

    def mask_augment(
        self,
        data: np.ndarray,
        mask_prob: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if not (0.0 <= mask_prob <= 1.0):
            raise ValueError("Mask probability must be in the range [0.0, 1.0].")
        if mask_prob == 0.0:
            return data.copy()

        if rng is None:
            rng = np.random.default_rng()

        new_data = data.copy()
        T, N, C = new_data.shape
        for t in range(T):
            for n in range(N):
                if rng.random() < mask_prob:
                    new_data[t, n, :2] = 0.0
                    new_data[t, n, 2] = 0.0
        return new_data

    def get_pose(self, data_id: str) -> np.ndarray:
        if data_id not in self._annotations:
            raise ValueError(f"Data ID {data_id} not found in annotations.")

        annotation = self._annotations[data_id]
        shard_index = annotation.get('shard_index')
        assert (
            shard_index is not None
        ), f"Annotation for {data_id} is missing internal 'shard_index'."

        memmap_file = self._memmap_files[shard_index]
        start, end = annotation['pose_index']
        return memmap_file[start:end]

    def get_adjacency_matrix(
        self, normalize: bool = False
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        if not self.transform:
            raise ValueError("Transform processor is not set.")
        adj_mat = self.transform.gen_adjacency_matrix(
            normalize=normalize, split_part=True
        )
        if isinstance(adj_mat, dict):
            adj_mat = {
                k: torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v
                for k, v in adj_mat.items()
            }
            adj_mat = {
                k: v.to(
                    dtype=torch.bfloat16 if self.dtype == 'bfloat16' else torch.float32
                )
                for k, v in adj_mat.items()
            }
        elif isinstance(adj_mat, np.ndarray):
            adj_mat = torch.from_numpy(adj_mat)
            adj_mat = adj_mat.to(
                dtype=torch.bfloat16 if self.dtype == 'bfloat16' else torch.float32
            )
        else:
            raise ValueError(
                "Adjacency matrix must be a numpy array or a dict of tensors."
            )
        return adj_mat

    def set_epoch(self, epoch: int) -> None:
        """Allow the caller to update the epoch for RNG derivation."""
        self._epoch = int(epoch)

    def _make_rng(self, idx: int) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            base_seed = worker_info.seed
        else:
            base_seed = self._base_seed
        seed = base_seed + self._epoch * 1_000_003 + idx * 97
        return np.random.default_rng(seed)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        """
        我们保证，输出的pose的满足shape: [#Chunk, Chunk_Length, #Keypoints, Channel]
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")
        data_id = self.data_ids[idx]
        annotation = self._annotations[data_id]
        sample = {}
        rng = self._make_rng(idx)
        pose = self.get_pose(data_id)  # (T, K, C)
        T = pose.shape[0]
        augs = self.config.get('augmentations', '')
        prob = self.config.get('aug_prob', 0.5)
        if rng.random() < prob and self.split == 'train':
            if 'speed' in augs:
                speed_factor = rng.uniform(
                    self.config.get('aug_speed_min', 0.9),
                    self.config.get('aug_speed_max', 1.1),
                )
                pose = self.speed_augment(pose, speed_factor)
            if 'mask' in augs:
                mask_prob = self.config.get('aug_mask_prob', 0.05)
                if mask_prob > 0:
                    pose = self.mask_augment(pose, mask_prob, rng)

        t_prime = pose.shape[0]
        original_t_prime = t_prime
        if self.pad_last:
            # pad to fit sliding window
            if t_prime < self.window:
                pad_len = self.window - t_prime
                pad_shape = (pad_len,) + pose.shape[1:]
                pad_values = pose[-1:].repeat(pad_len, axis=0)
                pose = np.concatenate([pose, pad_values], axis=0)
            else:
                remain = (t_prime - self.window) % self.stride
                if remain != 0:
                    pad_len = self.stride - remain
                    pad_shape = (pad_len,) + pose.shape[1:]
                    pad_values = pose[-1:].repeat(pad_len, axis=0)
                    pose = np.concatenate([pose, pad_values], axis=0)
            t_prime = pose.shape[0]
        chunk_cnt = (t_prime - self.window) // self.stride + 1
        min_reserved = max(1, int(chunk_cnt * self.min_reserved_ratio))
        if chunk_cnt < min_reserved:
            chunk_cnt = min_reserved
        reversed_chunk_cnt = int(rng.integers(min_reserved, chunk_cnt + 1))
        if self.verbose:
            print(
                f"Data ID {data_id}: original frames {T}, after aug {t_prime}, chunks {chunk_cnt}, reserved {reversed_chunk_cnt}."
            )
        assert 1 <= reversed_chunk_cnt <= chunk_cnt
        reversed_t = (reversed_chunk_cnt - 1) * self.stride + self.window
        if self.split != 'train':
            reversed_t = t_prime  # use all frames for val/test
        pose = pose[:reversed_t]
        if self.transform is not None:
            pose = self.transform(
                pose
            )  # {str: np.ndarray}, keypoints after normalization, (T', K, C)
            pose = {k: torch.from_numpy(v) for k, v in pose.items()}
        else:
            pose = torch.from_numpy(pose)  # (T', K, C)

        # cut to chunks for each part
        def _slice_chunks(x: torch.Tensor) -> torch.Tensor:
            # (T', K, C) -> (N, window, K, C)
            assert (
                x.shape[0] - self.window
            ) % self.stride == 0, f"Data length {x.shape[0]} is not compatible with window {self.window} and stride {self.stride}."
            return (
                x.unfold(0, self.window, self.stride).contiguous().permute(0, 3, 1, 2)
            )

        pose = {
            k: _slice_chunks(v) for k, v in pose.items()
        }  # {str: Tensor}, (N, window, K, C)
        pose = {
            k: v.to(dtype=torch.bfloat16 if self.dtype == 'bfloat16' else torch.float32)
            for k, v in pose.items()
        }
        sample['pose'] = pose  # {str: Tensor} or Tensor
        sample['text'] = annotation['text']
        sample['gloss'] = annotation.get('gloss', '').split()  # List[str] or []
        sample['id'] = data_id
        sample['frame_cnt'] = reversed_t
        sample['adjacency_matrix'] = self.get_adjacency_matrix(
            normalize=True
        )  # Tensor or Dict[str, Tensor]
        sample['original_frame_cnt'] = original_t_prime
        sample['stride'] = self.stride
        return sample


def my_collate_fn(batch):
    """
    对于最终输出的 pose，要求：
    shape: [B, #Chunk, #Part, Chunk_Length, #Keypoints, Channel]
    """
    batch_size = len(batch)
    assert batch_size > 0 and isinstance(batch[0]["pose"], dict), "pose 必须是字典"

    body_parts = list(batch[0]["pose"].keys())

    def _get_chunk_cnt(item) -> int:
        # 获取某个样本的chunk数量，并确保所有part的chunk数量相同
        num_chunks = [item["pose"][part].shape[0] for part in body_parts]
        assert all(
            n == num_chunks[0] for n in num_chunks
        ), "所有part的chunk数量必须相同"
        return num_chunks[0]

    def _get_chunk_len(item) -> int:
        # 获取某个样本的chunk长度，并确保所有part的chunk长度相同
        chunk_lens = [item["pose"][part].shape[1] for part in body_parts]
        assert all(
            l == chunk_lens[0] for l in chunk_lens
        ), "所有part的chunk长度必须相同"
        return chunk_lens[0]

    def _stack_parts(
        poses: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        part_tensors = [
            poses[part] for part in body_parts
        ]  # List[Tensor], each (N, window, K_part, C)
        part_lens = [t.shape[2] for t in part_tensors]
        return (
            torch.cat(part_tensors, dim=2),
            part_lens,
        )  # (N, window, sum(K_part), C), List[int]

    def _calc_effecient_frames_at_last_chunk(item) -> int:
        # 计算最后一个chunk中有效帧的数量
        chunk_cnt = _get_chunk_cnt(item)
        chunk_len = _get_chunk_len(item)
        stride = item['stride']
        total_effecient_frames = item['original_frame_cnt']
        real_sent_frames = (chunk_cnt - 1) * stride + chunk_len
        if real_sent_frames <= total_effecient_frames:
            return chunk_len
        return total_effecient_frames - (chunk_cnt - 1) * stride

    padded_poses = []
    pose_lens = []
    text_list = []
    gloss_list = []
    gloss_lens = []
    last_chunk_effecient_frames = []

    max_chunks = max(_get_chunk_cnt(item) for item in batch)
    max_gloss_len = (
        max(len(item['gloss']) for item in batch) if batch[0]['gloss'] else 0
    )
    part_lens = None  # List[int], sum(K_part)
    for i, item in enumerate(batch):
        num_chunks = _get_chunk_cnt(item)
        pose_lens.append(num_chunks)
        text_list.append(item['text'])
        last_chunk_effecient_frames.append(_calc_effecient_frames_at_last_chunk(item))
        if item['gloss']:
            gloss_list.append(item['gloss'])
            gloss_lens.append(len(item['gloss']))
        else:
            gloss_list.append(None)
            gloss_lens.append(0)
        if num_chunks < max_chunks:
            last_pos = {part: item['pose'][part][-1:] for part in body_parts}
            pad_len = max_chunks - num_chunks
            pad_values = {
                part: last_pos[part].repeat(pad_len, 1, 1, 1) for part in body_parts
            }
            padded_part_poses = {
                part: torch.cat([item['pose'][part], pad_values[part]], dim=0)
                for part in body_parts
            }  # {str: Tensor}, (max_chunks, window, K, C)
        else:
            padded_part_poses = item['pose']
        stacked_pose, part_lens = _stack_parts(
            padded_part_poses
        )  # (max_chunks, window, sum(K_part), C)
        padded_poses.append(stacked_pose)
    padded_poses = torch.stack(
        padded_poses, dim=0
    )  # (B, max_chunks, window, sum(K_part), C)
    pose_lens = torch.tensor(pose_lens, dtype=torch.long)  # (B,)
    gloss_lens = torch.tensor(gloss_lens, dtype=torch.long) if any(gloss_lens) else None

    out = {
        "pose": padded_poses,  # Tensor, (B, N, window, sum(K_part), C)
        "text": text_list,  # List[str]
        "pose_len": pose_lens,  # (B,), 代表每个样本的有效chunk数量
        "gloss": gloss_list,  # List[List[str]] 或 None
        "gloss_len": gloss_lens,  # (B,) 或 None
        "parts": body_parts,  # 便于下游知道顺序
        "part_lens": part_lens,  # List[int], 对应每个part的关键点数量
        "adjacency_matrix": batch[0]["adjacency_matrix"],  # {str: Tensor}
        "last_chunk_effecient_frames": torch.tensor(
            last_chunk_effecient_frames, dtype=torch.long
        ),  # (B,), 代表每个样本最后一个有效chunk中有效帧的数量，即先使用pose_len来决定多少个chunk是有效的，然后对于最后一个有效chunk，只使用前多少帧是有效的
    }
    return out


def _seed_worker(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    seed = worker_info.seed % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def create_dataloader(
    config: Dict,
    split: Literal['train', 'val', 'test'],
    transform: NormalizeProcessor,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
    verbose: bool = True,
) -> DataLoader:
    start_time = time.time()
    dataset = MyDataset(config, transform, split=split, verbose=verbose)
    base_seed = int(config.get('seed', 3407))
    split_offset = {'train': 0, 'val': 1, 'test': 2}.get(split, 0)
    generator = torch.Generator()
    generator.manual_seed(base_seed + split_offset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate_fn,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    if verbose:
        elapsed = time.time() - start_time
        print(
            f"Created {split} dataloader with {len(dataset)} samples, batch size {batch_size}, shuffle={shuffle}, num_workers={num_workers}, pin_memory={pin_memory} in {elapsed:.2f} seconds."
        )
    return dataloader


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--data_dir',
        type=str,
        required=False,
        default='',
        help='Path to the dataset directory containing memmap and annotation files.',
    )
    args_parser.add_argument(
        '--have_sub_dirs',
        action='store_true',
        help='Whether the data_dir contains multiple sub-directories for shards.',
    )
    args = args_parser.parse_args()
    if args.data_dir:
        test_data_path = Path(args.data_dir)
    else:
        test_data_path = Path(__file__).parent / 'test_data' / 'shm_overfit'
    test_config = {
        'data_dir': str(test_data_path),
        'have_sub_dirs': args.have_sub_dirs,
        'seed': 3407,
        'window': 32,
        'stride': 16,
        'min_reserved_ratio': 0.6,
        'pad_last': True,
    }
    transform = NormalizeProcessor()
    dataset = MyDataset(test_config, transform=transform, split='train')
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    dataloader = create_dataloader(
        test_config,
        split='train',
        transform=transform,
        batch_size=2,
        shuffle=True,
        pin_memory=False,
        num_workers=0,
    )
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Pose parts:", batch['parts'])
        print("Pose shape for each part:")
        print(batch['pose'].shape)  # (B, N, window, sum(K_part), C)
        print("Text sample:", batch['text'][:2])
        print("Pose lengths:", batch['pose_len'])
        if batch['gloss'] is not None:
            print("Gloss sample:", batch['gloss'][:2])
            print("Gloss lengths:", batch['gloss_len'])
        break
