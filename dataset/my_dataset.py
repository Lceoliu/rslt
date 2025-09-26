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
    ):
        self.config = config
        self.transform = transform
        self.split = split

        self._memmap_data = None
        self._annotations = None
        self.data_ids = []
        self._memmap_data_type = np.float32
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
        self._data_dir = Path(self.config['data_dir'])
        assert (
            self._data_dir.exists()
        ), f"Data directory {self._data_dir} does not exist."
        memmap_path = list(self._data_dir.glob('*.dat'))
        assert (
            len(memmap_path) == 1
        ), f"Expected one .dat file in {self._data_dir}, found {len(memmap_path)}."
        meta_path = self._data_dir / 'meta.json'
        if not meta_path.exists():
            meta_path = self._data_dir / 'meta.pkl'
        assert meta_path.exists(), f"Meta file {meta_path} does not exist."
        annotation_path = self._data_dir / 'annotation.json'
        assert (
            annotation_path.exists()
        ), f"Annotation file {annotation_path} does not exist."
        print(f"Start loading dataset from {self._data_dir} ...")
        if meta_path.suffix == '.pkl':
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
        else:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        with open(annotation_path, 'r', encoding='utf-8') as f:
            self._annotations = json.load(f)
        try:
            self._memmap_data_type = np.dtype(meta['dtype'])
        except KeyError as e:
            raise ValueError(f"Missing key in meta.json: {e}")
        except Exception as e:
            if meta['dtype'] == 'float16':
                self._memmap_data_type = np.float16
            elif meta['dtype'] == 'float32':
                self._memmap_data_type = np.float32
            elif meta['dtype'] == 'float64':
                self._memmap_data_type = np.float64
        shape = list(meta['shape'])
        assert len(shape) == 3, f"Expected shape to be of length 3, got {len(shape)}."
        self._memmap_data = np.memmap(
            memmap_path[0],
            dtype=self._memmap_data_type,
            mode='r',
            shape=tuple(shape),
        )
        self.data_ids = list(self._annotations.keys())
        print(
            f"Loaded {len(self.data_ids)} samples from {self._data_dir}, memmap shape: {self._memmap_data.shape}, dtype: {self._memmap_data_type}"
        )

    def _split_dataset(self):
        self._data_dir = Path(self.config['data_dir'])
        split_file = self._data_dir / 'split.json'
        lock_file = split_file.with_suffix('.lock')

        def _atomic_write_json(path: Path, data: dict):
            tmp = path.with_suffix(path.suffix + '.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)  # 原子替换

        # 已存在则直接读
        if split_file.exists():
            with open(split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            self.data_ids = split_data.get(self.split, [])
            print(f"Dataset split '{self.split}': {len(self.data_ids)} samples.")
            return

        # 并发安全的创建：用 O_EXCL 获取简单文件锁，其他进程轮询等待
        got_lock = False
        lock_fd = None
        try:
            try:
                lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                got_lock = True
            except FileExistsError:
                got_lock = False

            if got_lock:
                all_ids = list(self._annotations.keys())
                # ensure must have one training sample
                train_cnt = max(1, int(len(all_ids) * 0.8))
                train_ids = all_ids[:train_cnt]
                val_ids = all_ids[train_cnt : train_cnt + int(len(all_ids) * 0.1)]
                test_ids = all_ids[train_cnt + int(len(all_ids) * 0.1) :]
                split_data = {"train": train_ids, "val": val_ids, "test": test_ids}
                _atomic_write_json(split_file, split_data)
            else:
                # 等待写入完成（最多 ~300s）
                for _ in range(6000):
                    if split_file.exists():
                        break
                    time.sleep(0.05)
                if not split_file.exists():
                    raise TimeoutError(
                        f"Timed out waiting for {split_file} to be created."
                    )
        finally:
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except Exception:
                    pass
            if got_lock:
                try:
                    os.remove(lock_file)
                except Exception:
                    pass

        # 若此时已经初始化分布式，可再同步一下（可选）
        try:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        self.data_ids = split_data.get(self.split, [])
        print(f"Dataset split '{self.split}': {len(self.data_ids)} samples.")

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
        start, end = self._annotations[data_id]['pose_index']
        return self._memmap_data[start:end]

    def get_adjacency_matrix(self, normalize: bool = False) -> np.ndarray:
        if not self.transform:
            raise ValueError("Transform processor is not set.")
        return self.transform.gen_adjacency_matrix(normalize=normalize, split_part=True)

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

        sample['pose'] = pose  # {str: Tensor} or Tensor
        sample['text'] = annotation['text']
        sample['gloss'] = annotation.get('gloss', '').split()  # List[str] or []
        sample['id'] = data_id
        sample['frame_cnt'] = reversed_t
        sample['adjacency_matrix'] = {
            k: torch.from_numpy(v)
            for k, v in self.get_adjacency_matrix(normalize=True).items()
        }  # {str: np.ndarray}
        return sample


def my_collate_fn(batch):
    """
    对于最终输出的 pose，要求：
    shape: [B, #Chunk, #Part, Chunk_Length, #Keypoints, Channel]
    """
    batch_size = len(batch)
    assert batch_size > 0 and isinstance(batch[0]["pose"], dict), "pose 必须是字典"

    body_parts = list(batch[0]["pose"].keys())

    def _get_chunk_lens(item) -> int:
        num_chunks = [item["pose"][part].shape[0] for part in body_parts]
        assert all(
            n == num_chunks[0] for n in num_chunks
        ), "所有part的chunk数量必须相同"
        return num_chunks[0]

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

    padded_poses = []
    pose_lens = []
    text_list = []
    gloss_list = []
    gloss_lens = []

    max_chunks = max(_get_chunk_lens(item) for item in batch)
    max_gloss_len = (
        max(len(item['gloss']) for item in batch) if batch[0]['gloss'] else 0
    )
    part_lens = None  # List[int], sum(K_part)
    for i, item in enumerate(batch):
        num_chunks = _get_chunk_lens(item)
        pose_lens.append(num_chunks)
        text_list.append(item['text'])
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
    dataset = MyDataset(config, transform, split=split)
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
    test_data_path = Path(__file__).parent / 'test_data' / 'shm_overfit'
    test_config = {
        'data_dir': str(test_data_path),
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
