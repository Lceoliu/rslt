__all__ = ['MyDataset', 'create_dataloader']

import torch
import numpy as np
import json
import time

from .transform import NormalizeProcessor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Literal, List, Tuple
from scipy.interpolate import interp1d
from functools import lru_cache


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
        assert meta_path.exists(), f"Meta file {meta_path} does not exist."
        annotation_path = self._data_dir / 'annotation.json'
        assert (
            annotation_path.exists()
        ), f"Annotation file {annotation_path} does not exist."
        print(f"Start loading dataset from {self._data_dir} ...")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        with open(annotation_path, 'r') as f:
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

    def mask_augment(self, data: np.ndarray, mask_prob: float) -> np.ndarray:
        '''
        随机遮挡部分骨骼数据以进行数据增强。

        :param data: 输入的骨架数据，形状为 (T, 133, 3)。T是时间帧数。
        :param mask_prob: 遮挡概率，范围在 [0.0, 1.0] 之间。
        :return: 经过遮挡增强后的新数据，形状为 (T, 133, 3)。
        '''
        if not (0.0 <= mask_prob <= 1.0):
            raise ValueError("Mask probability must be in the range [0.0, 1.0].")
        if mask_prob == 0.0:
            return data.copy()

        new_data = data.copy()
        T, N, C = new_data.shape
        for t in range(T):
            for n in range(N):
                if np.random.rand() < mask_prob:
                    new_data[t, n, :] = 0.0
        return new_data

    @lru_cache(maxsize=128)
    def get_pose(self, data_id: str) -> np.ndarray:
        if data_id not in self._annotations:
            raise ValueError(f"Data ID {data_id} not found in annotations.")
        start, end = self._annotations[data_id]['pose_index']
        return self._memmap_data[start:end]

    def get_adjacency_matrix(self, normalize: bool = False) -> np.ndarray:
        if not self.transform:
            raise ValueError("Transform processor is not set.")
        return self.transform.gen_adjacency_matrix(normalize=normalize, split_part=True)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        annotation = self._annotations[data_id]
        sample = {}
        pose = self.get_pose(data_id)  # (T, K, C)
        T = pose.shape[0]
        augs = self.config.get('augmentations', '')
        prob = self.config.get('aug_prob', 0.5)
        if np.random.rand() < prob and self.split == 'train':
            if 'speed' in augs:
                speed_factor = np.random.uniform(
                    self.config.get('aug_speed_min', 0.9),
                    self.config.get('aug_speed_max', 1.1),
                )
                pose = self.speed_augment(pose, speed_factor)
            if 'mask' in augs:
                mask_prob = self.config.get('aug_mask_prob', 0.05)
                pose = self.mask_augment(pose, mask_prob)

        if self.transform is not None:
            pose = self.transform(
                pose
            )  # {str: np.ndarray}, keypoints after normalization
            pose = {k: torch.from_numpy(v) for k, v in pose.items()}
        else:
            pose = torch.from_numpy(pose)  # (T, K, C)
        sample['pose'] = pose  # {str: Tensor} or Tensor
        sample['text'] = annotation['text']
        sample['gloss'] = annotation.get('gloss', '').split()
        sample['frame_cnt'] = T
        sample['adjacency_matrix'] = {
            k: torch.from_numpy(v)
            for k, v in self.get_adjacency_matrix(normalize=True).items()
        }  # {str: np.ndarray}
        return sample


def my_collate_fn(batch):
    batch_size = len(batch)
    assert batch_size > 0 and isinstance(batch[0]["pose"], dict), "pose 必须是字典"

    body_parts = list(batch[0]["pose"].keys())

    def _get_T(item) -> int:
        any_part = next(iter(item["pose"].values()))
        return any_part.shape[0]

    # 维度 (K, C)
    part_dims: Dict[str, Tuple[int, int]] = {
        part: batch[0]["pose"][part].shape[1:] for part in body_parts
    }

    for item in batch:
        T0 = _get_T(item)
        for part in body_parts:
            assert item["pose"][part].shape[0] == T0, f"{part} 的 T 不一致"
            assert (
                item["pose"][part].shape[1:] == part_dims[part]
            ), f"{part} 的 (K,C) 不一致"

    max_T = max(_get_T(item) for item in batch)

    # 按 part 分别做 pad: (B, max_T, K, C)
    padded_poses: Dict[str, torch.Tensor] = {}
    for part in body_parts:
        K, C = part_dims[part]
        dtype = batch[0]["pose"][part].dtype
        padded = torch.zeros((batch_size, max_T, K, C), dtype=dtype)
        for i, item in enumerate(batch):
            T = _get_T(item)
            padded[i, :T] = item["pose"][part]
        padded_poses[part] = padded

    # 文本与 gloss 处理
    text_list: List[str] = [item["text"] for item in batch]
    has_gloss = ("gloss" in batch[0]) and (batch[0]["gloss"] is not None)
    if has_gloss:
        max_gloss_len = max(len(item["gloss"]) for item in batch)
        gloss_list = []
        for item in batch:
            g = item["gloss"]
            g = g + ["<pad>"] * (max_gloss_len - len(g))
            gloss_list.append(g)
    else:
        gloss_list = None

    pose_len = torch.tensor([_get_T(item) for item in batch], dtype=torch.long)
    gloss_len = (
        torch.tensor([len(item["gloss"]) for item in batch], dtype=torch.long)
        if has_gloss
        else None
    )

    out = {
        "pose": padded_poses,  # {part: (B, max_T, K, C)}
        "text": text_list,  # List[str]
        "pose_len": pose_len,  # (B,)
        "gloss": gloss_list,  # List[List[str]] 或 None
        "gloss_len": gloss_len,  # (B,) 或 None
        "parts": body_parts,  # 便于下游知道顺序
        "adjacency_matrix": batch[0]["adjacency_matrix"],  # {str: Tensor}
    }
    return out


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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=my_collate_fn,
        pin_memory=pin_memory,
    )
    if verbose:
        elapsed = time.time() - start_time
        print(
            f"Created {split} dataloader with {len(dataset)} samples, batch size {batch_size}, shuffle={shuffle}, num_workers={num_workers}, pin_memory={pin_memory} in {elapsed:.2f} seconds."
        )
    return dataloader


if __name__ == "__main__":
    config = {
        "data_dir": "/nas/DDDataLang/250916/csl_dental",
        "augmentations": "speed,mask",
    }
    transform = NormalizeProcessor()
    dataset = MyDataset(config, transform, split='train')
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    dataloader = create_dataloader(
        config,
        split='train',
        transform=transform,
        batch_size=32,
        shuffle=True,
    )
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Pose parts:", batch['parts'])
        print("Pose shape for each part:")
        for part in batch['parts']:
            print(f"  {part}: {batch['pose'][part].shape}")
        print("Text sample:", batch['text'][:2])
        print("Pose lengths:", batch['pose_len'])
        if batch['gloss'] is not None:
            print("Gloss sample:", batch['gloss'][:2])
            print("Gloss lengths:", batch['gloss_len'])
        break
