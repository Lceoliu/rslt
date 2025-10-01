import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import av
import gc

from typing import Literal, Optional, Tuple, List, Dict
from pathlib import Path
from tqdm import tqdm

BODY_INFO_PATH = Path(__file__).parent / 'body_info.json'
TEST_OUTPUT_DIR = Path(__file__).parent / 'test_outputs'
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class NormalizeProcessor:
    def __init__(
        self,
        keypoint_format: Literal['COCO_Wholebody'] = 'COCO_Wholebody',
        conf_threshold: float = 0.25,
        add_fullbody_channel: bool = True,
    ):
        self.keypoint_format = keypoint_format
        self.conf_threshold = conf_threshold
        self.add_fullbody_channel = add_fullbody_channel
        try:
            with open(BODY_INFO_PATH, 'r') as f:
                self.body_info = json.load(f)[self.keypoint_format]
        except FileNotFoundError:
            raise FileNotFoundError(f'body_info.json not found at {BODY_INFO_PATH}')
        except Exception as e:
            raise RuntimeError(f'Error loading body_info.json: {e}')
        self.num_keypoints = self.body_info['NUM_KEYPOINTS']
        self.center_indices = self.body_info['CENTER_INDICES']
        self.normalize_bone = self.body_info['NORMALIZE_BONE']
        self.edges = self.body_info['EDGES']
        self.body_part_intervals = self.body_info['BODY_PARTS_INTERVALS']
        self.discarded_keypoints = self.body_info.get('DISCARDED_KEYPOINTS', [])
        self.all_indices = []
        for interval in self.body_part_intervals.values():
            self.all_indices.extend(list(range(interval[0], interval[1])))
        self.all_indices = set(self.all_indices) - set(self.discarded_keypoints)
        self.all_indices = sorted(list(self.all_indices))

        self.indices_map = {idx: i for i, idx in enumerate(self.all_indices)}

        if self.add_fullbody_channel:
            self.body_part_intervals['fullbody'] = [0, self.num_keypoints]
            self.center_indices['fullbody'] = 0  # 使用 body 的中心点
            self.edges['fullbody'] = (
                self.edges['body']
                + self.edges['face']
                + self.edges['left_hand']
                + self.edges['right_hand']
            )
        self.body_part_start_idx = {
            part: interval[0] for part, interval in self.body_part_intervals.items()
        }
        self.body_parts = list(self.body_part_intervals.keys())
        assert (
            self.center_indices.keys() == self.body_part_intervals.keys()
        ), "CENTER_INDICES and BODY_PART_INTERVALS keys must match"

    def gen_adjacency_matrix(
        self, normalize: bool = False, split_part: bool = False
    ) -> np.ndarray | dict[str, np.ndarray]:
        if self.discarded_keypoints:
            num_keypoints = len(self.all_indices)
        else:
            num_keypoints = self.num_keypoints
        adjacency_matrix = np.zeros((num_keypoints, num_keypoints), dtype=np.float32)

        for part, edges in self.edges.items():
            for p1, p2 in edges:
                if self.discarded_keypoints:
                    if (p1 not in self.all_indices) or (p2 not in self.all_indices):
                        continue
                adjacency_matrix[self.indices_map[p1], self.indices_map[p2]] = 1.0
                adjacency_matrix[self.indices_map[p2], self.indices_map[p1]] = 1.0
        if normalize:
            adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(
                axis=1, keepdims=True
            )
        if split_part:
            part_matrices = []
            for part in self.body_parts:
                start, end = self.body_part_intervals[part]

                indices = [
                    self.indices_map[idx]
                    for idx in range(start, end)
                    if idx in self.all_indices
                ]
                part_matrix = adjacency_matrix[np.ix_(indices, indices)]
                part_matrices.append(part_matrix)
            adjacency_matrix = {
                part: mat for part, mat in zip(self.body_parts, part_matrices)
            }
        return adjacency_matrix

    def __call__(
        self,
        keypoints: np.ndarray,
        concat_velocity: bool = False,
        concat_acceleration: bool = False,
        generate_video: bool = False,
        save_video_path: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """Normalize keypoints.

        Args:
            keypoints (np.ndarray): Keypoints to normalize. Shape (T, 134, 3)
            concat_velocity (bool, optional): Whether to concatenate velocity. Defaults to False.
            concat_acceleration (bool, optional): Whether to concatenate acceleration. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: Normalized keypoints by body part. And optionally video frames if generate_video is True.
        """
        W, H = keypoints[0, 0, 0], keypoints[0, 0, 1]
        keypoints = keypoints[:, 1:, :]  # (T, 133, 3)
        self.keypoints = keypoints
        splited_kpts = self._split_to_parts()  # Dict[str, (T, K, 3)]
        splited_kpts = {
            k: NormalizeProcessor.interpolate_low_confidence_linear(
                v, self.conf_threshold
            )[0]
            for k, v in splited_kpts.items()
            if v is not None
        }
        # if generate_video:
        #     self.visualize_keypoints(
        #         splited_kpts,
        #         # original_size=(W, H),
        #         edges=self.edges,
        #         body_part_start_idx=self.body_part_start_idx,
        #         save_path=TEST_OUTPUT_DIR / 'keypoints_visualization_interpolated.mp4',
        #     )
        splited_kpts = self._centralize(splited_kpts)  # 中心化
        # if generate_video:
        #     self.visualize_keypoints(
        #         splited_kpts,
        #         # original_size=(W, H),
        #         edges=self.edges,
        #         body_part_start_idx=self.body_part_start_idx,
        #         save_path=TEST_OUTPUT_DIR / 'keypoints_visualization_centralized.mp4',
        #     )
        splited_kpts = self._bone_normalize(splited_kpts, unit_length=8.0)  # 骨骼归一化

        # if generate_video:
        #     self.visualize_keypoints(
        #         splited_kpts,
        #         edges=self.edges,
        #         body_part_start_idx=self.body_part_start_idx,
        #         keep_indices=self.all_indices,
        #         save_path=save_video_path / 'keypoints_visualization_normalized.mp4',
        #     )
        splited_kpts = self._discard_keypoints(splited_kpts)  # 舍弃部分关键点
        return splited_kpts

    def _discard_keypoints(
        self, splited_kpts: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Discard specified keypoints.

        Args:
            splited_kpts (Dict[str, np.ndarray]): Keypoints split by body part. Each value has shape (T, K, 3).

        Returns:
            Dict[str, np.ndarray]: Keypoints with specified keypoints discarded.
        """
        # Always align kept joints with self.all_indices (global kept set)
        # This ensures 'fullbody' excludes gaps (e.g., indices 17..22) and any discarded joints.
        global_kept = set(self.all_indices)
        processed_kpts = {}
        for part, kpts in splited_kpts.items():
            start_abs = self.body_part_start_idx[part]
            indices = [
                i for i in range(kpts.shape[1]) if (start_abs + i) in global_kept
            ]
            processed_kpts[part] = kpts[:, indices, :]
        return processed_kpts

    def _bone_normalize(
        self, splited_kpts: Dict[str, np.ndarray], unit_length: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """Normalize keypoints by the length of a specific bone.

        Args:
            splited_kpts (Dict[str, np.ndarray]): Keypoints split by body part. Each value has shape (T, K, 3).
            unit_length (float, optional): Desired length of the normalization bone. Defaults to 1.0.

        Returns:
            Dict[str, np.ndarray]: Bone-normalized keypoints.
        """
        normalized_kpts = {}
        bone_start, bone_end = self.normalize_bone
        assert hasattr(self, 'keypoints'), 'Keypoints not set. Run __call__ first.'
        fullbody_kpts = self.keypoints  # (T, 133, 3)
        bone_vec = (
            fullbody_kpts[:, bone_end, :2] - fullbody_kpts[:, bone_start, :2]
        )  # (T, 2)
        bone_length = np.linalg.norm(bone_vec, axis=-1)  # (T,)
        bone_length[bone_length == 0] = 1.0  # 防止除零
        bone_length[bone_length < 1e-6] = 1e-6  # 防止过小
        confidence_indices = (
            fullbody_kpts[:, bone_start, 2] + fullbody_kpts[:, bone_end, 2]
        ) >= 2 * self.conf_threshold
        avg_bone_length = (
            np.mean(bone_length[confidence_indices])
            if np.any(confidence_indices)
            else 1.0
        )
        for part, kpts in splited_kpts.items():
            kpts_normalized = kpts.copy()
            kpts_normalized[..., :2] = (
                kpts_normalized[..., :2] / avg_bone_length * unit_length
            )
            normalized_kpts[part] = kpts_normalized
        return normalized_kpts

    def _centralize(self, splited_kpts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Centralize keypoints by their respective center indices.

        Args:
            splited_kpts (Dict[str, np.ndarray]): Keypoints split by body part. Each value has shape (T, K, 3).

        Returns:
            Dict[str, np.ndarray]: Centralized keypoints.
        """
        centralized_kpts = {}
        for part, kpts in splited_kpts.items():
            center_idx = self.center_indices[part]
            center_idx -= self.body_part_start_idx[part]  # 转为局部索引
            center = kpts[:, center_idx : center_idx + 1, :2]  # (T, 1, 2)
            kpts_centralized = kpts.copy()
            kpts_centralized[..., :2] -= center  # 中心化
            centralized_kpts[part] = kpts_centralized
        return centralized_kpts

    def _split_to_parts(self) -> Dict[str, np.ndarray]:
        assert hasattr(self, 'keypoints'), 'Keypoints not set. Run __call__ first.'
        parts = {}
        for part in self.body_parts:
            parts[part] = self.keypoints[
                :,
                self.body_part_intervals[part][0] : self.body_part_intervals[part][1],
                :,
            ]
        return parts

    @staticmethod
    def calc_bbox(pose: np.ndarray) -> Tuple[float, float, float, float]:
        """计算关键点的边界框 (x_min, y_min, x_max, y_max)，忽略置信度

        Args:
            pose (np.ndarray): [T, K, 3]
        Returns:
            bbox: (x_min, y_min, x_max, y_max)
        """
        assert pose.ndim == 3 and pose.shape[-1] == 3
        x = pose[..., 0]
        y = pose[..., 1]
        conf = pose[..., 2]
        valid = conf > 0
        if not np.any(valid):
            return 0, 0, 0, 0
        x_valid = x[valid]
        y_valid = y[valid]
        x_min, x_max = np.min(x_valid), np.max(x_valid)
        y_min, y_max = np.min(y_valid), np.max(y_valid)
        return x_min, y_min, x_max, y_max

    @staticmethod
    def visualize_keypoints(
        splited_kpts: Dict[str, np.ndarray],
        original_size: Tuple[int, int] = None,
        edges: Dict[str, List[List[int]]] = None,
        body_part_start_idx: Dict[str, int] = None,
        keep_indices: List[int] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Use matplotlib and av to visualize keypoints.
        """
        matplotlib.use('Agg')
        part_cnts = len(splited_kpts)
        col_num = 2
        row_num = (part_cnts + 1) // col_num
        if original_size is not None:
            splited_bboxes = {
                part: (0, 0, original_size[0], original_size[1])
                for part, kpt in splited_kpts.items()
            }
        else:
            splited_bboxes = {
                part: NormalizeProcessor.calc_bbox(kpt)
                for part, kpt in splited_kpts.items()
            }
        T = next(iter(splited_kpts.values())).shape[0]
        if save_path is None:
            save_path = TEST_OUTPUT_DIR / 'keypoints_visualization.mp4'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(save_path, mode='w')
        stream = container.add_stream('libx264', rate=30)
        stream.width = min(1280, int(col_num * 640))
        stream.height = min(960, int(row_num * 480))
        stream.pix_fmt = 'yuv420p'
        for t in tqdm(range(T), desc='Visualizing keypoints'):
            plt.close('all')
            fig, axes = plt.subplots(
                row_num, col_num, figsize=(6 * col_num, 6 * row_num)
            )
            axes = axes.flatten()

            for i, (part, kpt) in enumerate(splited_kpts.items()):
                ax = axes[i]
                abs_idx = body_part_start_idx[part] + np.arange(kpt.shape[1])
                abs_idx = [idx for idx in abs_idx if idx in keep_indices]
                rel_idx = [idx - body_part_start_idx[part] for idx in abs_idx]
                if len(rel_idx) == 0:
                    continue
                ax.plot(kpt[t, rel_idx, 0], kpt[t, rel_idx, 1], 'o', markersize=5)
                size = splited_bboxes[part]
                ax.set_xlim(size[0], size[2])
                ax.set_ylim(size[3], size[1])
                ax.set_title(part)
                if edges and part in edges:
                    for edge in edges[part]:
                        p1, p2 = edge
                        if (p1 not in abs_idx) or (p2 not in abs_idx):
                            continue
                        p1 = p1 - body_part_start_idx[part]
                        p2 = p2 - body_part_start_idx[part]
                        avg_conf = (kpt[t, p1, 2] + kpt[t, p2, 2]) / 2.0  # 平均置信度
                        if avg_conf > 0.25:
                            ax.plot(
                                [kpt[t, p1, 0], kpt[t, p2, 0]],
                                [kpt[t, p1, 1], kpt[t, p2, 1]],
                                'r-',
                                linewidth=2,
                            )
                        else:
                            ax.plot(
                                [kpt[t, p1, 0], kpt[t, p2, 0]],
                                [kpt[t, p1, 1], kpt[t, p2, 1]],
                                'b--',
                                linewidth=1,
                                alpha=0.3,
                            )
                # ax.scatter(kpt[t, :, 0], kpt[t, :, 1], c='b', s=5, alpha=0.5)
                if kpt.shape[-1] == 5:
                    # add velocity
                    vx = kpt[t, :, 3]
                    vy = kpt[t, :, 4]
                    ax.quiver(
                        kpt[t, :, 0],
                        kpt[t, :, 1],
                        vx,
                        vy,
                        angles='xy',
                        scale_units='xy',
                        scale=1,
                        alpha=0.5,
                    )
            plt.tight_layout()
            # Convert plot to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = av.VideoFrame.from_ndarray(img, format='rgba')
            packet = stream.encode(frame)
            container.mux(packet)
            plt.close(fig)
            del fig, axes
            if t % 10 == 0:
                gc.collect()
        # flush stream
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        print(f'Keypoints visualization saved to {save_path}')

    @staticmethod
    def interpolate_low_confidence_linear(
        pose: np.ndarray,  # [T, K, 3]
        confidence_threshold: float = 0.1,
        max_search: int = 10,
        set_conf_zero: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回:
        pose_filled: [T, K, 3] 补点后的结果（x,y 被替换；conf 视 set_conf_zero 可能被清零）
        fill_mask:   [T, K]    True 表示该点是插值/填充出来的
        """
        assert pose.ndim == 3 and pose.shape[-1] == 3
        T, K, _ = pose.shape

        out = pose.copy()
        xy = out[..., :2]
        conf = out[..., 2]
        valid = conf >= confidence_threshold
        fill_mask = np.zeros((T, K), dtype=bool)

        for k in range(K):
            vk = valid[:, k]  # [T]
            xk = xy[:, k, 0]
            yk = xy[:, k, 1]
            ck = conf[:, k]

            idx_valid = np.nonzero(vk)[0]
            if idx_valid.size == 0:
                xk[:] = 0.0
                yk[:] = 0.0
                if set_conf_zero:
                    ck[:] = 0.0
                fill_mask[:, k] = True
                continue

            for t in range(T):
                if vk[t]:
                    continue  # 本帧有效，跳过
                # 向左找
                t_left = None
                for dt in range(1, max_search + 1):
                    tl = t - dt
                    if tl < 0:
                        break
                    if vk[tl]:
                        t_left = tl
                        break
                # 向右找
                t_right = None
                for dt in range(1, max_search + 1):
                    tr = t + dt
                    if tr >= T:
                        break
                    if vk[tr]:
                        t_right = tr
                        break

                if (t_left is not None) and (t_right is not None):
                    # 线性插值
                    w = (t - t_left) / float(t_right - t_left)
                    xk[t] = (1 - w) * xk[t_left] + w * xk[t_right]
                    yk[t] = (1 - w) * yk[t_left] + w * yk[t_right]
                    fill_mask[t, k] = True
                    if set_conf_zero:
                        ck[t] = 0.0
                elif (t_left is not None) or (t_right is not None):
                    # 最近邻
                    src = t_left if t_left is not None else t_right
                    xk[t] = xk[src]
                    yk[t] = yk[src]
                    fill_mask[t, k] = True
                    if set_conf_zero:
                        ck[t] = 0.0
                else:
                    # 完全无邻居：置零
                    xk[t] = 0.0
                    yk[t] = 0.0
                    fill_mask[t, k] = True
                    if set_conf_zero:
                        ck[t] = 0.0

        return out, fill_mask


if __name__ == "__main__":
    test_data_dir = Path(__file__).parent / 'test_data'
    npy_files = list(test_data_dir.glob('*.npy'))
    assert len(npy_files) > 0, f'No .npy files found in {test_data_dir}'
    for i in range(len(npy_files)):
        if i >= 1:
            break
        test_data = np.load(npy_files[i])  # (T, 134, 3)
        processor = NormalizeProcessor(conf_threshold=0.25, add_fullbody_channel=True)
        part_data = processor(
            test_data,
            generate_video=True,
            save_video_path=TEST_OUTPUT_DIR / f'npy_test_{npy_files[i].stem}',
        )
