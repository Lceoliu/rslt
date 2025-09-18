## 数据包结构

```bash
.
├── data_package1
│   ├── ${dataset}.dat # numpy内存映射文件
│   ├── meta.json # 数据包元信息
│   └── annotation.json # 标注文件
└── data_package2
```

说明：

- `${dataset}.dat`：包含所有姿态数据的 numpy 内存映射文件，便于高效读取大规模数据。
- `meta.json`：包含数据包的元信息：
  - `dtype`：数据类型（如`float32`）。
  - `shape`：数据形状（如`(Total, Num_Joints, 3)`）。
  - `video_root`：视频文件的根目录。
  - `video_frame`：FPS。
- `annotation.json`：包含每个样本的标注信息：

  ```python
  {
  	"hash_id_1": {
  		"video_path": "path/to/video.mp4",	# relative to video_root
  		"gloss": "gloss_1 gloss_2 ...",	# 空格分隔的手语词
  		"text": "full sentence text",	# 完整句子文本
  		"pose_index": [start, end]	# 在内存映射文件中的起止索引 [start, end)
  	},
  	"hash_id_2": {
  		...
  	}
  }
  ```

## body_info.json

```python
{
	"COCO_Wholebody": {
        # all indexes are 0-based
		"COCO_EDGES": {
			"body": [...],
			"face": [...],
			"right_hand": [...],
			"left_hand": [...]
		},
		"BODY_PARTS_INTERVALS": {
			"body": [0, 17],
			"face": [23, 91],
			"right_hand": [91, 112],
			"left_hand": [112, 133]
		},
		"NUM_KEYPOINTS": 133,
		"CENTER_INDICES": {
			"body": 0,
			"face": 31,
			"right_hand": 112,
			"left_hand": 91
		},
        # 用于骨骼归一化的骨骼对
		"NORMALIZE_BONE": [5, 6],
		"DISCARDED_KEYPOINTS": [13, 14, 15, 16] # 下半身关键点,
	}
}
```
