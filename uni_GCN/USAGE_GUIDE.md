# Uni-GCN 使用指南

## 🎯 如何使用自定义关键点格式

### 1. 使用COCO-Wholebody格式

```python
import torch
from uni_GCN import UniSTGCN

# 创建支持COCO-Wholebody的模型
model = UniSTGCN(
    parts=[
        ('coco_wholebody', 'body'),       # COCO身体17个关键点
        ('coco_wholebody', 'left_hand'),  # 左手21个关键点
        ('coco_wholebody', 'right_hand'), # 右手21个关键点
        ('coco_wholebody', 'face')        # 面部68个关键点
    ],
    keypoint_format='coco_wholebody',
    auto_convert=True,  # 自动从完整数据中提取部位
    hidden_dim=64
)

# COCO-Wholebody数据 (133个关键点)
coco_data = torch.randn(batch_size, seq_len, 133, 3)

# 直接输入完整数据，模型自动提取所需部位
output = model(coco_data)
```

### 2. 使用MediaPipe格式

```python
# MediaPipe Holistic格式 (543个关键点)
model = UniSTGCN(
    parts=[
        ('mediapipe_holistic', 'body'),
        ('mediapipe_holistic', 'left_hand'),
        ('mediapipe_holistic', 'right_hand')
    ],
    keypoint_format='mediapipe_holistic',
    auto_convert=True
)

# MediaPipe数据
mp_data = torch.randn(batch_size, seq_len, 543, 3)
output = model(mp_data)
```

### 3. 创建自定义格式

```python
from uni_GCN import create_custom_format, UniSTGCN

# 定义你的自定义格式
custom_format = create_custom_format(
    name="my_custom_format",
    parts_config={
        'torso': list(range(10)),        # 关键点索引0-9
        'left_arm': list(range(10, 20)), # 关键点索引10-19
        'right_arm': list(range(20, 30)) # 关键点索引20-29
    },
    connections_config={
        'torso': [[0,1], [1,2], [2,3]],     # 连接关系
        'left_arm': [[0,1], [1,2], [2,3]],
        'right_arm': [[0,1], [1,2], [2,3]]
    },
    centers_config={
        'torso': 5,      # 中心点索引
        'left_arm': 5,
        'right_arm': 5
    }
)

# 使用自定义格式
model = UniSTGCN(
    parts=[
        ('my_custom_format', 'torso'),
        ('my_custom_format', 'left_arm'),
        ('my_custom_format', 'right_arm')
    ],
    keypoint_format=custom_format,
    auto_convert=True
)
```

### 4. 格式转换

```python
from uni_GCN import KeypointConverter, convert_coco_to_unisign

# 方法1: 直接转换函数
coco_data = torch.randn(2, 64, 133, 3)
unisign_parts = convert_coco_to_unisign(coco_data)

# 方法2: 使用转换器
converter = KeypointConverter('coco_wholebody')

# 提取特定部位
body_keypoints = converter.extract_part(coco_data, 'body')
left_hand = converter.extract_part(coco_data, 'left_hand')

# 归一化（相对于中心点）
left_hand_normalized = converter.normalize_part(
    left_hand, 'left_hand', method='center'
)
```

## 🔧 高级配置

### 图构建策略

```python
model = UniSTGCN(
    parts=[('coco_wholebody', 'body')],
    graph_strategy='spatial',  # 'uniform', 'distance', 'spatial'
    adaptive_graph=True,       # 可学习的邻接矩阵
    max_hop=2                 # 最大跳数
)
```

### 部位交互

```python
# 模型会自动处理部位间的交互：
# - 左手特征 += 左手腕的身体特征
# - 右手特征 += 右手腕的身体特征
# - 面部特征 += 头部的身体特征

model = UniSTGCN(
    parts=[
        ('coco_wholebody', 'body'),      # 必须包含body才能进行部位交互
        ('coco_wholebody', 'left_hand'),
        ('coco_wholebody', 'right_hand')
    ],
    keypoint_format='coco_wholebody'
)
```

## 🚀 实际应用示例

### 手语识别系统

```python
class SignLanguageRecognizer(nn.Module):
    def __init__(self, num_classes, input_format='coco_wholebody'):
        super().__init__()
        self.stgcn = UniSTGCN(
            parts=[
                (input_format, 'body'),
                (input_format, 'left_hand'),
                (input_format, 'right_hand')
            ],
            keypoint_format=input_format,
            auto_convert=True
        )
        self.classifier = nn.Linear(self.stgcn.get_output_dim(), num_classes)

    def forward(self, keypoints):
        # keypoints: (batch, time, total_points, 3)
        features = self.stgcn(keypoints)          # (batch, time, feature_dim)
        pooled = features.mean(dim=1)             # (batch, feature_dim)
        logits = self.classifier(pooled)          # (batch, num_classes)
        return logits

# 使用
model = SignLanguageRecognizer(num_classes=100, input_format='coco_wholebody')
coco_keypoints = torch.randn(4, 64, 133, 3)
predictions = model(coco_keypoints)
```

### 多模态融合

```python
class MultiModalSTGCN(nn.Module):
    def __init__(self):
        super().__init__()
        # 不同格式的数据使用不同模型
        self.coco_model = UniSTGCN(
            parts=[('coco_wholebody', 'body'), ('coco_wholebody', 'left_hand')],
            keypoint_format='coco_wholebody',
            auto_convert=True
        )

        self.mediapipe_model = UniSTGCN(
            parts=[('mediapipe_holistic', 'body'), ('mediapipe_holistic', 'face')],
            keypoint_format='mediapipe_holistic',
            auto_convert=True
        )

        self.fusion = nn.Linear(self.coco_model.get_output_dim() +
                               self.mediapipe_model.get_output_dim(), 512)

    def forward(self, coco_data, mediapipe_data):
        coco_features = self.coco_model(coco_data)
        mp_features = self.mediapipe_model(mediapipe_data)

        combined = torch.cat([coco_features, mp_features], dim=-1)
        return self.fusion(combined)
```

## 🐛 常见问题

### Q: 如何处理不同长度的序列？
```python
# 使用padding
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # batch: list of (keypoints, label)
    keypoints = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad序列到相同长度
    padded_keypoints = pad_sequence(keypoints, batch_first=True, padding_value=0)

    return padded_keypoints, torch.stack(labels)
```

### Q: 如何处理缺失的关键点？
```python
# 在关键点数据中，置信度为0表示缺失
keypoints = torch.randn(2, 64, 133, 3)
keypoints[:, :, :, 2] = torch.where(
    torch.rand_like(keypoints[:, :, :, 2]) > 0.1,  # 90%的点有效
    keypoints[:, :, :, 2],
    torch.zeros_like(keypoints[:, :, :, 2])  # 10%的点缺失
)
```

### Q: 如何微调预训练模型？
```python
# 加载预训练权重
model = UniSTGCN(parts=['body', 'left', 'right'])
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint, strict=False)

# 冻结某些层
for name, param in model.named_parameters():
    if 'spatial_gcns' in name:
        param.requires_grad = False  # 冻结空间GCN

# 只训练时间GCN和分类器
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

## 📊 性能对比

不同格式的性能对比（参考数据）：

| 格式 | 提取时间 | 推理时间 | 内存占用 |
|------|---------|----------|----------|
| 手动提取 | ~2ms | ~15ms | 较低 |
| Auto-convert | ~0.5ms | ~16ms | 中等 |
| 预提取缓存 | ~0.1ms | ~15ms | 较高 |

建议：
- **训练时**: 使用预提取缓存，减少重复计算
- **推理时**: 使用auto_convert，代码简洁
- **内存受限**: 使用手动提取，精确控制

这个系统设计让你可以灵活地使用各种关键点格式，同时保持代码的简洁性和高性能！