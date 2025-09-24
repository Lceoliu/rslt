# Uni-GCN 与 MyDataset 集成指南

本指南详细说明如何将 Uni-GCN 模块与你现有的 MyDataset 数据处理管道无缝集成。

## 🎯 集成概述

你的 MyDataset 处理管道使用 COCO_Wholebody 格式，输出结构化的多部位姿态数据。Uni-GCN 的 `DatasetSTGCN` 适配器专门设计来处理这种数据格式。

### 数据流程图

```
MyDataset (COCO_Wholebody) → my_collate_fn → DatasetSTGCN → 特征提取/分类
     ↓                           ↓                ↓              ↓
133个关键点             多部位字典格式         ST-GCN处理       任务输出
```

## 📦 快速开始

### 1. 基本集成

```python
# 在你的训练脚本中
from uni_GCN import DatasetSTGCN, create_sign_recognition_model
from your_dataset_module import MyDataset, create_dataloader, NormalizeProcessor

# 创建数据加载器（你现有的代码）
config = {
    'data_dir': '/path/to/your/data',
    'augmentations': 'speed,mask',
    'aug_prob': 0.5
}

transform = NormalizeProcessor()
train_loader = create_dataloader(
    config=config,
    split='train',
    transform=transform,
    batch_size=32,
    shuffle=True
)

# 创建 ST-GCN 模型（新增）
model = DatasetSTGCN(
    parts=['body', 'left_hand', 'right_hand'],  # 使用你数据中的部位
    hidden_dim=128,
    temporal_pooling='mean'
)

# 训练循环
for batch in train_loader:
    # batch 来自 my_collate_fn，格式为:
    # {
    #     'pose': {'body': (B,T,K,C), 'left_hand': (B,T,K,C), ...},
    #     'pose_len': (B,),
    #     'text': List[str],
    #     'gloss': List[List[str]],
    #     ...
    # }

    outputs = model(batch)
    features = outputs['features']  # (B, feature_dim)
    # 继续你的训练逻辑...
```

### 2. 手语识别任务

```python
from uni_GCN import create_sign_recognition_model, SignLanguageTrainer

# 创建识别模型
model = create_sign_recognition_model(
    num_classes=1000,  # 根据你的数据集调整
    parts=['body', 'left_hand', 'right_hand'],
    hidden_dim=256
)

# 创建训练器
trainer = SignLanguageTrainer(
    model=model,
    learning_rate=1e-3,
    weight_decay=1e-4
)

# 训练循环
for batch in train_loader:
    # 你需要添加标签到 batch 中
    batch['labels'] = get_labels_for_batch(batch)  # 你的标签获取函数

    # 执行训练步骤
    losses = trainer.train_step(batch)
    print(f"Training loss: {losses['total']:.4f}")
```

### 3. 多任务模型

```python
from uni_GCN import create_multi_task_model

# 创建多任务模型（识别 + 翻译）
model = create_multi_task_model(
    num_recognition_classes=1000,
    vocab_size=5000,
    parts=['body', 'left_hand', 'right_hand', 'face'],
    hidden_dim=256
)

# 使用
for batch in train_loader:
    # 添加任务相关的标签
    batch['labels'] = get_recognition_labels(batch)
    batch['target_ids'] = get_translation_targets(batch)

    outputs = model(batch)
    rec_logits = outputs['recognition']      # (B, num_classes)
    trans_logits = outputs['translation']    # (B, vocab_size)
```

## 🔧 详细配置

### 支持的身体部位

根据你的 `body_info.json`，可用的部位包括：

```python
# COCO_Wholebody 格式的部位映射
available_parts = {
    'body': [0, 17],        # 17个身体关键点
    'face': [23, 91],       # 68个面部关键点
    'left_hand': [112, 133], # 21个左手关键点
    'right_hand': [91, 112]  # 21个右手关键点
}

# 在模型中使用
model = DatasetSTGCN(
    parts=['body', 'left_hand', 'right_hand', 'face'],  # 选择需要的部位
    hidden_dim=128
)
```

### 模型配置选项

```python
model = DatasetSTGCN(
    parts=['body', 'left_hand', 'right_hand'],
    hidden_dim=256,                    # 隐藏层维度
    graph_strategy='spatial',          # 图构建策略: 'uniform', 'distance', 'spatial'
    adaptive_graph=True,               # 是否使用可学习的邻接矩阵
    max_hop=2,                        # 图的最大跳数
    output_pooling='mean',            # 顶点池化方法: 'mean', 'max', 'none'
    use_pose_length_mask=True,        # 是否使用序列长度掩码
    temporal_pooling='attention'      # 时间池化方法: 'mean', 'max', 'last', 'attention'
)
```

### 时间池化方法选择

```python
# 1. 平均池化 - 适合大多数任务
model = DatasetSTGCN(temporal_pooling='mean')

# 2. 最大池化 - 突出显著特征
model = DatasetSTGCN(temporal_pooling='max')

# 3. 最后帧 - 适合实时应用
model = DatasetSTGCN(temporal_pooling='last')

# 4. 注意力池化 - 适合复杂序列任务
model = DatasetSTGCN(temporal_pooling='attention')
```

## 📊 与现有代码的兼容性

### 数据增强兼容

你现有的数据增强（速度变化、遮挡）完全兼容：

```python
# 你的现有配置
config = {
    'data_dir': '/path/to/data',
    'augmentations': 'speed,mask',     # ✓ 兼容
    'aug_prob': 0.5,                   # ✓ 兼容
    'aug_speed_min': 0.9,              # ✓ 兼容
    'aug_speed_max': 1.1,              # ✓ 兼容
    'aug_mask_prob': 0.05              # ✓ 兼容
}

# ST-GCN 会自动处理增强后的数据
dataset = MyDataset(config, transform, split='train')
dataloader = create_dataloader(config, 'train', transform, 32, True)
```

### 序列长度处理

```python
# 你的 my_collate_fn 处理变长序列，ST-GCN 自动处理
model = DatasetSTGCN(use_pose_length_mask=True)  # 启用长度掩码

for batch in dataloader:
    # batch['pose_len'] 包含实际序列长度
    # ST-GCN 会自动应用掩码，忽略填充部分
    outputs = model(batch)
```

## 🎛️ 性能优化

### 1. 轻量级配置（实时应用）

```python
lightweight_model = DatasetSTGCN(
    parts=['body', 'left_hand', 'right_hand'],
    hidden_dim=64,                     # 减少隐藏维度
    graph_strategy='uniform',          # 简单的图策略
    adaptive_graph=False,              # 固定邻接矩阵
    temporal_pooling='last',           # 快速池化
    use_pose_length_mask=False         # 跳过掩码计算
)
```

### 2. 高精度配置（研究用）

```python
high_accuracy_model = DatasetSTGCN(
    parts=['body', 'left_hand', 'right_hand', 'face'],
    hidden_dim=512,                    # 大隐藏维度
    graph_strategy='spatial',          # 复杂图策略
    adaptive_graph=True,               # 可学习邻接矩阵
    max_hop=3,                        # 更大感受野
    temporal_pooling='attention',      # 注意力池化
    use_pose_length_mask=True         # 完整掩码
)
```

### 3. 内存优化

```python
# 对于大批次训练
model = DatasetSTGCN(
    parts=['body', 'left_hand', 'right_hand'],  # 减少部位
    hidden_dim=128,
    output_pooling='mean'              # 降低内存使用
)

# 使用梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = compute_loss(outputs, batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🔍 调试和监控

### 输出维度检查

```python
model = DatasetSTGCN(parts=['body', 'left_hand', 'right_hand'])

# 检查输出维度
print(f"Feature dimension: {model.get_output_dim()}")

# 检查中间特征
outputs = model(batch, return_features=True)
print("Available outputs:", outputs.keys())
print(f"Features shape: {outputs['features'].shape}")
print(f"Sequence features shape: {outputs['sequence_features'].shape}")
```

### 模型信息

```python
# 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 检查使用的身体部位
print(f"Model parts: {model.parts}")
print(f"ST-GCN part keys: {model.stgcn.part_keys}")
```

## 🚀 完整训练示例

```python
import torch
import torch.nn as nn
from uni_GCN import create_sign_recognition_model, SignLanguageTrainer
from your_dataset_module import MyDataset, create_dataloader, NormalizeProcessor

def main():
    # 配置
    config = {
        'data_dir': '/path/to/your/csl_dental',
        'augmentations': 'speed,mask',
        'aug_prob': 0.5
    }

    # 数据
    transform = NormalizeProcessor()
    train_loader = create_dataloader(config, 'train', transform, 32, True)
    val_loader = create_dataloader(config, 'val', transform, 32, False)

    # 模型
    model = create_sign_recognition_model(
        num_classes=1000,  # 根据你的数据调整
        parts=['body', 'left_hand', 'right_hand'],
        hidden_dim=256
    )

    # 训练器
    trainer = SignLanguageTrainer(model, learning_rate=1e-3)

    # 训练循环
    for epoch in range(100):
        # 训练
        total_loss = 0
        for batch in train_loader:
            # 添加标签 (你需要实现这个函数)
            batch['labels'] = get_labels_from_gloss(batch['gloss'])

            losses = trainer.train_step(batch)
            total_loss += losses['total']

        # 验证
        val_acc = 0
        val_count = 0
        for batch in val_loader:
            batch['labels'] = get_labels_from_gloss(batch['gloss'])
            metrics = trainer.validate_step(batch)
            if 'accuracy' in metrics:
                val_acc += metrics['accuracy']
                val_count += 1

        avg_val_acc = val_acc / val_count if val_count > 0 else 0
        print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}, Val Acc={avg_val_acc:.4f}")

def get_labels_from_gloss(gloss_list):
    """将 gloss 转换为标签 - 你需要根据数据实现"""
    # 这里需要你的词汇表映射逻辑
    # 例如: gloss -> label_id
    vocab = load_your_vocabulary()  # 你的词汇表
    labels = []
    for glosses in gloss_list:
        if glosses and len(glosses) > 0:
            # 取第一个gloss或者合并多个gloss的逻辑
            label_id = vocab.get(glosses[0], 0)  # 默认为0
            labels.append(label_id)
        else:
            labels.append(0)
    return torch.tensor(labels)

if __name__ == "__main__":
    main()
```

## 💡 最佳实践

1. **选择合适的部位**: 根据你的任务选择必要的身体部位，过多部位会增加计算量
2. **调整隐藏维度**: 从128开始，根据性能需求调整
3. **使用序列长度掩码**: 对于变长序列，始终启用 `use_pose_length_mask=True`
4. **时间池化选择**: 分类任务用'mean'，序列任务用'attention'
5. **数据增强**: 利用你现有的增强策略，效果很好
6. **监控内存**: 大模型或长序列时注意GPU内存使用

现在你可以将 Uni-GCN 无缝集成到现有的训练框架中！🎉