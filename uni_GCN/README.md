# Uni-GCN: Spatio-Temporal Graph Convolutional Networks

A standalone PyTorch implementation of **Spatio-Temporal Graph Convolutional Networks (ST-GCN)** extracted from the Uni-Sign project. This module provides efficient graph-based modeling for multi-part skeletal data including body keypoints, hand gestures, and facial expressions.

## 🌟 Features

- **Multi-Format Support**: COCO-Wholebody, MediaPipe Holistic, Uni-Sign, and custom formats
- **Auto-Conversion**: Automatically extract body parts from full keypoint data
- **Multi-Part Modeling**: Support for body, left/right hands, and face keypoints
- **Flexible Graph Construction**: Multiple adjacency strategies (uniform, distance, spatial)
- **Format Conversion Tools**: Easy conversion between different keypoint formats
- **Modular Design**: Easy to integrate into existing projects
- **Efficient Implementation**: Optimized for batch processing
- **Well-Documented**: Comprehensive API documentation and examples

## 📦 Installation

```bash
# Clone or copy the uni_GCN directory to your project
cd your_project
cp -r /path/to/uni_GCN .

# Install dependencies
pip install torch torchvision numpy
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from uni_GCN import UniSTGCN

# Method 1: Legacy format (original Uni-Sign)
model = UniSTGCN(
    parts=['body', 'left', 'right', 'face_all'],
    input_dim=3,
    hidden_dim=64,
    graph_strategy='distance'
)

keypoints = {
    'body': torch.randn(2, 64, 9, 3),     # Body: 9 joints
    'left': torch.randn(2, 64, 21, 3),    # Left hand: 21 joints
    'right': torch.randn(2, 64, 21, 3),   # Right hand: 21 joints
    'face_all': torch.randn(2, 64, 18, 3) # Face: 18 joints
}

output = model(keypoints)
print(f"Output shape: {output.shape}")
```

### COCO-Wholebody Format

```python
from uni_GCN import UniSTGCN

# Method 2: COCO-Wholebody format with auto-conversion
model = UniSTGCN(
    parts=[
        ('coco_wholebody', 'body'),      # 17 COCO body keypoints
        ('coco_wholebody', 'left_hand'), # 21 left hand keypoints
        ('coco_wholebody', 'right_hand') # 21 right hand keypoints
    ],
    keypoint_format='coco_wholebody',
    auto_convert=True,  # Automatically extract parts from full data
    hidden_dim=64
)

# Full COCO-Wholebody data (133 keypoints)
coco_keypoints = torch.randn(2, 64, 133, 3)

# Model automatically extracts required parts
output = model(coco_keypoints)
print(f"COCO output shape: {output.shape}")
```

### MediaPipe Holistic Format

```python
# Method 3: MediaPipe format
model = UniSTGCN(
    parts=[
        ('mediapipe_holistic', 'body'),
        ('mediapipe_holistic', 'left_hand'),
        ('mediapipe_holistic', 'right_hand')
    ],
    keypoint_format='mediapipe_holistic',
    auto_convert=True
)

# MediaPipe data (543 keypoints)
mp_keypoints = torch.randn(2, 64, 543, 3)
output = model(mp_keypoints)
```

### Simplified Interface

```python
from uni_GCN import MultiPartSTGCN

# Predefined configurations
model = MultiPartSTGCN(config='upper')  # Body + hands only

# Generate sample data
keypoints = {
    'body': torch.randn(2, 64, 9, 3),
    'left': torch.randn(2, 64, 21, 3),
    'right': torch.randn(2, 64, 21, 3)
}

output = model(keypoints)
```

## 🏗️ Architecture Overview

### Network Structure

```
Input Keypoints (B, T, V, 3)
         ↓
1. Linear Projection: (x,y,score) → 64D
         ↓
2. Spatial ST-GCN Chain:
   - 64 → 128 → 256 dimensions
   - Graph convolution across joints
         ↓
3. Part Interaction:
   - Hands enhanced with wrist context
   - Face enhanced with head context
         ↓
4. Temporal ST-GCN Chain:
   - 3-layer temporal modeling
   - Kernel size: 5
         ↓
5. Vertex Pooling: mean/max across joints
         ↓
6. Feature Concatenation: All parts → Final output
```

### Supported Keypoint Formats

| Format | Total Points | Description | Parts Available |
|--------|-------------|-------------|-----------------|
| `uni_sign` | 133 | Original Uni-Sign format | `body`(9), `left`(21), `right`(21), `face_all`(18) |
| `coco_wholebody` | 133 | COCO-Wholebody format | `body`(17), `left_hand`(21), `right_hand`(21), `face`(68), `left_foot`(3), `right_foot`(3) |
| `mediapipe_holistic` | 543 | MediaPipe Holistic | `body`(33), `left_hand`(21), `right_hand`(21), `face`(468) |
| Custom | Variable | User-defined format | User-defined parts |

### Legacy Body Parts (Uni-Sign Format)

| Part | Joints | Description |
|------|--------|-------------|
| `body` | 9 | Main body keypoints (head, shoulders, hips, etc.) |
| `left` | 21 | Left hand landmarks |
| `right` | 21 | Right hand landmarks |
| `face_all` | 18 | Facial keypoints (contour + features) |

## 📊 Graph Construction

### Adjacency Strategies

- **Uniform**: Single adjacency matrix with uniform edge weights
- **Distance**: Multiple matrices based on hop distances
- **Spatial**: Direction-aware partitioning (centripetal, centrifugal)

### Example: Hand Graph Structure

```python
from uni_GCN import Graph

# Create hand graph
graph = Graph(layout='left', strategy='distance', max_hop=2)
print(f\"Nodes: {graph.num_node}\")
print(f\"Adjacency shape: {graph.A.shape}\")

# Hand connections:
# - Thumb: 0→1→2→3→4
# - Index: 0→5→6→7→8
# - Middle: 0→9→10→11→12
# - Ring: 0→13→14→15→16
# - Pinky: 0→17→18→19→20
```

## 🔧 Advanced Usage

### Format Conversion

```python
from uni_GCN import KeypointConverter, convert_coco_to_unisign

# Method 1: Direct conversion functions
coco_data = torch.randn(2, 64, 133, 3)
unisign_parts = convert_coco_to_unisign(coco_data)

# Method 2: Using converter class
converter = KeypointConverter('coco_wholebody')
parts = converter.extract_multiple_parts(
    coco_data,
    ['body', 'left_hand', 'right_hand', 'face']
)

# Normalize hand keypoints
parts['left_hand'] = converter.normalize_part(
    parts['left_hand'], 'left_hand', method='center'
)
```

### Custom Keypoint Formats

```python
from uni_GCN import create_custom_format, UniSTGCN

# Define your custom format
custom_format = create_custom_format(
    name="my_format",
    parts_config={
        'torso': list(range(10)),        # First 10 keypoints
        'left_arm': list(range(10, 20)), # Next 10 keypoints
        'right_arm': list(range(20, 30)) # Next 10 keypoints
    },
    connections_config={
        'torso': [[i, i+1] for i in range(9)],
        'left_arm': [[i, i+1] for i in range(10, 19)],
        'right_arm': [[i, i+1] for i in range(20, 29)]
    },
    centers_config={'torso': 5, 'left_arm': 5, 'right_arm': 5}
)

# Use custom format
model = UniSTGCN(
    parts=[
        ('my_format', 'torso'),
        ('my_format', 'left_arm'),
        ('my_format', 'right_arm')
    ],
    keypoint_format=custom_format,
    auto_convert=True
)
```

### Custom Graph Construction

```python
from uni_GCN import Graph, UniSTGCN

# Create custom graph with spatial partitioning
graph = Graph(layout='body', strategy='spatial', max_hop=2)

# Use in model
model = UniSTGCN(
    parts=['body'],
    graph_strategy='spatial',
    adaptive_graph=True,    # Learnable adjacency matrices
    max_hop=2
)
```

### Feature Extraction

```python
# Get intermediate features
output, features = model(keypoints, return_features=True)

# Available features:
# - {part}_projected: After linear projection
# - {part}_spatial: After spatial ST-GCN
# - {part}_temporal: After temporal ST-GCN
# - {part}_pooled: After vertex pooling
# - combined: Final concatenated features
```

### Individual Components

```python
from uni_GCN import STGCN_block, get_stgcn_chain

# Create spatial processing chain
spatial_gcn, out_dim = get_stgcn_chain(
    in_dim=64,
    level='spatial',
    kernel_size=(1, adjacency_size),
    A=adjacency_matrix,
    adaptive=True
)

# Create temporal processing chain
temporal_gcn, out_dim = get_stgcn_chain(
    in_dim=256,
    level='temporal',
    kernel_size=(5, adjacency_size),
    A=adjacency_matrix,
    adaptive=True
)
```

## 📝 Configuration Options

### Model Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `parts` | Body parts to model | `['body', 'left', 'right', 'face_all']` |
| `input_dim` | Input feature dimension | `3` (x,y,confidence) |
| `hidden_dim` | Hidden dimension | `64`, `128`, etc. |
| `graph_strategy` | Adjacency strategy | `'uniform'`, `'distance'`, `'spatial'` |
| `adaptive_graph` | Learnable adjacency | `True`/`False` |
| `max_hop` | Graph connectivity | `1`, `2`, etc. |
| `output_pooling` | Vertex pooling method | `'mean'`, `'max'`, `'none'` |

### Predefined Configurations

```python
configs = {
    'full': ['body', 'left', 'right', 'face_all'],     # Complete modeling
    'upper': ['body', 'left', 'right'],                # Upper body
    'hands': ['left', 'right'],                        # Hands only
    'body_only': ['body']                              # Body only
}
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
cd uni_GCN
python example.py
```

Test results include:
- ✅ Basic functionality
- ✅ Different configurations
- ✅ Graph strategies
- ✅ Individual components
- ✅ Performance benchmark

## 📈 Performance

**Benchmark Results** (on sample data):
- **Model Size**: ~2.1M parameters
- **Inference Speed**: ~15ms per batch (batch_size=4, seq_len=128)
- **Memory Efficient**: Shared weights between left/right hands

## 🔬 Technical Details

### ST-GCN Block Structure

```python
STGCN_Block:
├── Spatial GCN
│   ├── Conv2D (temporal)
│   ├── Graph Convolution (spatial)
│   └── BatchNorm + ReLU
├── Temporal CNN
│   ├── Conv2D (kernel_size=5)
│   ├── BatchNorm + Dropout
└── Residual Connection
```

### Part Interaction Mechanism

```python
# Hand features enhanced with body context
if part == 'left':
    gcn_feat += body_feat[..., left_wrist_idx].unsqueeze(-1)
elif part == 'right':
    gcn_feat += body_feat[..., right_wrist_idx].unsqueeze(-1)
elif part == 'face_all':
    gcn_feat += body_feat[..., head_idx].unsqueeze(-1)
```

## 🤝 Integration Examples

### With Sign Language Recognition

```python
from uni_GCN import UniSTGCN
import torch.nn as nn

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stgcn = UniSTGCN(parts=['body', 'left', 'right'])
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, keypoints):
        features = self.stgcn(keypoints)      # (B, T, 768)
        pooled = features.mean(dim=1)         # (B, 768)
        logits = self.classifier(pooled)      # (B, num_classes)
        return logits
```

### With Action Recognition

```python
class ActionRecognitionModel(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.stgcn = UniSTGCN(parts=['body'])
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_actions)

    def forward(self, keypoints):
        features = self.stgcn(keypoints)           # (B, T, 256)
        features = features.transpose(1, 2)        # (B, 256, T)
        pooled = self.temporal_pool(features)      # (B, 256, 1)
        pooled = pooled.squeeze(-1)                # (B, 256)
        return self.classifier(pooled)
```

## 📚 References

- **Uni-Sign Paper**: \"Uni-Sign: Toward Unified Sign Language Understanding at Scale\" (ICLR 2025)
- **ST-GCN**: \"Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition\" (AAAI 2018)
- **CoSign**: \"Exploring Co-occurrence Signals in Skeleton-based Continuous Sign Language Recognition\" (ICCV 2023)

## 🐛 Troubleshooting

### Common Issues

1. **Shape Mismatch**: Ensure keypoints follow format `(batch, time, vertices, features)`
2. **Missing Parts**: All parts specified in model initialization must be provided in input
3. **Graph Strategy**: Different strategies produce different adjacency matrix shapes

### Debug Tips

```python
# Check model info
print(model.get_part_info())
print(f\"Expected output dim: {model.get_output_dim()}\")

# Inspect intermediate features
output, features = model(keypoints, return_features=True)
for name, feat in features.items():
    print(f\"{name}: {feat.shape}\")
```

## 🛠️ Contributing

This module is extracted from the Uni-Sign project. For improvements or extensions:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Maintain backward compatibility

## 📄 License

This implementation is based on the Uni-Sign project. Please refer to the original project's license terms.

---

**Happy coding with Uni-GCN! 🚀**