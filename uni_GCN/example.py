"""
Example usage of Uni-GCN module for skeletal data processing.

This script demonstrates how to use the ST-GCN components for
processing multi-part skeletal keypoints data.
"""

import torch
import numpy as np
from .uni_stgcn import UniSTGCN, MultiPartSTGCN
from .graph import Graph


def generate_sample_keypoints():
    """Generate sample keypoint data for testing"""
    batch_size = 2
    seq_len = 64

    # Generate keypoints for different body parts
    keypoints = {}

    # Body: 9 joints
    keypoints['body'] = torch.randn(batch_size, seq_len, 9, 3)

    # Left hand: 21 joints
    keypoints['left'] = torch.randn(batch_size, seq_len, 21, 3)

    # Right hand: 21 joints
    keypoints['right'] = torch.randn(batch_size, seq_len, 21, 3)

    # Face: 18 joints
    keypoints['face_all'] = torch.randn(batch_size, seq_len, 18, 3)

    return keypoints


def test_basic_usage():
    """Test basic UniSTGCN usage"""
    print("=== Testing Basic UniSTGCN Usage ===")

    # Create model
    model = UniSTGCN(
        parts=['body', 'left', 'right', 'face_all'],
        input_dim=3,
        hidden_dim=64,
        graph_strategy='distance',
    )

    print(f"Model output dimension: {model.get_output_dim()}")
    print(f"Part information: {model.get_part_info()}")

    # Generate sample data
    keypoints = generate_sample_keypoints()

    # Forward pass
    with torch.no_grad():
        output = model(keypoints)
        print(f"Output shape: {output.shape}")

        # Test with intermediate features
        output, features = model(keypoints, return_features=True)
        print(f"Intermediate features keys: {list(features.keys())}")

    print("✓ Basic usage test passed\n")


def test_different_configurations():
    """Test different model configurations"""
    print("=== Testing Different Configurations ===")

    configs = ['full', 'upper', 'hands', 'body_only']

    for config in configs:
        print(f"Testing config: {config}")

        model = MultiPartSTGCN(config=config, hidden_dim=32)

        # Generate appropriate keypoints
        keypoints = generate_sample_keypoints()

        # Filter keypoints based on config
        if config == 'upper':
            keypoints = {
                k: v for k, v in keypoints.items() if k in ['body', 'left', 'right']
            }
        elif config == 'hands':
            keypoints = {k: v for k, v in keypoints.items() if k in ['left', 'right']}
        elif config == 'body_only':
            keypoints = {k: v for k, v in keypoints.items() if k in ['body']}

        with torch.no_grad():
            output = model(keypoints)
            print(f"  Output shape: {output.shape}")
            print(f"  Output dim: {model.get_output_dim()}")

    print("✓ Configuration tests passed\n")


def test_graph_strategies():
    """Test different graph adjacency strategies"""
    print("=== Testing Graph Strategies ===")

    strategies = ['uniform', 'distance', 'spatial']

    for strategy in strategies:
        print(f"Testing strategy: {strategy}")

        model = UniSTGCN(parts=['body'], graph_strategy=strategy, hidden_dim=32)

        keypoints = {'body': torch.randn(1, 32, 9, 3)}

        with torch.no_grad():
            output = model(keypoints)
            print(f"  Output shape: {output.shape}")

    print("✓ Graph strategy tests passed\n")


def test_individual_components():
    """Test individual ST-GCN components"""
    print("=== Testing Individual Components ===")

    from .stgcn_block import GCN_unit, STGCN_block, get_stgcn_chain

    # Test Graph construction
    print("Testing Graph construction...")
    layouts = ['body', 'left', 'face_all']

    for layout in layouts:
        graph = Graph(layout=layout, strategy='distance')
        print(f"  {layout}: {graph.num_node} nodes, adjacency shape: {graph.A.shape}")

    # Test ST-GCN blocks
    print("\nTesting ST-GCN blocks...")

    # Create sample adjacency matrix
    graph = Graph(layout='body', strategy='distance')
    A = torch.tensor(graph.A, dtype=torch.float32)

    # Test GCN unit
    gcn_unit = GCN_unit(3, 64, A.size(0), A)
    sample_input = torch.randn(2, 3, 32, 9)  # (B, C, T, V)

    with torch.no_grad():
        gcn_output = gcn_unit(sample_input)
        print(f"  GCN unit output shape: {gcn_output.shape}")

    # Test STGCN block
    stgcn_block = STGCN_block(64, 128, (5, A.size(0)), A)

    with torch.no_grad():
        stgcn_output = stgcn_block(gcn_output)
        print(f"  STGCN block output shape: {stgcn_output.shape}")

    # Test ST-GCN chain
    spatial_chain, out_dim = get_stgcn_chain(3, 'spatial', (1, A.size(0)), A, True)

    with torch.no_grad():
        chain_output = spatial_chain(sample_input)
        print(f"  STGCN chain output shape: {chain_output.shape}")

    print("✓ Individual component tests passed\n")


def benchmark_performance():
    """Benchmark model performance"""
    print("=== Performance Benchmark ===")

    import time

    model = UniSTGCN(parts=['body', 'left', 'right'])
    keypoints = {
        'body': torch.randn(4, 128, 9, 3),
        'left': torch.randn(4, 128, 21, 3),
        'right': torch.randn(4, 128, 21, 3),
    }

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(keypoints)

    # Benchmark
    num_runs = 20
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            output = model(keypoints)

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs

    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Output shape: {output.shape}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("✓ Performance benchmark completed\n")


def main():
    """Run all tests"""
    print("🚀 Starting Uni-GCN Tests\n")

    try:
        test_basic_usage()
        test_different_configurations()
        test_graph_strategies()
        test_individual_components()
        benchmark_performance()

        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
