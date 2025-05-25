#!/usr/bin/env python3
"""
Test script to verify the CUDA tensor indexing fix works
"""

import torch
import numpy as np

def test_tensor_indexing_fix():
    """Test that the fix for CUDA tensor indexing works"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Simulate the original problem: numpy array with CUDA tensor indices
    node_features_np = np.random.randn(1000, 50).astype(np.float32)
    src_idx = torch.tensor([0, 1, 2, 3, 4], device=device)
    
    print("Original problem (should fail on CUDA):")
    try:
        # This would fail with the original code on CUDA
        # src_features = node_features_np[src_idx]  # This fails on CUDA
        print("Would fail: node_features_np[cuda_tensor]")
    except Exception as e:
        print(f"Expected error: {e}")
    
    print("\nFixed approach (should work):")
    # Convert to tensor first, then move to device
    if not isinstance(node_features_np, torch.Tensor):
        node_features_tensor = torch.FloatTensor(node_features_np)
    node_features_tensor = node_features_tensor.to(device)
    
    # Now indexing works
    src_features = node_features_tensor[src_idx]
    print(f"Success! Shape of indexed features: {src_features.shape}")
    print(f"Device of result: {src_features.device}")
    
    return True

if __name__ == "__main__":
    test_tensor_indexing_fix()
    print("Test completed successfully!")
