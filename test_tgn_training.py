#!/usr/bin/env python3
"""
Test script to verify the Enhanced TGN training works without CUDA device mismatch errors.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')
from enhanced_tgn import TemporalGraphNetwork

def test_tgn_training():
    """Test that TGN training works with proper device handling"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create dummy data
    num_nodes = 100
    node_feat_dim = 50
    edge_feat_dim = 10
    batch_size = 16
    
    # Create node features on CPU first, then move to device
    node_features_np = np.random.randn(num_nodes, node_feat_dim).astype(np.float32)
    node_features_tensor = torch.FloatTensor(node_features_np).to(device)
    
    print(f"Node features tensor device: {node_features_tensor.device}")
    
    # Create a dummy batch of data
    batch = {
        'src_idx': torch.randint(0, num_nodes, (batch_size,)).to(device),
        'dst_idx': torch.randint(0, num_nodes, (batch_size,)).to(device),
        'timestamp': torch.rand(batch_size).to(device),
        'edge_features': torch.randn(batch_size, edge_feat_dim).to(device),
        'label': torch.ones(batch_size, 1).to(device)
    }
    
    print(f"Batch src_idx device: {batch['src_idx'].device}")
    
    # Initialize TGN model
    model = TemporalGraphNetwork(
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        memory_dim=64,
        time_dim=8,
        embedding_dim=64,
        message_dim=64,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        use_memory=True,
        message_function='mlp',
        memory_updater='gru',
        aggregator='mean'
    ).to(device)
    
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test indexing node features with device tensors
    try:
        src_features = node_features_tensor[batch['src_idx']]
        dst_features = node_features_tensor[batch['dst_idx']]
        print(f"✓ Node feature indexing successful!")
        print(f"  - src_features shape: {src_features.shape}, device: {src_features.device}")
        print(f"  - dst_features shape: {dst_features.shape}, device: {dst_features.device}")
    except Exception as e:
        print(f"✗ Node feature indexing failed: {e}")
        return False
    
    # Test model forward pass
    try:
        model.train()
        model.reset_memory()
        
        # Generate negative samples
        neg_dst_idx = torch.randint(0, num_nodes, batch['src_idx'].size(), device=device)
        neg_dst_features = node_features_tensor[neg_dst_idx]
        
        # Forward pass for positive samples
        pos_prob, _, _ = model(
            src_ids=batch['src_idx'],
            dst_ids=batch['dst_idx'],
            src_features=src_features,
            dst_features=dst_features,
            timestamps=batch['timestamp'],
            edge_features=batch['edge_features']
        )
        
        print(f"✓ Model forward pass successful!")
        print(f"  - pos_prob shape: {pos_prob.shape}, device: {pos_prob.device}")
        
        # Test backward pass
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(pos_prob, batch['label'])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ Backward pass and optimization successful!")
        print(f"  - loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Enhanced TGN training with CUDA device compatibility...")
    success = test_tgn_training()
    
    if success:
        print("\n🎉 All tests passed! TGN training should work without CUDA device mismatch errors.")
    else:
        print("\n❌ Tests failed. There may still be device compatibility issues.")
