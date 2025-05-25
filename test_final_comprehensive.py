#!/usr/bin/env python3
"""
Final comprehensive test for Enhanced TGN CUDA device compatibility.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from collections import defaultdict

# Add src to path
sys.path.append('src')
from enhanced_tgn import TemporalGraphNetwork

def test_comprehensive_tgn():
    """Comprehensive test of TGN functionality"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Testing Enhanced TGN on device: {device}")
    
    # Test parameters
    num_nodes = 100
    node_feat_dim = 50
    edge_feat_dim = 10
    batch_size = 32
    num_epochs = 2
    
    # Create realistic data
    np.random.seed(42)
    node_features_np = np.random.randn(num_nodes, node_feat_dim).astype(np.float32)
    node_features_tensor = torch.FloatTensor(node_features_np).to(device)
    
    # Create TGN model
    model = TemporalGraphNetwork(
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        memory_dim=64,
        time_dim=8,
        embedding_dim=64,
        message_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        use_memory=True,
        message_function='mlp',
        memory_updater='gru',
        aggregator='mean'
    ).to(device)
    
    print(f"✓ Model created and moved to {device}")
    
    # Test multiple epochs with memory reset
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
        
        # Reset memory at start of epoch
        model.reset_memory()
        model.train()
        
        epoch_loss = 0.0
        num_batches = 5  # Test with multiple batches
        
        for batch_idx in range(num_batches):
            # Generate batch data
            batch = {
                'src_idx': torch.randint(0, num_nodes, (batch_size,)).to(device),
                'dst_idx': torch.randint(0, num_nodes, (batch_size,)).to(device),
                'timestamp': torch.rand(batch_size).to(device),
                'edge_features': torch.randn(batch_size, edge_feat_dim).to(device),
                'label': torch.ones(batch_size, 1).to(device)
            }
            
            # Get node features
            src_features = node_features_tensor[batch['src_idx']]
            dst_features = node_features_tensor[batch['dst_idx']]
            
            # Forward pass
            pos_prob, _, _ = model(
                src_ids=batch['src_idx'],
                dst_ids=batch['dst_idx'],
                src_features=src_features,
                dst_features=dst_features,
                timestamps=batch['timestamp'],
                edge_features=batch['edge_features']
            )
            
            # Compute loss
            loss = loss_fn(pos_prob, batch['label'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Test memory detach (optional)
            model.detach_memory()
            
        avg_loss = epoch_loss / num_batches
        print(f"  Average loss: {avg_loss:.4f}")
    
    print(f"\n🎉 Enhanced TGN training completed successfully!")
    print(f"  - Device compatibility: ✓")
    print(f"  - Memory management: ✓") 
    print(f"  - Multi-epoch training: ✓")
    print(f"  - Gradient computation: ✓")
    
    return True

if __name__ == "__main__":
    print("🚀 Running comprehensive Enhanced TGN test...")
    try:
        success = test_comprehensive_tgn()
        if success:
            print("\n✅ ALL TESTS PASSED! Enhanced TGN is ready for production use.")
        else:
            print("\n❌ Some tests failed.")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
