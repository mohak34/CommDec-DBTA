#!/usr/bin/env python3
"""
Simple test to reproduce the attention dimension mismatch
"""

import torch
import torch.nn as nn

def test_attention_dimensions():
    """Test the dimension mismatch in MultiheadAttention"""
    
    # These are the dimensions from the TemporalAttentionLayer
    node_feat_dim = 100
    edge_feat_dim = 100  # This is the problem - memory_dim instead of edge_feat_dim
    time_feat_dim = 10
    
    query_dim = node_feat_dim + time_feat_dim  # 110
    key_dim = node_feat_dim + edge_feat_dim + time_feat_dim  # 210
    value_dim = key_dim  # 210
    
    print(f"query_dim: {query_dim}")
    print(f"key_dim: {key_dim}")
    print(f"value_dim: {value_dim}")
    
    # This will fail because embed_dim != kdim and vdim
    try:
        attention = nn.MultiheadAttention(
            embed_dim=query_dim,  # 110
            kdim=key_dim,         # 210
            vdim=value_dim,       # 210
            num_heads=2,
            dropout=0.1
        )
        print("✓ MultiheadAttention created successfully")
        
        # Test with dummy data
        batch_size = 2
        n_neighbors = 10
        
        query = torch.randn(1, batch_size, query_dim)
        key = torch.randn(n_neighbors, batch_size, key_dim)
        value = torch.randn(n_neighbors, batch_size, value_dim)
        
        output, _ = attention(query, key, value)
        print(f"✓ Forward pass successful: {output.shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_attention_dimensions()
