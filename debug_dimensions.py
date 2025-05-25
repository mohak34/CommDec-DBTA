#!/usr/bin/env python3
"""
Debug script to identify the dimension mismatch in TGN model
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_tgn import TemporalGraphNetwork

def debug_model_dimensions():
    """Debug the dimensional issues in the TGN model"""
    
    print("Debugging TGN model dimensions...")
    
    # Model parameters based on the conversation
    num_nodes = 1000  # This seems to be inconsistent with actual data (54075)
    node_feat_dim = 50
    edge_feat_dim = 10
    memory_dim = 100
    time_dim = 10
    embedding_dim = 100
    message_dim = 100
    n_layers = 2
    n_heads = 2
    dropout = 0.1
    
    print(f"Model config:")
    print(f"  num_nodes: {num_nodes}")
    print(f"  node_feat_dim: {node_feat_dim}")
    print(f"  edge_feat_dim: {edge_feat_dim}")
    print(f"  memory_dim: {memory_dim}")
    print(f"  time_dim: {time_dim}")
    print(f"  embedding_dim: {embedding_dim}")
    
    try:
        # Initialize model
        model = TemporalGraphNetwork(
            num_nodes=num_nodes,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            embedding_dim=embedding_dim,
            message_dim=message_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        print("Model initialized successfully")
        
        # Test with dummy data similar to actual usage
        batch_size = 2
        
        # Create dummy inputs
        src_ids = torch.randint(0, num_nodes, (batch_size,))
        dst_ids = torch.randint(0, num_nodes, (batch_size,))
        src_features = torch.randn(batch_size, node_feat_dim)
        dst_features = torch.randn(batch_size, node_feat_dim)
        timestamps = torch.randn(batch_size)
        edge_features = torch.randn(batch_size, edge_feat_dim)
        
        print(f"\nInput shapes:")
        print(f"  src_ids: {src_ids.shape}")
        print(f"  dst_ids: {dst_ids.shape}")
        print(f"  src_features: {src_features.shape}")
        print(f"  dst_features: {dst_features.shape}")
        print(f"  timestamps: {timestamps.shape}")
        print(f"  edge_features: {edge_features.shape}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        
        with torch.no_grad():
            output = model(
                src_ids=src_ids,
                dst_ids=dst_ids,
                src_features=src_features,
                dst_features=dst_features,
                timestamps=timestamps,
                edge_features=edge_features
            )
            
            link_prob, src_embeddings, dst_embeddings = output
            print(f"\nOutput shapes:")
            print(f"  link_prob: {link_prob.shape}")
            print(f"  src_embeddings: {src_embeddings.shape}")
            print(f"  dst_embeddings: {dst_embeddings.shape}")
            
        print("✓ Forward pass successful!")
        
        # Test with neighbor data (this is likely where the error occurs)
        print("\nTesting with neighbor data...")
        n_neighbors = 10
        
        src_neighbor_ids = torch.randint(0, num_nodes, (batch_size, n_neighbors))
        dst_neighbor_ids = torch.randint(0, num_nodes, (batch_size, n_neighbors))
        src_neighbor_times = torch.randn(batch_size, n_neighbors)
        dst_neighbor_times = torch.randn(batch_size, n_neighbors)
        src_neighbor_features = torch.randn(batch_size, n_neighbors, node_feat_dim)
        dst_neighbor_features = torch.randn(batch_size, n_neighbors, node_feat_dim)
        src_edge_features = torch.randn(batch_size, n_neighbors, edge_feat_dim)
        dst_edge_features = torch.randn(batch_size, n_neighbors, edge_feat_dim)
        
        print(f"Neighbor input shapes:")
        print(f"  src_neighbor_ids: {src_neighbor_ids.shape}")
        print(f"  src_neighbor_features: {src_neighbor_features.shape}")
        print(f"  src_edge_features: {src_edge_features.shape}")
        
        with torch.no_grad():
            output = model(
                src_ids=src_ids,
                dst_ids=dst_ids,
                src_features=src_features,
                dst_features=dst_features,
                timestamps=timestamps,
                edge_features=edge_features,
                src_neighbor_ids=src_neighbor_ids,
                dst_neighbor_ids=dst_neighbor_ids,
                src_neighbor_times=src_neighbor_times,
                dst_neighbor_times=dst_neighbor_times,
                src_neighbor_features=src_neighbor_features,
                dst_neighbor_features=dst_neighbor_features,
                src_edge_features=src_edge_features,
                dst_edge_features=dst_edge_features
            )
            
            link_prob, src_embeddings, dst_embeddings = output
            print(f"Output shapes with neighbors:")
            print(f"  link_prob: {link_prob.shape}")
            print(f"  src_embeddings: {src_embeddings.shape}")
            print(f"  dst_embeddings: {dst_embeddings.shape}")
            
        print("✓ Forward pass with neighbors successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to identify the specific layer causing issues
        print("\nDebugging individual components...")
        
        try:
            # Test the TemporalAttentionLayer specifically
            from enhanced_tgn import TemporalAttentionLayer
            
            attention_layer = TemporalAttentionLayer(
                node_feat_dim=memory_dim,
                edge_feat_dim=memory_dim,
                time_feat_dim=time_dim,
                memory_dim=memory_dim,
                output_dim=embedding_dim,
                n_heads=n_heads
            )
            
            print(f"\nTemporalAttentionLayer dimensions:")
            print(f"  query_dim: {attention_layer.query_dim}")
            print(f"  key_dim: {attention_layer.key_dim}")
            print(f"  value_dim: {attention_layer.value_dim}")
            
            # Test attention layer with specific inputs
            node_features = torch.randn(batch_size, memory_dim)
            node_time_features = torch.randn(batch_size, time_dim)
            neighbor_features = torch.randn(batch_size, n_neighbors, memory_dim)
            neighbor_time_features = torch.randn(batch_size, n_neighbors, time_dim)
            edge_features = torch.randn(batch_size, n_neighbors, memory_dim)
            
            print(f"\nAttention layer input shapes:")
            print(f"  node_features: {node_features.shape}")
            print(f"  node_time_features: {node_time_features.shape}")
            print(f"  neighbor_features: {neighbor_features.shape}")
            print(f"  neighbor_time_features: {neighbor_time_features.shape}")
            print(f"  edge_features: {edge_features.shape}")
            
            with torch.no_grad():
                attention_output = attention_layer(
                    node_features, node_time_features, neighbor_features,
                    neighbor_time_features, edge_features
                )
                print(f"  attention_output: {attention_output.shape}")
                
        except Exception as e2:
            print(f"❌ TemporalAttentionLayer error: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_model_dimensions()
