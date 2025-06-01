#!/usr/bin/env python3
"""
Decay Factor Implementation Demo for TGN Model

This script demonstrates the decay factor implementation and shows
how different decay values affect the temporal attention mechanism.
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from decay_tgn import DecayTemporalGraphNetwork, create_decay_tgn_model, analyze_decay_effects

def demonstrate_decay_curves():
    """Demonstrate how different decay factors affect attention over time"""
    print("=" * 80)
    print("DECAY FACTOR DEMONSTRATION")
    print("=" * 80)
    
    # Time differences to analyze
    time_diffs = np.linspace(0, 20, 100)
    decay_factors = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    print("\n1. Theoretical Decay Curves:")
    print("   Shows how attention weights decay over time for different factors")
    
    plt.figure(figsize=(12, 8))
    
    # Plot decay curves
    for decay_factor in decay_factors:
        decay_weights = np.exp(-decay_factor * time_diffs)
        plt.plot(time_diffs, decay_weights, label=f'λ = {decay_factor}', linewidth=2)
    
    plt.title('Temporal Decay: Attention Weight vs Time Difference', fontsize=14, fontweight='bold')
    plt.xlabel('Time Difference (units)')
    plt.ylabel('Attention Weight Multiplier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Add annotations
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% retention')
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='10% retention')
    
    plt.tight_layout()
    plt.savefig('decay_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print decay statistics
    print("\n2. Decay Statistics:")
    print(f"{'Factor':<8} {'Half-life':<10} {'10% at':<10} {'1% at':<10}")
    print("-" * 40)
    
    for decay_factor in decay_factors:
        half_life = np.log(2) / decay_factor
        ten_percent_time = -np.log(0.1) / decay_factor
        one_percent_time = -np.log(0.01) / decay_factor
        
        print(f"{decay_factor:<8} {half_life:<10.1f} {ten_percent_time:<10.1f} {one_percent_time:<10.1f}")

def test_multiple_decay_models():
    """Test multiple decay models with different factors"""
    print("\n3. Multi-Model Decay Testing:")
    
    # Model parameters
    num_nodes = 1000
    node_feat_dim = 32
    edge_feat_dim = 8
    batch_size = 16
    
    decay_factors = [0.01, 0.05, 0.1, 0.2, 0.5]
    models = {}
    results = {}
    
    print("   Creating models with different decay factors...")
    
    # Create models
    for decay_factor in decay_factors:
        try:
            model = create_decay_tgn_model(
                num_nodes=num_nodes,
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                decay_factor=decay_factor,
                device='cpu'
            )
            models[decay_factor] = model
            print(f"   ✓ Created model with λ = {decay_factor}")
        except Exception as e:
            print(f"   ✗ Failed to create model with λ = {decay_factor}: {e}")
    
    # Test models with synthetic data
    print("\n   Testing models with synthetic temporal data...")
    
    # Generate synthetic temporal interactions
    src_ids = torch.randint(0, num_nodes, (batch_size,))
    dst_ids = torch.randint(0, num_nodes, (batch_size,))
    src_features = torch.randn(batch_size, node_feat_dim)
    dst_features = torch.randn(batch_size, node_feat_dim)
    edge_features = torch.randn(batch_size, edge_feat_dim)
    
    # Test with different time scenarios
    time_scenarios = {
        'recent': torch.tensor([100.0] * batch_size),      # All recent interactions
        'mixed': torch.tensor([100.0, 95.0, 80.0, 50.0] * (batch_size // 4 + 1))[:batch_size],  # Mixed ages
        'old': torch.tensor([10.0] * batch_size)           # All old interactions
    }
    
    for scenario_name, timestamps in time_scenarios.items():
        print(f"\n   Scenario: {scenario_name.upper()} interactions")
        print(f"   {'Factor':<8} {'Link Prob':<12} {'Decay Score':<12} {'Memory Age':<12}")
        print("   " + "-" * 50)
        
        for decay_factor, model in models.items():
            try:
                with torch.no_grad():
                    # Reset model memory
                    model.reset_memory()
                    
                    # Forward pass
                    link_probs, decay_analysis = model(
                        src_ids, dst_ids, src_features, dst_features,
                        timestamps, edge_features
                    )
                    
                    avg_link_prob = link_probs.mean().item()
                    avg_decay_score = decay_analysis['avg_decay_score'].item() if decay_analysis['avg_decay_score'] is not None else 0.0
                    avg_memory_age = decay_analysis['src_memory_age'].mean().item()
                    
                    print(f"   {decay_factor:<8} {avg_link_prob:<12.4f} {avg_decay_score:<12.4f} {avg_memory_age:<12.2f}")
                    
                    # Store results
                    if decay_factor not in results:
                        results[decay_factor] = {}
                    results[decay_factor][scenario_name] = {
                        'link_prob': avg_link_prob,
                        'decay_score': avg_decay_score,
                        'memory_age': avg_memory_age
                    }
                    
            except Exception as e:
                print(f"   {decay_factor:<8} Error: {str(e)[:30]}...")
    
    return results

def visualize_model_comparison(results):
    """Visualize comparison between different decay models"""
    if not results:
        print("No results to visualize")
        return
    
    print("\n4. Model Comparison Visualization:")
    
    decay_factors = list(results.keys())
    scenarios = ['recent', 'mixed', 'old']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot link probabilities for each scenario
    for i, scenario in enumerate(scenarios):
        link_probs = [results[df][scenario]['link_prob'] for df in decay_factors if scenario in results[df]]
        valid_factors = [df for df in decay_factors if scenario in results[df]]
        
        axes[i].bar([str(f) for f in valid_factors], link_probs, alpha=0.7, 
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(valid_factors)])
        axes[i].set_title(f'{scenario.title()} Interactions')
        axes[i].set_xlabel('Decay Factor')
        axes[i].set_ylabel('Average Link Probability')
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Link Prediction Performance Across Decay Factors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_adaptive_decay():
    """Demonstrate adaptive decay mechanism"""
    print("\n5. Adaptive Decay Demonstration:")
    
    try:
        # Create adaptive decay model
        adaptive_model = create_decay_tgn_model(
            num_nodes=100,
            node_feat_dim=32,
            edge_feat_dim=8,
            decay_factor=0.1,  # Base decay factor
            adaptive_decay=True,
            device='cpu'
        )
        
        print("   ✓ Created adaptive decay model")
        
        # Test with varying time patterns
        batch_size = 8
        src_ids = torch.randint(0, 100, (batch_size,))
        dst_ids = torch.randint(0, 100, (batch_size,))
        src_features = torch.randn(batch_size, 32)
        dst_features = torch.randn(batch_size, 32)
        edge_features = torch.randn(batch_size, 8)
        
        # Different temporal patterns
        patterns = {
            'regular': torch.linspace(90, 100, batch_size),
            'burst': torch.tensor([99.9, 99.8, 99.7, 50.0, 49.0, 48.0, 10.0, 9.0]),
            'decay': torch.tensor([100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 1.0])
        }
        
        print("\n   Testing adaptive decay with different temporal patterns:")
        print(f"   {'Pattern':<10} {'Avg Link Prob':<15} {'Avg Decay Score':<15}")
        print("   " + "-" * 45)
        
        for pattern_name, timestamps in patterns.items():
            with torch.no_grad():
                adaptive_model.reset_memory()
                
                link_probs, decay_analysis = adaptive_model(
                    src_ids, dst_ids, src_features, dst_features,
                    timestamps, edge_features
                )
                
                avg_link_prob = link_probs.mean().item()
                avg_decay_score = decay_analysis['avg_decay_score'].item() if decay_analysis['avg_decay_score'] is not None else 0.0
                
                print(f"   {pattern_name:<10} {avg_link_prob:<15.4f} {avg_decay_score:<15.4f}")
        
    except Exception as e:
        print(f"   ✗ Error testing adaptive decay: {e}")

def main():
    """Main demonstration function"""
    print("Decay Factor Implementation Demo for TGN Model")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. Show decay curves
        demonstrate_decay_curves()
        
        # 2. Test multiple models
        results = test_multiple_decay_models()
        
        # 3. Visualize comparison
        visualize_model_comparison(results)
        
        # 4. Demonstrate adaptive decay
        demonstrate_adaptive_decay()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. Lower decay factors (0.01-0.05) preserve long-term memory")
        print("2. Higher decay factors (0.2-0.5) focus on recent interactions")
        print("3. Medium decay factors (0.1) provide balanced temporal modeling")
        print("4. Adaptive decay can learn optimal rates for different patterns")
        print("5. Decay factors significantly impact model behavior and performance")
        
        print("\nFiles generated:")
        print("- decay_curves.png: Theoretical decay curves")
        print("- model_comparison.png: Model performance comparison")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
