import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecayTemporalAttention(nn.Module):
    """
    Decay-based temporal attention mechanism for Temporal Graph Networks.
    This implementation includes configurable decay factors to model how
    the importance of historical interactions decreases over time.
    """

    def __init__(self, time_dim, method="sum", decay_factor=0.1, adaptive_decay=False):
        super(DecayTemporalAttention, self).__init__()
        self.time_dim = time_dim
        self.method = method
        self.decay_factor = decay_factor
        self.adaptive_decay = adaptive_decay

        # Weight vector for computing temporal attention scores
        self.w = nn.Parameter(torch.ones(time_dim))

        # Adaptive decay parameters if enabled
        if adaptive_decay:
            self.decay_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim // 2),
                nn.ReLU(),
                nn.Linear(time_dim // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, memory, message_time, current_time):
        """
        Apply decay-based temporal attention to memory

        Args:
            memory: Node memory [batch_size, memory_dim]
            message_time: Time when the message was created [batch_size]
            current_time: Current time [batch_size]

        Returns:
            weighted_memory: Memory weighted by temporal decay [batch_size, memory_dim]
        """
        # Compute time difference
        delta_t = current_time - message_time

        # Ensure delta_t is non-negative (time moves forward)
        delta_t = torch.clamp(delta_t, min=0.0)

        if self.adaptive_decay:
            # Learn decay rate based on time features
            time_features = self.w.unsqueeze(0) * delta_t.unsqueeze(1)
            adaptive_decay_rate = self.decay_mlp(time_features).squeeze(-1)
            score = torch.exp(-adaptive_decay_rate * delta_t)
        else:
            # Fixed decay rate
            score = torch.exp(-self.decay_factor * delta_t)

        # Apply decay-based attention
        if self.method == "sum":
            weighted_memory = memory * score.unsqueeze(1)
        else:  # product
            weighted_memory = memory * score.unsqueeze(1)

        return weighted_memory, score


class DecayTGNMemoryModule(nn.Module):
    """
    Enhanced TGN Memory Module with decay-based temporal attention.
    This module incorporates temporal decay when updating and retrieving memories.
    """

    def __init__(
        self,
        num_nodes,
        memory_dim,
        message_dim,
        time_dim,
        decay_factor=0.1,
        adaptive_decay=False,
    ):
        super(DecayTGNMemoryModule, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim
        self.decay_factor = decay_factor

        # Decay attention mechanism
        self.decay_attention = DecayTemporalAttention(
            time_dim=time_dim, decay_factor=decay_factor, adaptive_decay=adaptive_decay
        )

        # Memory storage
        self.memory = nn.Parameter(
            torch.zeros(num_nodes, memory_dim), requires_grad=False
        )
        self.last_update = nn.Parameter(torch.zeros(num_nodes), requires_grad=False)

        # Message processing
        self.message_processor = nn.Sequential(
            nn.Linear(message_dim, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )

        # Memory update mechanism
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)

    def reset_state(self):
        """Reset memory state"""
        with torch.no_grad():
            self.memory.fill_(0.0)
            self.last_update.fill_(0.0)

    def get_memory(self, node_ids, current_time=None):
        """
        Get memory for specified nodes with optional temporal decay

        Args:
            node_ids: Node IDs to retrieve memory for [batch_size]
            current_time: Current time for decay calculation [batch_size]

        Returns:
            node_memory: Retrieved memories [batch_size, memory_dim]
            decay_scores: Decay scores if current_time provided [batch_size]
        """
        # Clamp node IDs to valid range
        node_ids = torch.clamp(node_ids, 0, self.num_nodes - 1)

        # Get raw memory
        node_memory = self.memory[node_ids]

        # Apply temporal decay if current time is provided
        if current_time is not None:
            last_update_times = self.last_update[node_ids]
            decayed_memory, decay_scores = self.decay_attention(
                node_memory, last_update_times, current_time
            )
            return decayed_memory, decay_scores
        else:
            return node_memory, None

    def update_memory(self, node_ids, messages, timestamps):
        """
        Update memory for specified nodes with new messages

        Args:
            node_ids: Node IDs to update [batch_size]
            messages: New messages [batch_size, message_dim]
            timestamps: Message timestamps [batch_size]
        """
        # Clamp node IDs to valid range
        node_ids = torch.clamp(node_ids, 0, self.num_nodes - 1)

        # Process messages
        processed_messages = self.message_processor(messages)

        # Get current memory with decay
        current_memory, _ = self.get_memory(node_ids, timestamps)
        # Update memory using GRU
        with torch.no_grad():
            for i, node_id in enumerate(node_ids):
                node_idx = node_id.item()
                new_memory = self.memory_updater(
                    processed_messages[i : i + 1], current_memory[i : i + 1]
                )
                self.memory[node_idx] = new_memory.squeeze(0)
                self.last_update[node_idx] = timestamps[i]


class DecayTemporalGraphNetwork(nn.Module):
    """
    Enhanced TGN with decay-based temporal attention mechanisms.
    This model incorporates temporal decay in both memory management and attention layers.
    """

    def __init__(
        self,
        num_nodes,
        node_feat_dim,
        edge_feat_dim,
        memory_dim,
        time_dim,
        embedding_dim,
        decay_factor=0.1,
        adaptive_decay=False,
    ):
        super(DecayTemporalGraphNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.decay_factor = decay_factor

        # Memory module with decay
        self.memory_module = DecayTGNMemoryModule(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            message_dim=memory_dim,
            time_dim=time_dim,
            decay_factor=decay_factor,
            adaptive_decay=adaptive_decay,
        )

        # Node and edge embedding layers
        self.node_embedding = nn.Linear(node_feat_dim, memory_dim)
        self.edge_embedding = nn.Linear(edge_feat_dim, memory_dim)

        # Temporal attention layers with decay
        self.temporal_attention = DecayTemporalAttention(
            time_dim=time_dim, decay_factor=decay_factor, adaptive_decay=adaptive_decay
        )

        # Final embedding projection
        self.embedding_projection = nn.Sequential(
            nn.Linear(memory_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Link predictor
        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

        # Time encoder
        self.time_encoder = nn.Linear(1, time_dim)

    def compute_embeddings(self, node_ids, node_features, timestamps):
        """
        Compute node embeddings with temporal decay

        Args:
            node_ids: Node IDs [batch_size]
            node_features: Node features [batch_size, node_feat_dim]
            timestamps: Current timestamps [batch_size]

        Returns:
            embeddings: Node embeddings [batch_size, embedding_dim]
            decay_info: Dictionary with decay information
        """
        # Get memory with decay
        memory, decay_scores = self.memory_module.get_memory(node_ids, timestamps)

        # Process node features
        node_emb = self.node_embedding(node_features)

        # Apply temporal attention to combine memory and current features
        time_features = self.time_encoder(timestamps.unsqueeze(1))
        attended_memory, attention_scores = self.temporal_attention(
            memory, self.memory_module.last_update[node_ids], timestamps
        )

        # Combine memory and current features
        combined_features = torch.cat([attended_memory, node_emb], dim=1)
        embeddings = self.embedding_projection(combined_features)

        # Collect decay information
        decay_info = {
            "decay_scores": decay_scores,
            "attention_scores": attention_scores,
            "memory_age": timestamps - self.memory_module.last_update[node_ids],
        }

        return embeddings, decay_info

    def forward(
        self, src_ids, dst_ids, src_features, dst_features, timestamps, edge_features
    ):
        """
        Forward pass for link prediction with decay analysis

        Args:
            src_ids: Source node IDs [batch_size]
            dst_ids: Destination node IDs [batch_size]
            src_features: Source node features [batch_size, node_feat_dim]
            dst_features: Destination node features [batch_size, node_feat_dim]
            timestamps: Edge timestamps [batch_size]
            edge_features: Edge features [batch_size, edge_feat_dim]

        Returns:
            link_probs: Link prediction probabilities [batch_size, 1]
            decay_analysis: Dictionary with decay analysis results
        """
        # Update memory with current interactions
        edge_emb = self.edge_embedding(edge_features)
        self.memory_module.update_memory(src_ids, edge_emb, timestamps)
        self.memory_module.update_memory(dst_ids, edge_emb, timestamps)

        # Compute embeddings with decay information
        src_embeddings, src_decay_info = self.compute_embeddings(
            src_ids, src_features, timestamps
        )
        dst_embeddings, dst_decay_info = self.compute_embeddings(
            dst_ids, dst_features, timestamps
        )

        # Predict link probability
        link_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        link_probs = self.link_predictor(link_features)

        # Compile decay analysis
        decay_analysis = {
            "src_decay_scores": src_decay_info["decay_scores"],
            "dst_decay_scores": dst_decay_info["decay_scores"],
            "src_attention_scores": src_decay_info["attention_scores"],
            "dst_attention_scores": dst_decay_info["attention_scores"],
            "src_memory_age": src_decay_info["memory_age"],
            "dst_memory_age": dst_decay_info["memory_age"],
            "avg_decay_score": (
                (
                    src_decay_info["decay_scores"].mean()
                    + dst_decay_info["decay_scores"].mean()
                )
                / 2
                if src_decay_info["decay_scores"] is not None
                else None
            ),
        }

        return link_probs, decay_analysis

    def reset_memory(self):
        """Reset memory state"""
        self.memory_module.reset_state()

    def get_decay_statistics(self, node_ids, current_time):
        """
        Get decay statistics for analysis

        Args:
            node_ids: Node IDs to analyze [num_nodes]
            current_time: Current timestamp

        Returns:
            stats: Dictionary with decay statistics
        """
        with torch.no_grad():
            memory, decay_scores = self.memory_module.get_memory(
                node_ids, torch.full_like(node_ids, current_time, dtype=torch.float)
            )

            memory_ages = current_time - self.memory_module.last_update[node_ids]

            stats = {
                "decay_scores": decay_scores,
                "memory_ages": memory_ages,
                "avg_decay_score": (
                    decay_scores.mean() if decay_scores is not None else None
                ),
                "min_decay_score": (
                    decay_scores.min() if decay_scores is not None else None
                ),
                "max_decay_score": (
                    decay_scores.max() if decay_scores is not None else None
                ),
                "avg_memory_age": memory_ages.mean(),
                "memory_variance": memory.var(dim=1).mean(),
            }

        return stats


def create_decay_tgn_model(
    num_nodes,
    node_feat_dim,
    edge_feat_dim,
    decay_factor=0.1,
    adaptive_decay=False,
    device="cpu",
):
    """
    Factory function to create a DecayTemporalGraphNetwork with standard parameters

    Args:
        num_nodes: Number of nodes in the graph
        node_feat_dim: Dimension of node features
        edge_feat_dim: Dimension of edge features
        decay_factor: Decay rate for temporal attention
        adaptive_decay: Whether to use adaptive decay rates
        device: Device to place the model on

    Returns:
        model: Configured DecayTemporalGraphNetwork
    """
    model = DecayTemporalGraphNetwork(
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        memory_dim=64,
        time_dim=8,
        embedding_dim=64,
        decay_factor=decay_factor,
        adaptive_decay=adaptive_decay,
    ).to(device)

    return model


def analyze_decay_effects(
    model, data_loader, decay_factors=[0.01, 0.05, 0.1, 0.2, 0.5]
):
    """
    Analyze the effects of different decay factors on model performance

    Args:
        model: DecayTemporalGraphNetwork model
        data_loader: DataLoader with temporal graph data
        decay_factors: List of decay factors to test

    Returns:
        results: Dictionary with analysis results for each decay factor
    """
    results = {}

    for decay_factor in decay_factors:
        # Update model decay factor
        model.decay_factor = decay_factor
        model.memory_module.decay_factor = decay_factor
        model.memory_module.decay_attention.decay_factor = decay_factor
        model.temporal_attention.decay_factor = decay_factor

        # Reset memory for fair comparison
        model.reset_memory()

        batch_results = []

        with torch.no_grad():
            for batch in data_loader:
                src_ids = batch["src_idx"].squeeze()
                dst_ids = batch["dst_idx"].squeeze()
                src_features = torch.randn(len(src_ids), model.node_feat_dim)
                dst_features = torch.randn(len(dst_ids), model.node_feat_dim)
                timestamps = batch["timestamp"].squeeze()
                edge_features = batch["edge_features"]

                # Forward pass
                link_probs, decay_analysis = model(
                    src_ids,
                    dst_ids,
                    src_features,
                    dst_features,
                    timestamps,
                    edge_features,
                )

                batch_results.append(
                    {
                        "link_probs": link_probs.mean().item(),
                        "avg_decay_score": (
                            decay_analysis["avg_decay_score"].item()
                            if decay_analysis["avg_decay_score"] is not None
                            else 0.0
                        ),
                        "avg_memory_age": decay_analysis["src_memory_age"]
                        .mean()
                        .item(),
                    }
                )

                # Only analyze first few batches for efficiency
                if len(batch_results) >= 5:
                    break

        # Aggregate results
        results[decay_factor] = {
            "avg_link_prob": np.mean([r["link_probs"] for r in batch_results]),
            "avg_decay_score": np.mean([r["avg_decay_score"] for r in batch_results]),
            "avg_memory_age": np.mean([r["avg_memory_age"] for r in batch_results]),
        }

    return results
