import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import GRUCell
from collections import defaultdict

# Time Encoding Module
class TimeEncoder(nn.Module):
    """
    Time encoding module that maps time differences to a vector representation.
    Time features can be used to incorporate temporal information in the model.
    """
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()
        self.dimension = dimension
        self.w = nn.Linear(1, dimension)
        
        # Initialize with non-trainable fixed frequencies
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))).float().reshape(dimension, 1))
        self.w.bias = nn.Parameter(torch.zeros(dimension))
        
    def forward(self, t):
        # t has shape [batch_size, 1] or [batch_size]
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
            
        # Return shape [batch_size, dimension]
        return torch.cos(self.w(t))

# --------------------------------
# Message Function Modules
# --------------------------------
class MessageFunction(nn.Module):
    """
    Abstract message function class.
    A message function computes messages for each interaction.
    """
    def __init__(self):
        super(MessageFunction, self).__init__()
        
    def compute_message(self, raw_messages):
        """
        Computes messages from raw features
        """
        raise NotImplementedError("Message function must implement compute_message method")

class MLPMessageFunction(MessageFunction):
    """
    MLP-based message function that uses a multi-layer perceptron
    to transform raw message features.
    """
    def __init__(self, raw_message_dim, message_dim):
        super(MLPMessageFunction, self).__init__()
        self.out_channels = message_dim  # Needed for TGNMemory compatibility

        # MLP layers to transform raw message
        self.mlp = nn.Sequential(
            nn.Linear(raw_message_dim, raw_message_dim // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dim // 2, message_dim),
        )
    
    def compute_message(self, raw_messages):
        """
        Computes messages using an MLP
        
        Args:
            raw_messages: Tensor of shape [batch_size, raw_message_dim]
            
        Returns:
            messages: Tensor of shape [batch_size, message_dim]
        """
        return self.mlp(raw_messages)
        
    def forward(self, raw_messages):
        return self.compute_message(raw_messages)

class IdentityMessageFunction(MessageFunction):
    """
    Identity message function that returns raw message features unchanged.
    """
    def __init__(self, message_dim):
        super(IdentityMessageFunction, self).__init__()
        self.out_channels = message_dim  # Needed for TGNMemory compatibility
    
    def compute_message(self, raw_messages):
        return raw_messages
        
    def forward(self, raw_messages):
        return self.compute_message(raw_messages)

# --------------------------------
# Message Aggregator Modules
# --------------------------------
class MessageAggregator(nn.Module):
    """
    Abstract message aggregator class.
    A message aggregator aggregates messages from neighbors.
    """
    def __init__(self):
        super(MessageAggregator, self).__init__()
        
    def aggregate(self, messages):
        """
        Aggregates messages from neighbors
        """
        raise NotImplementedError("Message aggregator must implement aggregate method")

class MeanMessageAggregator(MessageAggregator):
    """
    Mean aggregator that takes the mean of messages.
    """
    def __init__(self):
        super(MeanMessageAggregator, self).__init__()
    
    def aggregate(self, messages):
        """
        Aggregates messages using mean
        
        Args:
            messages: List of tensors, each of shape [message_dim]
            
        Returns:
            aggregated_message: Tensor of shape [message_dim]
        """
        return torch.mean(messages, dim=0) if len(messages) > 0 else torch.zeros_like(messages[0])
        
    def forward(self, messages):
        return self.aggregate(messages)

class LSTMAggregator(MessageAggregator):
    """
    LSTM-based message aggregator. Processes messages sequentially with an LSTM.
    """
    def __init__(self, message_dim, memory_dim):
        super(LSTMAggregator, self).__init__()
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        self.lstm = nn.LSTM(input_size=message_dim, hidden_size=memory_dim, batch_first=True)
        
    def aggregate(self, messages):
        """
        Aggregates messages using LSTM
        
        Args:
            messages: Tensor of shape [n_messages, message_dim]
            
        Returns:
            aggregated_message: Tensor of shape [memory_dim]
        """
        # Convert messages to sequence for LSTM
        if len(messages) == 0:
            return torch.zeros(self.memory_dim, device=messages.device)
            
        messages_seq = messages.unsqueeze(0)  # [1, n_messages, message_dim]
        
        # Get the final hidden state
        _, (h_n, _) = self.lstm(messages_seq)
        return h_n.squeeze(0).squeeze(0)  # [memory_dim]
        
    def forward(self, messages):
        return self.aggregate(messages)

# --------------------------------
# Memory Updater Modules
# --------------------------------
class MemoryUpdater(nn.Module):
    """
    Abstract memory updater class.
    A memory updater updates memory based on messages.
    """
    def __init__(self):
        super(MemoryUpdater, self).__init__()
        
    def update_memory(self, messages, memory):
        """
        Updates memory based on messages
        """
        raise NotImplementedError("Memory updater must implement update_memory method")

class GRUMemoryUpdater(MemoryUpdater):
    """
    GRU-based memory updater that updates memory using a GRU cell.
    """
    def __init__(self, message_dim, memory_dim):
        super(GRUMemoryUpdater, self).__init__()
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        self.gru = GRUCell(input_size=message_dim, hidden_size=memory_dim)
        
    def update_memory(self, messages, memory):
        """
        Updates memory using GRU
        
        Args:
            messages: Tensor of shape [batch_size, message_dim]
            memory: Tensor of shape [batch_size, memory_dim]
            
        Returns:
            updated_memory: Tensor of shape [batch_size, memory_dim]
        """
        return self.gru(messages, memory)
        
    def forward(self, messages, memory):
        return self.update_memory(messages, memory)

class RNNMemoryUpdater(MemoryUpdater):
    """
    RNN-based memory updater that uses a standard RNN cell.
    """
    def __init__(self, message_dim, memory_dim):
        super(RNNMemoryUpdater, self).__init__()
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        self.rnn = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)
        
    def update_memory(self, messages, memory):
        """
        Updates memory using RNN
        
        Args:
            messages: Tensor of shape [batch_size, message_dim]
            memory: Tensor of shape [batch_size, memory_dim]
            
        Returns:
            updated_memory: Tensor of shape [batch_size, memory_dim]
        """
        return self.rnn(messages, memory)
        
    def forward(self, messages, memory):
        return self.update_memory(messages, memory)

# --------------------------------
# TGN Memory Module
# --------------------------------
class TGNMemoryModule(nn.Module):
    """
    Memory module for Temporal Graph Networks.
    Stores the memory state for each node in the graph and handles updates.
    """
    def __init__(self, num_nodes, memory_dim, message_dim, time_dim, 
                 message_function, memory_updater, aggregator):
        super(TGNMemoryModule, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_dim = message_dim
        self.time_dim = time_dim
        
        # Components for memory management
        self.message_function = message_function
        self.memory_updater = memory_updater
        self.aggregator = aggregator
        self.time_encoder = TimeEncoder(time_dim)
        
        # Initialize memory
        self.reset_state()
        
    def reset_state(self):
        """
        Reset memory state at the beginning of each epoch
        """
        self.memory = nn.Parameter(torch.zeros(self.num_nodes, self.memory_dim), 
                                   requires_grad=False)
        self.last_update = nn.Parameter(torch.zeros(self.num_nodes), 
                                        requires_grad=False)
        # Messages buffer for each node
        self.messages_buffer = defaultdict(list)
        
    def detach(self):
        """
        Detach memory from computation graph to prevent memory leaks
        """
        self.memory.detach_()
        
    def update_state(self, source_nodes, destination_nodes, timestamps, edge_features):
        """
        Update memory with new messages from temporal edges
        
        Args:
            source_nodes: Source node IDs of edges [batch_size]
            destination_nodes: Destination node IDs of edges [batch_size]
            timestamps: Timestamps of edges [batch_size]
            edge_features: Edge features [batch_size, edge_feature_dim]
        """
        # Create unique set of nodes to update
        unique_nodes = torch.cat([source_nodes, destination_nodes]).unique()
        
        # Create raw messages for each edge
        source_memory = self.get_memory(source_nodes)
        destination_memory = self.get_memory(destination_nodes)
        
        # Time features
        source_time_delta = timestamps - self.last_update[source_nodes]
        source_time_features = self.time_encoder(source_time_delta.unsqueeze(1))
        
        # Create raw messages (concatenation of source memory, destination memory, edge features, time features)
        raw_messages = torch.cat([
            source_memory,
            destination_memory,
            edge_features,
            source_time_features
        ], dim=1)
        
        # Process raw messages
        messages = self.message_function(raw_messages)
        
        # Store messages for both source and destination nodes
        # This simulates the default dict approach in the original repo
        for i, (src, dst) in enumerate(zip(source_nodes, destination_nodes)):
            # Store message for source node (outgoing message)
            self.messages_buffer[src.item()].append(messages[i])
            
            # Store message for destination node (incoming message)
            self.messages_buffer[dst.item()].append(messages[i])
        
        # Update memory for all nodes that received a message
        for node in unique_nodes:
            node_idx = node.item()
            
            # If node has messages, aggregate them and update memory
            if node_idx in self.messages_buffer and len(self.messages_buffer[node_idx]) > 0:
                # Convert message list to tensor
                node_messages = torch.stack(self.messages_buffer[node_idx])
                
                # Aggregate messages
                aggregated_message = self.aggregator(node_messages)
                
                # Update memory
                current_memory = self.memory[node].unsqueeze(0)
                updated_memory = self.memory_updater(aggregated_message.unsqueeze(0), current_memory)
                
                # Update memory and last update time
                self.memory[node] = updated_memory.squeeze(0)
                self.last_update[node] = timestamps[0]  # Using the batch timestamp for simplicity
                
                # Clear the messages buffer for this node
                self.messages_buffer[node_idx] = []
    
    def get_memory(self, node_idxs):
        """
        Get memory for specified nodes
        
        Args:
            node_idxs: Node indices to get memory for [batch_size]
            
        Returns:
            node_memory: Memory of specified nodes [batch_size, memory_dim]
        """
        return self.memory[node_idxs]

# --------------------------------
# Temporal Attention Layer
# --------------------------------
class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention layer for Temporal Graph Networks.
    Uses multi-head attention to weight messages based on temporal information.
    """
    def __init__(self, node_feat_dim, edge_feat_dim, time_feat_dim, memory_dim, 
                 output_dim, n_heads=2, dropout=0.1):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        
        # Dimension calculations
        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim
        self.value_dim = self.key_dim
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim,
            vdim=self.value_dim,
            num_heads=n_heads,
            dropout=dropout
        )
        
        # Output projection layer
        self.output_layer = nn.Linear(self.query_dim, output_dim)
        
    def forward(self, node_features, node_time_features, neighbor_features, 
                neighbor_time_features, edge_features, attention_mask=None):
        """
        Forward pass of temporal attention layer
        
        Args:
            node_features: Features of the target node [batch_size, node_feat_dim]
            node_time_features: Time features of the target node [batch_size, time_feat_dim]
            neighbor_features: Features of neighbor nodes [batch_size, n_neighbors, node_feat_dim]
            neighbor_time_features: Time features of neighbor connections [batch_size, n_neighbors, time_feat_dim]
            edge_features: Features of edges to neighbors [batch_size, n_neighbors, edge_feat_dim]
            attention_mask: Mask for attention (1 for valid neighbors, 0 for padding) [batch_size, n_neighbors]
            
        Returns:
            output: Updated node features incorporating temporal graph information [batch_size, output_dim]
        """
        batch_size, n_neighbors = neighbor_features.size(0), neighbor_features.size(1)
        
        # Create query from node features and time features (target node)
        query = torch.cat([node_features, node_time_features], dim=1).unsqueeze(1)  # [batch_size, 1, query_dim]
        
        # Create key/value from neighbor features, time features, and edge features
        key = torch.cat([
            neighbor_features.view(batch_size * n_neighbors, -1),  # Flatten neighbor features
            neighbor_time_features.view(batch_size * n_neighbors, -1),  # Flatten neighbor time features
            edge_features.view(batch_size * n_neighbors, -1)  # Flatten edge features
        ], dim=1).view(batch_size, n_neighbors, -1)  # [batch_size, n_neighbors, key_dim]
        
        # Value is same as key for self-attention
        value = key
        
        # Create attention mask if provided
        attn_mask = None
        if attention_mask is not None:
            attn_mask = ~attention_mask.bool()  # Convert to boolean mask (True for invalid positions)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, n_neighbors, -1)  # [batch_size, n_neighbors, n_neighbors]
        
        # Apply attention
        # Transpose to get the expected shape for nn.MultiheadAttention
        query = query.transpose(0, 1)  # [1, batch_size, query_dim]
        key = key.transpose(0, 1)  # [n_neighbors, batch_size, key_dim]
        value = value.transpose(0, 1)  # [n_neighbors, batch_size, value_dim]
        
        output, _ = self.attention(query, key, value, attn_mask=attn_mask)
        
        # Reshape output and pass through output layer
        output = output.transpose(0, 1).squeeze(1)  # [batch_size, query_dim]
        output = self.output_layer(output)  # [batch_size, output_dim]
        
        return output

# --------------------------------
# Temporal Graph Network (TGN)
# --------------------------------
class TemporalGraphNetwork(nn.Module):
    """
    A complete implementation of the Temporal Graph Network.
    Follows the architecture described in the Twitter Research paper.
    """
    def __init__(self, num_nodes, node_feat_dim, edge_feat_dim, memory_dim, time_dim, 
                 embedding_dim, message_dim, n_layers=2, n_heads=2, dropout=0.1,
                 use_memory=True, message_function='mlp', memory_updater='gru', 
                 aggregator='lstm'):
        """
        Initialize the TGN model
        
        Args:
            num_nodes: Number of nodes in the graph
            node_feat_dim: Dimension of node features
            edge_feat_dim: Dimension of edge features
            memory_dim: Dimension of node memory
            time_dim: Dimension of time encoding
            embedding_dim: Dimension of final node embeddings
            message_dim: Dimension of messages
            n_layers: Number of graph attention layers
            n_heads: Number of attention heads
            dropout: Dropout probability
            use_memory: Whether to use node memory
            message_function: Message function type ('mlp' or 'identity')
            memory_updater: Memory updater type ('gru' or 'rnn')
            aggregator: Message aggregator type ('mean' or 'lstm')
        """
        super(TemporalGraphNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.message_dim = message_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_memory = use_memory
        
        # Raw message dimension (source/destination memory + edge features + time features)
        raw_message_dim = 2 * memory_dim + edge_feat_dim + time_dim
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Initialize components based on specified types
        # 1. Message function
        if message_function == 'mlp':
            self.message_function = MLPMessageFunction(raw_message_dim, message_dim)
        else:  # 'identity'
            self.message_function = IdentityMessageFunction(message_dim)
        
        # 2. Memory updater
        if memory_updater == 'gru':
            self.memory_updater = GRUMemoryUpdater(message_dim, memory_dim)
        else:  # 'rnn'
            self.memory_updater = RNNMemoryUpdater(message_dim, memory_dim)
            
        # 3. Message aggregator
        if aggregator == 'mean':
            self.aggregator = MeanMessageAggregator()
        else:  # 'lstm'
            self.aggregator = LSTMAggregator(message_dim, memory_dim)
        
        # Memory module
        if use_memory:
            self.memory_module = TGNMemoryModule(
                num_nodes=num_nodes,
                memory_dim=memory_dim,
                message_dim=message_dim,
                time_dim=time_dim,
                message_function=self.message_function,
                memory_updater=self.memory_updater,
                aggregator=self.aggregator
            )
        
        # Node embedding layer (project raw features to memory_dim)
        self.node_embedding = nn.Linear(node_feat_dim, memory_dim)
        
        # Edge embedding layer
        self.edge_embedding = nn.Linear(edge_feat_dim, memory_dim)
        
        # Graph attention layers
        self.graph_attention_layers = nn.ModuleList([
            TemporalAttentionLayer(
                node_feat_dim=memory_dim,
                edge_feat_dim=memory_dim,
                time_feat_dim=time_dim,
                memory_dim=memory_dim,
                output_dim=embedding_dim if layer == n_layers - 1 else memory_dim,
                n_heads=n_heads,
                dropout=dropout
            )
            for layer in range(n_layers)
        ])
        
        # For link prediction task
        self.link_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def compute_temporal_embeddings(self, node_ids, node_features, timestamps, 
                                    neighbor_ids=None, neighbor_times=None, 
                                    neighbor_features=None, edge_features=None):
        """
        Compute temporal embeddings for nodes
        
        Args:
            node_ids: Node IDs to compute embeddings for [batch_size]
            node_features: Features of nodes [batch_size, node_feat_dim]
            timestamps: Current timestamps for nodes [batch_size]
            neighbor_ids: IDs of neighbors for each node [batch_size, n_neighbors]
            neighbor_times: Timestamps of connections to neighbors [batch_size, n_neighbors]
            neighbor_features: Features of neighbors [batch_size, n_neighbors, node_feat_dim]
            edge_features: Features of edges to neighbors [batch_size, n_neighbors, edge_feat_dim]
            
        Returns:
            node_embeddings: Computed embeddings [batch_size, embedding_dim]
        """
        batch_size = len(node_ids)
        
        # Get memory for nodes if using memory
        if self.use_memory:
            node_memory = self.memory_module.get_memory(node_ids)
        else:
            # If not using memory, initialize to zeros
            node_memory = torch.zeros(batch_size, self.memory_dim, device=node_features.device)
        
        # Compute time features
        time_features = self.time_encoder(timestamps.view(-1, 1))  # [batch_size, time_dim]
        
        # Initialize embeddings with memory and node features
        node_embeddings = self.node_embedding(node_features)  # [batch_size, memory_dim]
        
        # If no neighbors, return only node embeddings
        if neighbor_ids is None or len(neighbor_ids) == 0:
            return node_embeddings
        
        # Process neighbor features
        if neighbor_features is not None:
            neighbor_embeddings = self.node_embedding(neighbor_features)  # [batch_size, n_neighbors, memory_dim]
        else:
            # If no neighbor features, use memory or zeros
            n_neighbors = neighbor_ids.size(1)
            neighbor_embeddings = torch.zeros(batch_size, n_neighbors, self.memory_dim, device=node_features.device)
            
            # Get memory for neighbors if using memory
            if self.use_memory:
                for i, neighbors in enumerate(neighbor_ids):
                    neighbor_embeddings[i] = self.memory_module.get_memory(neighbors)
        
        # Process edge features
        if edge_features is not None:
            edge_embeddings = self.edge_embedding(edge_features)  # [batch_size, n_neighbors, memory_dim]
        else:
            # If no edge features, use zeros
            n_neighbors = neighbor_ids.size(1)
            edge_embeddings = torch.zeros(batch_size, n_neighbors, self.memory_dim, device=node_features.device)
        
        # Compute neighbor time features
        if neighbor_times is not None:
            # Compute time difference features for each neighbor
            neighbor_time_features = self.time_encoder(neighbor_times.view(-1, 1)).view(batch_size, -1, self.time_dim)
        else:
            # If no neighbor times, use zeros
            n_neighbors = neighbor_ids.size(1)
            neighbor_time_features = torch.zeros(batch_size, n_neighbors, self.time_dim, device=node_features.device)
        
        # Create attention mask for valid neighbors
        attention_mask = (neighbor_ids != -1) if neighbor_ids is not None else None
        
        # Apply graph attention layers
        x = node_embeddings
        for layer in self.graph_attention_layers:
            x = layer(
                node_features=x,
                node_time_features=time_features,
                neighbor_features=neighbor_embeddings,
                neighbor_time_features=neighbor_time_features,
                edge_features=edge_embeddings,
                attention_mask=attention_mask
            )
        
        return x
    
    def forward(self, src_ids, dst_ids, src_features, dst_features, timestamps, 
                edge_features, src_neighbor_ids=None, dst_neighbor_ids=None, 
                src_neighbor_times=None, dst_neighbor_times=None,
                src_neighbor_features=None, dst_neighbor_features=None, 
                src_edge_features=None, dst_edge_features=None):
        """
        Forward pass for link prediction task
        
        Args:
            src_ids: Source node IDs [batch_size]
            dst_ids: Destination node IDs [batch_size]
            src_features: Source node features [batch_size, node_feat_dim]
            dst_features: Destination node features [batch_size, node_feat_dim]
            timestamps: Edge timestamps [batch_size]
            edge_features: Edge features [batch_size, edge_feat_dim]
            src_neighbor_ids: Neighbor IDs of source nodes [batch_size, n_neighbors]
            dst_neighbor_ids: Neighbor IDs of destination nodes [batch_size, n_neighbors]
            src_neighbor_times: Timestamps of source node neighbors [batch_size, n_neighbors]
            dst_neighbor_times: Timestamps of destination node neighbors [batch_size, n_neighbors]
            src_neighbor_features: Features of source node neighbors [batch_size, n_neighbors, node_feat_dim]
            dst_neighbor_features: Features of destination node neighbors [batch_size, n_neighbors, node_feat_dim]
            src_edge_features: Edge features of source node connections [batch_size, n_neighbors, edge_feat_dim]
            dst_edge_features: Edge features of destination node connections [batch_size, n_neighbors, edge_feat_dim]
            
        Returns:
            link_prob: Probability of link existence [batch_size, 1]
        """
        # Update memory if using memory
        if self.use_memory:
            self.memory_module.update_state(src_ids, dst_ids, timestamps, edge_features)
        
        # Compute embeddings for source nodes
        src_embeddings = self.compute_temporal_embeddings(
            node_ids=src_ids,
            node_features=src_features,
            timestamps=timestamps,
            neighbor_ids=src_neighbor_ids,
            neighbor_times=src_neighbor_times,
            neighbor_features=src_neighbor_features,
            edge_features=src_edge_features
        )
        
        # Compute embeddings for destination nodes
        dst_embeddings = self.compute_temporal_embeddings(
            node_ids=dst_ids,
            node_features=dst_features,
            timestamps=timestamps,
            neighbor_ids=dst_neighbor_ids,
            neighbor_times=dst_neighbor_times,
            neighbor_features=dst_neighbor_features,
            edge_features=dst_edge_features
        )
        
        # Predict link probability
        link_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
        link_prob = self.link_predictor(link_features)
        
        return link_prob, src_embeddings, dst_embeddings
    
    def reset_memory(self):
        """Reset the memory at the beginning of each epoch"""
        if self.use_memory:
            self.memory_module.reset_state()
    
    def detach_memory(self):
        """Detach memory from computation graph to prevent memory leaks"""
        if self.use_memory:
            self.memory_module.detach()

# --------------------------------
# Temporal Attention with Decay
# --------------------------------
class DecayTemporalAttention(nn.Module):
    """
    Temporal attention mechanism with exponential decay
    The decay factor has been set to 0.0 to evaluate the model without decay metrics
    """
    def __init__(self, time_dim, method="sum", decay_factor=0.1):
        super(DecayTemporalAttention, self).__init__()
        self.time_dim = time_dim
        self.method = method
        # self.decay_factor = decay_factor
        self.decay_factor = 0.0  # Setting to zero to disable decay effect
        
        # Weight vector for computing temporal attention scores
        self.w = nn.Parameter(torch.ones(time_dim))
        
    def forward(self, memory, message_time, current_time):
        """
        Apply temporal attention with decay
        
        Args:
            memory: Memory state batch [batch_size, memory_dim]
            message_time: Timestamp of each message [batch_size]
            current_time: Current timestamp [batch_size]
            
        Returns:
            weighted_memory: Memory weighted by attention scores [batch_size, memory_dim]
        """
        # Convert times to tensors
        if not isinstance(message_time, torch.Tensor):
            message_time = torch.tensor(message_time, device=memory.device)
        if not isinstance(current_time, torch.Tensor):
            current_time = torch.tensor(current_time, device=memory.device)
            
        # Compute time difference
        delta_t = (current_time - message_time).float()
        
        # Apply exponential decay based on time difference
        # Since decay_factor = 0.0, this will not affect the embeddings
        score = torch.exp(-self.decay_factor * delta_t)
        
        # Apply attention weights
        if self.method == "sum":
            weighted_memory = memory * score.unsqueeze(1)
        else:  # product
            weighted_memory = memory * score.unsqueeze(1)
            
        return weighted_memory
