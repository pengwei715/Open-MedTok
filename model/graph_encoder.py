import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool


class GraphEncoder(nn.Module):
    """
    Graph encoder module for MEDTOK.
    
    This module encodes graph-based representations of medical codes
    using Graph Neural Networks.
    """
    
    def __init__(self, config):
        """
        Initialize the graph encoder.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        
        # Define dimensions
        self.node_feature_dim = config.node_feature_dim
        self.embedding_dim = config.embedding_dim
        
        # GNN layers (GCN as the default choice)
        self.conv1 = GCNConv(self.node_feature_dim, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.embedding_dim)
        )
        
        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, x, edge_index, batch=None):
        """
        Encode graph-based representations into embeddings.
        
        Args:
            x: Node features of shape [num_nodes, node_feature_dim]
            edge_index: Graph connectivity in COO format
            batch: Batch assignment for nodes in disjoint graphs
            
        Returns:
            Graph embeddings of shape [batch_size, embedding_dim]
        """
        # Apply GNN layers with residual connections
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index)) + h1
        h3 = F.relu(self.conv3(h2, edge_index)) + h2
        
        # Apply global pooling to get graph-level representations
        if batch is None:
            batch = torch.zeros(h3.size(0), dtype=torch.long, device=h3.device)
        
        graph_embedding = global_mean_pool(h3, batch)
        
        # Project to the desired dimension
        graph_embedding = self.projection(graph_embedding)
        graph_embedding = self.layer_norm(graph_embedding)
        
        return graph_embedding


class GATGraphEncoder(GraphEncoder):
    """
    Graph Attention Network encoder for MEDTOK.
    
    This variant uses Graph Attention Networks instead of GCNs
    for potentially better performance on complex graph structures.
    """
    
    def __init__(self, config):
        """Initialize the GAT graph encoder."""
        super().__init__(config)
        
        # Replace GCN layers with GAT layers
        self.conv1 = GATConv(self.node_feature_dim, 256, heads=4, concat=False, dropout=config.dropout_rate)
        self.conv2 = GATConv(256, 256, heads=4, concat=False, dropout=config.dropout_rate)
        self.conv3 = GATConv(256, 256, heads=4, concat=False, dropout=config.dropout_rate)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the GAT encoder."""
        # Apply GAT layers with residual connections
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index)) + h1
        h3 = F.relu(self.conv3(h2, edge_index)) + h2
        
        # Apply global pooling to get graph-level representations
        if batch is None:
            batch = torch.zeros(h3.size(0), dtype=torch.long, device=h3.device)
        
        graph_embedding = global_mean_pool(h3, batch)
        
        # Project to the desired dimension
        graph_embedding = self.projection(graph_embedding)
        graph_embedding = self.layer_norm(graph_embedding)
        
        return graph_embedding


class HierarchicalGraphEncoder(nn.Module):
    """
    Hierarchical Graph Encoder for MEDTOK.
    
    This encoder captures hierarchical relationships in medical ontologies
    by using multiple levels of graph processing.
    """
    
    def __init__(self, config):
        """Initialize the hierarchical graph encoder."""
        super().__init__()
        self.config = config
        
        # Node-level encoding
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 256)
        )
        
        # Local structure encoding with GAT
        self.local_conv = GATConv(256, 256, heads=4, concat=False, dropout=config.dropout_rate)
        
        # Global structure encoding with GCN
        self.global_conv1 = GCNConv(256, 256)
        self.global_conv2 = GCNConv(256, 256)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.embedding_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through the hierarchical graph encoder."""
        # Node-level encoding
        x = self.node_encoder(x)
        
        # Local structure encoding
        x_local = F.relu(self.local_conv(x, edge_index))
        
        # Global structure encoding
        x_global = F.relu(self.global_conv1(x, edge_index))
        x_global = F.relu(self.global_conv2(x_global, edge_index))
        
        # Combine local and global information
        x_combined = x_local + x_global
        
        # Apply global pooling to get graph-level representations
        if batch is None:
            batch = torch.zeros(x_combined.size(0), dtype=torch.long, device=x_combined.device)
        
        graph_embedding = global_mean_pool(x_combined, batch)
        
        # Project to the desired dimension
        graph_embedding = self.projection(graph_embedding)
        graph_embedding = self.layer_norm(graph_embedding)
        
        return graph_embedding
