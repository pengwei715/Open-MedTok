import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder, WeightedPoolingTextEncoder
from .graph_encoder import GraphEncoder, GATGraphEncoder, HierarchicalGraphEncoder
from .vector_quantizer import MultiPartCodebook
from .token_packer import TokenPacker


class MedTok(nn.Module):
    """
    MEDTOK: Multimodal Medical Code Tokenizer.
    
    This model tokenizes medical codes by integrating their textual descriptions
    and graph-based representations from biomedical knowledge graphs. It uses
    vector quantization to map both modalities into a unified token space.
    """
    
    def __init__(self, config):
        """
        Initialize the MEDTOK model.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        
        # Initialize encoders
        if config.text_encoder_type == "weighted_pooling":
            self.text_encoder = WeightedPoolingTextEncoder(config)
        else:
            self.text_encoder = TextEncoder(config)
        
        if config.graph_encoder_type == "gat":
            self.graph_encoder = GATGraphEncoder(config)
        elif config.graph_encoder_type == "hierarchical":
            self.graph_encoder = HierarchicalGraphEncoder(config)
        else:
            self.graph_encoder = GraphEncoder(config)
        
        # Linear projectors for modality-specific embeddings
        self.text_projector = nn.Linear(config.text_encoder_dim, config.embedding_dim)
        self.graph_projector = nn.Linear(config.graph_encoder_dim, config.embedding_dim)
        
        # Cross-attention modules for cross-modality embeddings
        self.text_cross_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        self.graph_cross_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Initialize the multi-part codebook
        self.codebook = MultiPartCodebook(config)
        
        # Initialize the token packer
        self.token_packer = TokenPacker(config)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def encode_text(self, text):
        """
        Encode text descriptions into embeddings.
        
        Args:
            text: Text descriptions of medical codes
            
        Returns:
            Text embeddings
        """
        text_embedding = self.text_encoder(text)
        return text_embedding
    
    def encode_graph(self, x, edge_index, batch=None):
        """
        Encode graph representations into embeddings.
        
        Args:
            x: Node features
            edge_index: Graph connectivity in COO format
            batch: Batch assignment for nodes in disjoint graphs
            
        Returns:
            Graph embeddings
        """
        graph_embedding = self.graph_encoder(x, edge_index, batch)
        return graph_embedding
    
    def derive_modality_embeddings(self, text_embedding, graph_embedding):
        """
        Derive modality-specific and cross-modality embeddings.
        
        Args:
            text_embedding: Text embeddings from the text encoder
            graph_embedding: Graph embeddings from the graph encoder
            
        Returns:
            Tuple of modality-specific and cross-modality embeddings
        """
        # Project embeddings to the common dimension
        text_proj = self.text_projector(text_embedding)
        graph_proj = self.graph_projector(graph_embedding)
        
        # Normalize projections
        text_proj = F.normalize(text_proj, p=2, dim=1)
        graph_proj = F.normalize(graph_proj, p=2, dim=1)
        
        # Derive modality-specific embeddings
        e_text_s = text_proj
        e_graph_s = graph_proj
        
        # Derive cross-modality embeddings using cross-attention
        # Reshape for attention
        text_proj_reshaped = text_proj.unsqueeze(1)
        graph_proj_reshaped = graph_proj.unsqueeze(1)
        
        # Text-to-graph attention
        e_text_c, _ = self.text_cross_attention(
            text_proj_reshaped,
            graph_proj_reshaped,
            graph_proj_reshaped
        )
        e_text_c = e_text_c.squeeze(1)
        
        # Graph-to-text attention
        e_graph_c, _ = self.graph_cross_attention(
            graph_proj_reshaped,
            text_proj_reshaped,
            text_proj_reshaped
        )
        e_graph_c = e_graph_c.squeeze(1)
        
        # Apply layer normalization
        e_text_s = self.layer_norm(e_text_s)
        e_graph_s = self.layer_norm(e_graph_s)
        e_text_c = self.layer_norm(e_text_c)
        e_graph_c = self.layer_norm(e_graph_c)
        
        return e_text_s, e_graph_s, e_text_c, e_graph_c
    
    def quantize_embeddings(self, e_text_s, e_graph_s, e_text_c, e_graph_c):
        """
        Quantize the embeddings into tokens.
        
        Args:
            e_text_s: Text-specific embeddings
            e_graph_s: Graph-specific embeddings
            e_text_c: Text cross-modality embeddings
            e_graph_c: Graph cross-modality embeddings
            
        Returns:
            Quantized embeddings, token indices, and quantization loss
        """
        # Quantize text-specific embeddings
        z_text_s, indices_text_s, q_loss_text_s = self.codebook.quantize_text_specific(e_text_s)
        
        # Quantize graph-specific embeddings
        z_graph_s, indices_graph_s, q_loss_graph_s = self.codebook.quantize_graph_specific(e_graph_s)
        
        # Quantize cross-modality embeddings
        z_text_c, indices_text_c, q_loss_text_c = self.codebook.quantize_shared(e_text_c)
        z_graph_c, indices_graph_c, q_loss_graph_c = self.codebook.quantize_shared(e_graph_c)
        
        # Combine quantization losses
        q_loss = q_loss_text_s + q_loss_graph_s + q_loss_text_c + q_loss_graph_c
        
        # Return quantized embeddings and token indices
        return {
            'z_text_s': z_text_s,
            'z_graph_s': z_graph_s,
            'z_text_c': z_text_c,
            'z_graph_c': z_graph_c,
            'indices_text_s': indices_text_s,
            'indices_graph_s': indices_graph_s,
            'indices_text_c': indices_text_c,
            'indices_graph_c': indices_graph_c,
            'q_loss': q_loss
        }
    
    def forward(self, text, graph_features, graph_edge_index, graph_batch=None):
        """
        Forward pass through the MEDTOK tokenizer.
        
        Args:
            text: Text descriptions of medical codes
            graph_features: Node features for graphs
            graph_edge_index: Graph connectivity in COO format
            graph_batch: Batch assignment for nodes in disjoint graphs
            
        Returns:
            Dictionary containing token indices, embeddings, and losses
        """
        # Encode modalities
        text_embedding = self.encode_text(text)
        graph_embedding = self.encode_graph(graph_features, graph_edge_index, graph_batch)
        
        # Derive modality-specific and cross-modality embeddings
        e_text_s, e_graph_s, e_text_c, e_graph_c = self.derive_modality_embeddings(
            text_embedding, graph_embedding
        )
        
        # Quantize embeddings into tokens
        quantization_results = self.quantize_embeddings(e_text_s, e_graph_s, e_text_c, e_graph_c)
        
        # Apply token packing
        packing_results = self.token_packer(
            text_embedding, 
            graph_embedding, 
            e_text_s,
            e_graph_s, 
            e_text_c, 
            e_graph_c
        )
        
        # Combine all token indices
        token_indices = torch.cat([
            quantization_results['indices_text_s'],
            quantization_results['indices_graph_s'],
            quantization_results['indices_text_c'],
            quantization_results['indices_graph_c']
        ], dim=1)
        
        # Compute the total loss
        total_loss = quantization_results['q_loss'] + packing_results['token_packing_loss']
        
        return {
            'token_indices': token_indices,
            'z_text_s': quantization_results['z_text_s'],
            'z_graph_s': quantization_results['z_graph_s'],
            'z_text_c': quantization_results['z_text_c'],
            'z_graph_c': quantization_results['z_graph_c'],
            'e_text_s': e_text_s,
            'e_graph_s': e_graph_s,
            'e_text_c': e_text_c,
            'e_graph_c': e_graph_c,
            'quantization_loss': quantization_results['q_loss'],
            'packing_loss': packing_results['token_packing_loss'],
            'total_loss': total_loss
        }
    
    def tokenize(self, text, graph_features, graph_edge_index, graph_batch=None):
        """
        Tokenize medical codes into discrete tokens.
        
        Args:
            text: Text descriptions of medical codes
            graph_features: Node features for graphs
            graph_edge_index: Graph connectivity in COO format
            graph_batch: Batch assignment for nodes in disjoint graphs
            
        Returns:
            Token indices for the medical codes
        """
        with torch.no_grad():
            results = self.forward(text, graph_features, graph_edge_index, graph_batch)
            return results['token_indices']
    
    def get_token_embedding(self, token_indices):
        """
        Get token embeddings for provided token indices.
        
        Args:
            token_indices: Indices of tokens in the codebook
            
        Returns:
            Token embeddings
        """
        # Convert relative indices to global indices based on codebook regions
        text_specific_indices = token_indices[:, :self.config.num_top_k_tokens]
        graph_specific_indices = token_indices[:, self.config.num_top_k_tokens:2*self.config.num_top_k_tokens]
        text_shared_indices = token_indices[:, 2*self.config.num_top_k_tokens:3*self.config.num_top_k_tokens]
        graph_shared_indices = token_indices[:, 3*self.config.num_top_k_tokens:]
        
        # Get embeddings for each token type
        embeddings = []
        
        for i in range(text_specific_indices.size(1)):
            embeddings.append(self.codebook.text_specific_quantizer.codebook(text_specific_indices[:, i]))
        
        for i in range(graph_specific_indices.size(1)):
            embeddings.append(self.codebook.graph_specific_quantizer.codebook(graph_specific_indices[:, i]))
        
        for i in range(text_shared_indices.size(1)):
            embeddings.append(self.codebook.shared_quantizer.codebook(text_shared_indices[:, i]))
        
        for i in range(graph_shared_indices.size(1)):
            embeddings.append(self.codebook.shared_quantizer.codebook(graph_shared_indices[:, i]))
        
        # Concatenate all embeddings
        token_embedding = torch.stack(embeddings, dim=1)
        
        return token_embedding

