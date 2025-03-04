import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for MEDTOK.
    
    This module quantizes input embeddings into discrete tokens by querying a learnable
    codebook and selecting the nearest vectors. It uses the straight-through estimator
    for backpropagation as described in van den Oord et al. (2017) and the MEDTOK paper.
    """
    
    def __init__(self, config, codebook_start_idx=0, codebook_end_idx=None):
        """
        Initialize the Vector Quantizer.
        
        Args:
            config: Configuration object containing model parameters
            codebook_start_idx: Starting index in the global codebook to use
            codebook_end_idx: Ending index in the global codebook to use
        """
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_top_k = config.num_top_k_tokens
        self.alpha = config.alpha
        
        # Define the range of codebook to use
        self.start_idx = codebook_start_idx
        self.end_idx = codebook_end_idx if codebook_end_idx is not None else config.codebook_size
        self.num_embeddings = self.end_idx - self.start_idx
        
        # Initialize the codebook
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
    
    def get_codebook(self):
        """Return the current codebook."""
        return self.codebook.weight
    
    def compute_distances(self, z):
        """
        Compute distances between input vectors and codebook entries.
        
        Args:
            z: Input vectors of shape [batch_size, embedding_dim]
            
        Returns:
            Distances of shape [batch_size, num_embeddings]
        """
        # Calculate squared L2 distances
        z_flattened = z.view(-1, self.embedding_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.codebook.weight.t())
        return d
    
    def get_token_indices(self, distances):
        """
        Get the indices of the closest tokens in the codebook.
        
        Args:
            distances: Distance matrix of shape [batch_size, num_embeddings]
            
        Returns:
            Indices of the top-k closest tokens for each input
        """
        _, indices = torch.topk(distances, k=self.num_top_k, dim=1, largest=False)
        return indices
    
    def forward(self, z):
        """
        Forward pass through the vector quantizer.
        
        Args:
            z: Input vectors of shape [batch_size, embedding_dim]
            
        Returns:
            quantized: Quantized vectors
            token_indices: Indices of the selected tokens
            quantization_loss: Loss for updating the codebook
        """
        # Calculate distances
        distances = self.compute_distances(z)
        
        # Get top-k token indices
        token_indices = self.get_token_indices(distances)
        
        # Compute weights based on distances
        weights = F.softmax(-distances.gather(1, token_indices), dim=1)
        
        # Get quantized vectors by weighted sum of codebook entries
        z_q = torch.zeros_like(z)
        for i in range(self.num_top_k):
            selected_indices = token_indices[:, i]
            selected_embeddings = self.codebook(selected_indices)
            z_q += weights[:, i:i+1] * selected_embeddings
        
        # Compute the VQ Objective (codebook loss)
        q_loss = F.mse_loss(z_q.detach(), z) + self.alpha * F.mse_loss(z_q, z.detach())
        
        # Apply straight-through estimator
        z_q = z + (z_q - z).detach()
        
        # Return quantities needed for training and inference
        return z_q, token_indices, q_loss


class MultiPartCodebook(nn.Module):
    """
    Implementation of a multi-part codebook for MEDTOK.
    
    This module divides the codebook into three regions:
    - Text-specific region
    - Graph-specific region
    - Shared region
    """
    
    def __init__(self, config):
        """
        Initialize the multi-part codebook.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.config = config
        self.device = config.device
        
        # Create the global codebook
        self.codebook = nn.Embedding(config.codebook_size, config.embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / config.codebook_size, 1.0 / config.codebook_size)
        
        # Define the regions
        self.text_specific_start_idx = 0
        self.text_specific_end_idx = config.text_specific_size
        
        self.graph_specific_start_idx = config.text_specific_size
        self.graph_specific_end_idx = config.text_specific_size + config.graph_specific_size
        
        self.shared_start_idx = config.text_specific_size + config.graph_specific_size
        self.shared_end_idx = config.codebook_size
        
        # Create separate quantizers for each region
        self.text_specific_quantizer = VectorQuantizer(
            config, 
            codebook_start_idx=self.text_specific_start_idx,
            codebook_end_idx=self.text_specific_end_idx
        )
        
        self.graph_specific_quantizer = VectorQuantizer(
            config, 
            codebook_start_idx=self.graph_specific_start_idx,
            codebook_end_idx=self.graph_specific_end_idx
        )
        
        self.shared_quantizer = VectorQuantizer(
            config, 
            codebook_start_idx=self.shared_start_idx,
            codebook_end_idx=self.shared_end_idx
        )
    
    def quantize_text_specific(self, z):
        """Quantize using the text-specific region of the codebook."""
        return self.text_specific_quantizer(z)
    
    def quantize_graph_specific(self, z):
        """Quantize using the graph-specific region of the codebook."""
        return self.graph_specific_quantizer(z)
    
    def quantize_shared(self, z):
        """Quantize using the shared region of the codebook."""
        return self.shared_quantizer(z)
