import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenPacker(nn.Module):
    """
    Token Packer module for MEDTOK.
    
    This module optimizes tokens by preserving both shared and modality-specific
    information across text and graph modalities as described in the paper.
    """
    
    def __init__(self, config):
        """
        Initialize the token packer.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        self.beta = config.beta
        self.lambda_val = config.lambda_val
        self.device = config.device
        
        # Cross-attention mechanism for shared information
        self.cross_attention_text = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        self.cross_attention_graph = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Projection layers for cross-modality and specific embeddings
        self.text_cross_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.graph_cross_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        self.text_specific_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.graph_specific_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def compute_cross_attention(self, q, k, v):
        """
        Compute cross-attention between modalities.
        
        Args:
            q: Query embeddings
            k: Key embeddings
            v: Value embeddings
            
        Returns:
            Context vectors after applying attention
        """
        # Reshape for attention if needed
        if q.dim() == 2:
            q = q.unsqueeze(1)
        if k.dim() == 2:
            k = k.unsqueeze(1)
        if v.dim() == 2:
            v = v.unsqueeze(1)
        
        # Apply cross-attention
        attn_output, _ = F.multi_head_attention_forward(
            query=q,
            key=k,
            value=v,
            embed_dim_to_check=q.size(-1),
            num_heads=4,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.config.dropout_rate,
            out_proj_weight=torch.eye(q.size(-1), device=q.device),
            out_proj_bias=torch.zeros(q.size(-1), device=q.device),
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=True,
            q_proj_weight=torch.eye(q.size(-1), device=q.device),
            k_proj_weight=torch.eye(k.size(-1), device=q.device),
            v_proj_weight=torch.eye(v.size(-1), device=q.device)
        )
        
        return attn_output.squeeze(1)
    
    def compute_infonce_loss(self, z1, z2):
        """
        Compute InfoNCE loss for self-supervised learning.
        
        Args:
            z1, z2: Batch of embeddings to compare
            
        Returns:
            InfoNCE loss value
        """
        batch_size = z1.size(0)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        logits = torch.matmul(z1, z2.t()) / 0.07  # Temperature scaling
        labels = torch.arange(batch_size, device=z1.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def compute_orthogonal_loss(self, z1, z2):
        """
        Compute orthogonality loss between embeddings.
        
        Args:
            z1, z2: Embeddings that should be orthogonal
            
        Returns:
            Orthogonality loss value
        """
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        correlation = torch.matmul(z1, z2.t())
        loss = torch.mean(correlation ** 2)
        return loss
    
    def forward(self, text_embedding, graph_embedding, e_text_s, e_graph_s, e_text_c, e_graph_c):
        """
        Optimize the representation of tokens to preserve modality shared and specific information.
        
        Args:
            text_embedding: Original text embeddings
            graph_embedding: Original graph embeddings
            e_text_s: Text-specific embeddings
            e_graph_s: Graph-specific embeddings
            e_text_c: Text cross-modality embeddings
            e_graph_c: Graph cross-modality embeddings
            
        Returns:
            Optimized embeddings and optimization losses
        """
        batch_size = text_embedding.size(0)
        
        # Generate augmented data for contrastive learning
        text_augmented = text_embedding + 0.1 * torch.randn_like(text_embedding)
        graph_augmented = graph_embedding + 0.1 * torch.randn_like(graph_embedding)
        
        # Optimize shared information with KL divergence
        text_attn_logits = -F.pairwise_distance(e_text_c.unsqueeze(1), 
                                               self.config.codebook.repeat(batch_size, 1, 1))
        graph_attn_logits = -F.pairwise_distance(e_graph_c.unsqueeze(1), 
                                                self.config.codebook.repeat(batch_size, 1, 1))
        
        text_dist = F.softmax(text_attn_logits, dim=1)
        graph_dist = F.softmax(graph_attn_logits, dim=1)
        
        kl_loss = F.kl_div(text_dist.log(), graph_dist, reduction='batchmean')
        
        # InfoNCE loss for cross-modal representations
        infonce_text = self.compute_infonce_loss(e_text_c, e_graph_c)
        infonce_graph = self.compute_infonce_loss(e_graph_c, e_text_c)
        
        # Orthogonal loss for modality-specific embeddings
        ortho_text = self.compute_orthogonal_loss(e_text_s, e_text_c)
        ortho_graph = self.compute_orthogonal_loss(e_graph_s, e_graph_c)
        
        # InfoNCE loss with augmented data
        infonce_text_aug = self.compute_infonce_loss(e_text_c, text_augmented)
        infonce_graph_aug = self.compute_infonce_loss(e_graph_c, graph_augmented)
        
        # Combined losses
        common_loss = infonce_text + infonce_graph - (2 * self.beta * torch.mean(e_text_c * e_graph_c))
        specific_loss = (infonce_text_aug + self.lambda_val * ortho_text + 
                        infonce_graph_aug + self.lambda_val * ortho_graph)
        
        # Total token packing loss
        token_packing_loss = common_loss + specific_loss + kl_loss
        
        return {
            'common_loss': common_loss,
            'specific_loss': specific_loss,
            'kl_loss': kl_loss,
            'token_packing_loss': token_packing_loss
        }
