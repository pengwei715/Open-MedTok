import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """
    Text encoder module for MEDTOK.
    
    This module encodes text descriptions of medical codes using a pretrained
    language model followed by a projection layer.
    """
    
    def __init__(self, config):
        """
        Initialize the text encoder.
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__()
        self.config = config
        
        # Initialize the transformer model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_model)
        self.transformer = AutoModel.from_pretrained(config.text_encoder_model)
        
        # Freeze the transformer if not trainable
        if not config.text_encoder_trainable:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Projection layer to map text embeddings to the desired dimension
        self.projection = nn.Linear(self.transformer.config.hidden_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Layer normalization for stabilizing training
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
    
    def forward(self, texts, attention_mask=None):
        """
        Encode text descriptions into embeddings.
        
        Args:
            texts: Either a list of text strings or already tokenized inputs
            attention_mask: Optional attention mask if texts are already tokenized
            
        Returns:
            Text embeddings of shape [batch_size, embedding_dim]
        """
        # Tokenize the texts if raw strings are provided
        if isinstance(texts[0], str):
            encoded_inputs = self.tokenizer(
                texts, 
                padding="max_length", 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.config.device)
            
            input_ids = encoded_inputs["input_ids"]
            attention_mask = encoded_inputs["attention_mask"]
        else:
            # Assume texts are already tokenized
            input_ids = texts
        
        # Get transformer outputs
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token embedding as the text representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Project to the desired dimension
        text_embedding = self.projection(cls_embedding)
        text_embedding = self.dropout(text_embedding)
        text_embedding = self.layer_norm(text_embedding)
        
        return text_embedding


class WeightedPoolingTextEncoder(TextEncoder):
    """
    Text encoder with weighted pooling for improved medical concept representation.
    
    This extends the basic TextEncoder by using a weighted pooling mechanism
    over token representations instead of just using the [CLS] token.
    """
    
    def __init__(self, config):
        """Initialize the weighted pooling text encoder."""
        super().__init__(config)
        
        # Attention mechanism for weighted pooling
        self.attention = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, texts, attention_mask=None):
        """
        Encode text descriptions using weighted pooling.
        
        Args:
            texts: Either a list of text strings or already tokenized inputs
            attention_mask: Optional attention mask if texts are already tokenized
            
        Returns:
            Text embeddings of shape [batch_size, embedding_dim]
        """
        # Tokenize the texts if raw strings are provided
        if isinstance(texts[0], str):
            encoded_inputs = self.tokenizer(
                texts, 
                padding="max_length", 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.config.device)
            
            input_ids = encoded_inputs["input_ids"]
            attention_mask = encoded_inputs["attention_mask"]
        else:
            # Assume texts are already tokenized
            input_ids = texts
        
        # Get transformer outputs
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Calculate attention weights
        attention_weights = self.attention(hidden_states).squeeze(-1)
        
        # Apply mask to ignore padding tokens
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask.eq(0), -1e9)
        
        # Apply softmax to get normalized weights
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply weighted pooling
        weighted_sum = torch.bmm(
            attention_weights.unsqueeze(1), 
            hidden_states
        ).squeeze(1)
        
        # Project to the desired dimension
        text_embedding = self.projection(weighted_sum)
        text_embedding = self.dropout(text_embedding)
        text_embedding = self.layer_norm(text_embedding)
        
        return text_embedding
