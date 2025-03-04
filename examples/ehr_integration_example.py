#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to integrate MEDTOK with an EHR model.

This example shows how to use MEDTOK tokens with a transformer-based EHR model.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.medtok import MedTok
from utils.config import MedTokConfig


class SimpleEHRTransformer(nn.Module):
    """
    Simple transformer-based EHR model.
    
    This is a demonstration model that uses MEDTOK token embeddings
    as input to a transformer encoder for patient outcome prediction.
    """
    
    def __init__(self, token_embedding_dim, hidden_dim=256, num_heads=4, 
                 num_layers=2, dropout=0.1, max_seq_len=128, num_classes=2):
        """
        Initialize the EHR transformer model.
        
        Args:
            token_embedding_dim: Dimension of token embeddings
            hidden_dim: Hidden dimension of the transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.token_embedding_dim = token_embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Projection layer for token embeddings
        self.projection = nn.Linear(token_embedding_dim, hidden_dim)
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, token_embeddings, attention_mask=None):
        """
        Forward pass through the model.
        
        Args:
            token_embeddings: Token embeddings [batch_size, seq_len, token_embedding_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Classification logits
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Project token embeddings to hidden dimension
        x = self.projection(token_embeddings)
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Create attention mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=token_embeddings.device)
        
        # Convert attention mask format for transformer
        attention_mask = attention_mask.bool()
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        
        # Take the representation of the first token (CLS token)
        x = x[:, 0]
        
        # Apply classification head
        logits = self.classifier(x)
        
        return logits


class EHRDataset(Dataset):
    """
    Dataset for EHR patient data.
    
    This dataset loads patient visit sequences with medical codes
    and corresponding outcomes.
    """
    
    def __init__(self, data_file, medtok_model, max_seq_len=128):
        """
        Initialize the EHR dataset.
        
        Args:
            data_file: Path to the data file
            medtok_model: MEDTOK model for tokenization
            max_seq_len: Maximum sequence length
        """
        self.max_seq_len = max_seq_len
        self.medtok_model = medtok_model
        
        # Load data
        self.data = pd.read_csv(data_file)
        
        # Get unique patient IDs
        self.patient_ids = self.data['patient_id'].unique()
        
        # Group by patient ID
        self.patient_groups = self.data.groupby('patient_id')
    
    def __len__(self):
        """Return number of patients."""
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        """
        Get patient data.
        
        Args:
            idx: Index of the patient
            
        Returns:
            Tokenized patient data and outcome label
        """
        patient_id = self.patient_ids[idx]
        patient_data = self.patient_groups.get_group(patient_id)
        
        # Sort by visit date
        patient_data = patient_data.sort_values('visit_date')
        
        # Get medical codes and outcomes
        codes = patient_data['medical_code'].tolist()
        code_descs = patient_data['code_description'].tolist()
        outcome = patient_data['outcome'].iloc[-1]  # Use outcome of the last visit
        
        # Tokenize medical codes
        token_embeddings = []
        for code, desc in zip(codes, code_descs):
            # In a real implementation, you would load the corresponding graph
            # and use medtok_model.tokenize() to get token indices
            # Then use medtok_model.get_token_embedding() to get token embeddings
            
            # For this example, we'll just create random embeddings
            # Simulate token embeddings with shape [num_tokens, embedding_dim]
            num_tokens = self.medtok_model.config.num_top_k_tokens * 4  # 4 types of tokens
            embedding_dim = self.medtok_model.config.embedding_dim
            token_emb = torch.randn(num_tokens, embedding_dim)
            token_embeddings.append(token_emb)
        
        # Pad or truncate to max_seq_len
        if len(token_embeddings) > self.max_seq_len:
            token_embeddings = token_embeddings[:self.max_seq_len]
        
        # Create attention mask
        attention_mask = torch.ones(self.max_seq_len)
        attention_mask[len(token_embeddings):] = 0
        
        # Pad with zeros
        while len(token_embeddings) < self.max_seq_len:
            embedding_dim = self.medtok_model.config.embedding_dim
            num_tokens = self.medtok_model.config.num_top_k_tokens * 4
            token_embeddings.append(torch.zeros(num_tokens, embedding_dim))
        
        # Stack token embeddings
        token_embeddings = torch.stack(token_embeddings)
        
        return {
            'token_embeddings': token_embeddings,
            'attention_mask': attention_mask,
            'outcome': torch.tensor(outcome, dtype=torch.long)
        }


def train_ehr_model(model, train_dataloader, val_dataloader, device, num_epochs=10, lr=1e-4):
    """
    Train the EHR model.
    
    Args:
        model: EHR model
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        device: Device to use
        num_epochs: Number of epochs
        lr: Learning rate
    
    Returns:
        Trained model
    """
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            # Move batch to device
            token_embeddings = batch['token_embeddings'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outcomes = batch['outcome'].to(device)
            
            # Forward pass
            logits = model(token_embeddings, attention_mask)
            loss = criterion(logits, outcomes)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * token_embeddings.size(0)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == outcomes).sum().item()
            train_total += outcomes.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                # Move batch to device
                token_embeddings = batch['token_embeddings'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outcomes = batch['outcome'].to(device)
                
                # Forward pass
                logits = model(token_embeddings, attention_mask)
                loss = criterion(logits, outcomes)
                
                # Track metrics
                val_loss += loss.item() * token_embeddings.size(0)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == outcomes).sum().item()
                val_total += outcomes.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return model


def main():
    """Main function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="EHR model with MEDTOK integration")
    parser.add_argument("--medtok_model", type=str, required=True, help="Path to trained MEDTOK model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="ehr_model", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MEDTOK model
    print(f"Loading MEDTOK model from {args.medtok_model}...")
    medtok_checkpoint = torch.load(args.medtok_model, map_location=device)
    medtok_config = medtok_checkpoint['config']
    medtok_model = MedTok(medtok_config)
    medtok_model.load_state_dict(medtok_checkpoint['model_state_dict'])
    medtok_model.eval()
    
    # Create datasets
    print(f"Creating datasets...")
    train_dataset = EHRDataset(args.train_data, medtok_model)
    val_dataset = EHRDataset(args.val_data, medtok_model)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create EHR model
    print(f"Creating EHR model...")
    ehr_model = SimpleEHRTransformer(
        token_embedding_dim=medtok_config.embedding_dim,
        num_classes=2  # Binary classification (e.g., mortality prediction)
    )
    
    # Train model
    print(f"Training EHR model...")
    ehr_model = train_ehr_model(
        ehr_model, 
        train_dataloader, 
        val_dataloader, 
        device, 
        num_epochs=args.epochs,
        lr=args.lr
    )
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': ehr_model.state_dict(),
        'config': {
            'token_embedding_dim': medtok_config.embedding_dim,
            'hidden_dim': ehr_model.hidden_dim,
            'max_seq_len': ehr_model.max_seq_len,
            'num_classes': 2
        }
    }, os.path.join(args.output_dir, 'ehr_model.pt'))
    
    print(f"Model saved to {os.path.join(args.output_dir, 'ehr_model.pt')}")


if __name__ == "__main__":
    main()
