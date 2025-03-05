#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TransformEHR integration script for MEDTOK.

This script shows how to integrate MEDTOK with the TransformEHR model.
TransformEHR adopts an encoder-decoder transformer with visit-level masking 
to pretrain on EHRs, enabling multi-task prediction.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import logging
import time
from transformers import get_linear_schedule_with_warmup

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.medtok import MedTok
from utils.config import MedTokConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrate MEDTOK with TransformEHR model")
    
    parser.add_argument("--medtok_model", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--transformehr_model", type=str, default=None, 
                        help="Path to TransformEHR model (if None, will initialize new)")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing data (MIMIC-III, MIMIC-IV, or EHRShot)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for results")
    parser.add_argument("--graph_dir", type=str, default=None, 
                        help="Directory containing graph files")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=256, 
                        help="Hidden size for TransformEHR")
    parser.add_argument("--num_layers", type=int, default=4, 
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, 
                        help="Number of attention heads")
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--task", type=str, default="mortality", 
                        help="Task to evaluate (mortality, readmission, los, phenotype, drugrec)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    parser.add_argument("--baseline", action="store_true", 
                        help="Run baseline TransformEHR (without MEDTOK)")
    
    return parser.parse_args()


def load_medtok_model(model_path, device):
    """
    Load a trained MEDTOK model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model onto
    
    Returns:
        MEDTOK model
    """
    logger.info(f"Loading MEDTOK model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', None)
    if config is None:
        logger.error("Config not found in checkpoint")
        return None
    
    # Update device
    config.device = device
    
    # Create model
    model = MedTok(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    logger.info("MEDTOK model loaded")
    
    return model


class TransformEHR(nn.Module):
    """
    Implementation of TransformEHR model.
    
    This is a simplified version of the model described in the paper.
    It includes an encoder-decoder transformer architecture with visit-level masking.
    """
    
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=4, 
                 dropout=0.1, max_seq_length=128, task="mortality"):
        """
        Initialize the TransformEHR model.
        
        Args:
            vocab_size: Size of the medical code vocabulary
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            task: Task to train for (mortality, readmission, etc.)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.task = task
        
        # Code embeddings
        self.code_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_length, hidden_size))
        
        # Visit embeddings
        self.visit_embeddings = nn.Parameter(torch.zeros(1, max_seq_length, hidden_size))
        
        # Encoder transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Decoder transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Task-specific heads
        if task == "mortality" or task == "readmission":
            self.task_head = nn.Linear(hidden_size, 1)
        elif task == "los":
            self.task_head = nn.Linear(hidden_size, 10)  # 10 LOS categories
        elif task == "phenotype":
            self.task_head = nn.Linear(hidden_size, 25)  # 25 phenotype categories
        elif task == "drugrec":
            self.task_head = nn.Linear(hidden_size, 5)  # 5 drug categories
        else:
            self.task_head = nn.Linear(hidden_size, 2)  # Default binary classification
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, code_indices, visit_indices, attention_mask=None):
        """
        Forward pass through the TransformEHR model.
        
        Args:
            code_indices: Indices of medical codes
            visit_indices: Indices of visits
            attention_mask: Optional attention mask
            
        Returns:
            Model predictions
        """
        batch_size, seq_length = code_indices.size()
        
        # Get code embeddings
        code_embeds = self.code_embeddings(code_indices)
        
        # Add position embeddings
        position_ids = torch.arange(seq_length, dtype=torch.long, device=code_indices.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings[:, :seq_length, :]
        
        # Add visit embeddings
        visit_embeds = self.visit_embeddings[:, :seq_length, :]
        
        # Combine embeddings
        embeddings = code_embeds + position_embeds + visit_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout_layer(embeddings)
        
        # Create causal mask for auto-regressive training
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=code_indices.device)
        
        # Create attention mask for transformer
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Apply transformer encoder
        encoder_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=~(attention_mask.bool())
        )
        
        # Apply transformer decoder (for masked prediction or seq2seq tasks)
        decoder_output = self.transformer_decoder(
            embeddings,
            encoder_output,
            tgt_key_padding_mask=~(attention_mask.bool()),
            memory_key_padding_mask=~(attention_mask.bool())
        )
        
        # For classification tasks, use the [CLS] token (first token)
        pooled_output = encoder_output[:, 0]
        
        # Apply task-specific head
        logits = self.task_head(pooled_output)
        
        # Apply sigmoid or softmax based on task
        if self.task in ["mortality", "readmission"]:
            predictions = torch.sigmoid(logits)
        else:
            predictions = torch.softmax(logits, dim=-1)
        
        return predictions


def load_transformehr_model(model_path=None, config=None, device="cuda"):
    """
    Load or initialize TransformEHR model.
    
    Args:
        model_path: Path to model (if None, will initialize new)
        config: Configuration dictionary
        device: Device to load model onto
        
    Returns:
        TransformEHR model
    """
    logger.info("Loading/initializing TransformEHR model...")
    
    try:
        if model_path is not None:
            # Load pre-trained model
            model_data = torch.load(model_path, map_location=device)
            
            if isinstance(model_data, dict):
                # Get model state dict and config
                state_dict = model_data.get("model_state_dict", model_data)
                loaded_config = model_data.get("config", {})
                
                # Update config with loaded config
                if config is None:
                    config = loaded_config
                else:
                    for key, value in loaded_config.items():
                        if key not in config:
                            config[key] = value
            else:
                state_dict = model_data
            
            # Create model with config
            model = TransformEHR(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                dropout=config.get("dropout", 0.1),
                max_seq_length=config.get("max_seq_length", 128),
                task=config.get("task", "mortality")
            )
            
            # Load state dict
            model.load_state_dict(state_dict)
        else:
            # Initialize new model
            model = TransformEHR(
                vocab_size=config["vocab_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                dropout=config.get("dropout", 0.1),
                max_seq_length=config.get("max_seq_length", 128),
                task=config.get("task", "mortality")
            )
        
        # Move to device
        model = model.to(device)
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading/initializing TransformEHR model: {e}")
        
        # Create a simple model with default parameters
        logger.info("Creating TransformEHR model with default parameters...")
        
        if config is None:
            config = {
                "vocab_size": 10000,
                "hidden_size": 256,
                "num_layers": 4,
                "num_heads": 4,
                "dropout": 0.1,
                "max_seq_length": 128,
                "task": "mortality"
            }
        
        model = TransformEHR(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config.get("dropout", 0.1),
            max_seq_length=config.get("max_seq_length", 128),
            task=config.get("task", "mortality")
        ).to(device)
        
        return model


def tokenize_ehr_data_with_medtok(data, medtok_model, graph_dir):
    """
    Tokenize EHR data with MEDTOK.
    
    Args:
        data: Dictionary with patient data
        medtok_model: MEDTOK model
        graph_dir: Directory containing graph files
        
    Returns:
        Dictionary with tokenized data
    """
    logger.info("Tokenizing EHR data with MEDTOK...")
    
    # Get device
    device = next(medtok_model.parameters()).device
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(medtok_model.config.text_encoder_model)
    
    # Process each patient
    tokenized_data = {
        "train": {"patients": [], "codes": [], "visits": [], "labels": []},
        "val": {"patients": [], "codes": [], "visits": [], "labels": []},
        "test": {"patients": [], "codes": [], "visits": [], "labels": []}
    }
    
    for split in ["train", "val", "test"]:
        for patient_idx, patient_id in enumerate(tqdm(data[split]["patients"], desc=f"Tokenizing {split} data")):
            # Get patient visits
            visits = data[split]["visits"][patient_idx]
            
            # Tokenize each code in each visit
            tokenized_visits = []
            for visit in visits:
                tokenized_codes = []
                
                for code in visit["codes"]:
                    # Get code details
                    code_id = code["code_id"]
                    description = code["description"]
                    
                    # Determine graph file path
                    graph_file = os.path.join(graph_dir, f"{code_id}.json")
                    
                    # Tokenize with MEDTOK
                    token_indices = tokenize_medical_code(
                        medtok_model,
                        tokenizer,
                        code_id,
                        description,
                        graph_file,
                        device
                    )
                    
                    # Get token embeddings
                    token_embeddings = medtok_model.get_token_embedding(token_indices)
                    
                    # Store tokenized code
                    tokenized_codes.append({
                        "code_id": code_id,
                        "token_indices": token_indices.cpu().numpy().tolist(),
                        "token_embeddings": token_embeddings.cpu().numpy().tolist()
                    })
                
                # Store tokenized visit
                tokenized_visits.append({
                    "visit_id": visit["visit_id"],
                    "codes": tokenized_codes
                })
            
            # Store tokenized patient data
            tokenized_data[split]["patients"].append(patient_id)
            tokenized_data[split]["visits"].append(tokenized_visits)
            tokenized_data[split]["labels"].append(data[split]["labels"][patient_idx])
    
    logger.info("EHR data tokenized with MEDTOK")
    
    return tokenized_data


def tokenize_medical_code(medtok_model, tokenizer, code_id, description, graph_file, device):
    """
    Tokenize a medical code with MEDTOK.
    
    Args:
        medtok_model: MEDTOK model
        tokenizer: Text tokenizer
        code_id: Medical code ID
        description: Description of the code
        graph_file: Path to the graph file
        device: Device to use
        
    Returns:
        Token indices for the code
    """
    # Tokenize text
    encoded_text = tokenizer(
        description,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Load graph
    if os.path.exists(graph_file):
        # Load graph file
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        
        # Convert to networkx graph
        import networkx as nx
        G = nx.node_link_graph(graph_data)
        
        # Extract node features
        node_features = []
        for node in G.nodes:
            if "features" in G.nodes[node]:
                node_features.append(G.nodes[node]["features"])
            else:
                # Create default features
                node_features.append([0.0] * medtok_model.config.node_feature_dim)
        
        # Extract edge indices
        edge_index = []
        for src, dst in G.edges:
            edge_index.append([src, dst])
            edge_index.append([dst, src])  # Add reverse edge for undirected graphs
        
        # Convert to torch tensors
        node_features = torch.tensor(node_features, dtype=torch.float).to(device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    else:
        # Create dummy graph
        node_features = torch.zeros((1, medtok_model.config.node_feature_dim), dtype=torch.float).to(device)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    
    # Create batch tensor for graph
    graph_batch = torch.zeros(node_features.size(0), dtype=torch.long, device=device)
    
    # Tokenize
    with torch.no_grad():
        token_indices = medtok_model.tokenize(
            encoded_text["input_ids"],
            node_features,
            edge_index,
            graph_batch
        )
    
    return token_indices


def enhance_transformehr_with_medtok(transformehr_model, medtok_model, data, graph_dir):
    """
    Enhance TransformEHR with MEDTOK embeddings.
    
    Args:
        transformehr_model: TransformEHR model
        medtok_model: MEDTOK model
        data: EHR data
        graph_dir: Directory containing graph files
        
    Returns:
        Enhanced TransformEHR model and code-to-embedding mapping
    """
    logger.info("Enhancing TransformEHR with MEDTOK embeddings...")
    
    # Create a mapping from codes to MEDTOK embeddings
    code_to_embedding = {}
    
    # Process each code in the data
    all_codes = set()
    for split in ["train", "val", "test"]:
        for patient_visits in data[split]["visits"]:
            for visit in patient_visits:
                for code in visit["codes"]:
                    all_codes.add(code["code_id"])
    
    logger.info(f"Processing {len(all_codes)} unique codes...")
    
    # Get device
    device = next(medtok_model.parameters()).device
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(medtok_model.config.text_encoder_model)
    
    # Process each code
    for code_id in tqdm(all_codes, desc="Processing codes"):
        # Get code description
        description = "Medical code: " + code_id  # Fallback description
        
        # Find description in data
        for split in ["train", "val", "test"]:
            for patient_visits in data[split]["visits"]:
                for visit in patient_visits:
                    for code in visit["codes"]:
                        if code["code_id"] == code_id and "description" in code:
                            description = code["description"]
                            break
        
        # Determine graph file path
        graph_file = os.path.join(graph_dir, f"{code_id}.json")
        
        # Tokenize with MEDTOK
        token_indices = tokenize_medical_code(
            medtok_model,
            tokenizer,
            code_id,
            description,
            graph_file,
            device
        )
        
        # Get token embeddings and average them
        token_embeddings = medtok_model.get_token_embedding(token_indices)
        avg_embedding = token_embeddings.mean(dim=1).cpu().numpy()
        
        # Store in mapping
        code_to_embedding[code_id] = avg_embedding
    
    # Update the embedding table in TransformEHR
    # First, create a mapping from code_id to index in the vocab
    code_to_idx = {code: idx+1 for idx, code in enumerate(sorted(all_codes))}  # +1 to reserve 0 for padding
    
    # Create a new embedding matrix
    embedding_dim = transformehr_model.hidden_size
    new_embeddings = torch.zeros(len(code_to_idx) + 1, embedding_dim)  # +1 for padding token
    
    # Initialize with random values (for codes without MEDTOK embeddings)
    nn.init.normal_(new_embeddings, mean=0, std=0.02)
    
    # Fill with MEDTOK embeddings
    for code_id, idx in code_to_idx.items():
        if code_id in code_to_embedding:
            embedding = code_to_embedding[code_id]
            
            # Ensure embedding has the right shape
            if embedding.shape[0] != embedding_dim:
                # Resize embedding if needed
                if embedding.shape[0] < embedding_dim:
                    # Pad with zeros
                    padding = np.zeros(embedding_dim - embedding.shape[0])
                    embedding = np.concatenate([embedding, padding])
                else:
                    # Truncate
                    embedding = embedding[:embedding_dim]
            
            new_embeddings[idx] = torch.tensor(embedding, dtype=torch.float)
    
    # Replace the embedding table in TransformEHR
    transformehr_model.code_embeddings = nn.Embedding.from_pretrained(new_embeddings, freeze=False)
    
    # Adjust model for new vocabulary size
    transformehr_model.vocab_size = len(code_to_idx) + 1
    
    logger.info(f"TransformEHR enhanced with MEDTOK embeddings for {len(code_to_idx)} codes")
    
    return transformehr_model, code_to_idx


def process_patient_data(patient_data, code_to_idx, max_seq_length=128):
    """
    Process patient data into model inputs.
    
    Args:
        patient_data: Patient data
        code_to_idx: Mapping from code_id to index
        max_seq_length: Maximum sequence length
        
    Returns:
        Processed inputs for the model
    """
    # Get all visits
    visits = patient_data["visits"]
    
    # Create code indices and visit indices
    code_indices = []
    visit_indices = []
    
    for visit_idx, visit in enumerate(visits):
        for code in visit["codes"]:
            code_id = code["code_id"]
            code_idx = code_to_idx.get(code_id, 0)  # 0 for unknown codes
            
            code_indices.append(code_idx)
            visit_indices.append(visit_idx)
    
    # Truncate or pad to max_seq_length
    if len(code_indices) > max_seq_length:
        # Truncate
        code_indices = code_indices[:max_seq_length]
        visit_indices = visit_indices[:max_seq_length]
    else:
        # Pad
        padding_length = max_seq_length - len(code_indices)
        code_indices.extend([0] * padding_length)
        visit_indices.extend([0] * padding_length)
    
    # Create attention mask
    attention_mask = [1] * len(code_indices)
    attention_mask.extend([0] * (max_seq_length - len(code_indices)))
    attention_mask = attention_mask[:max_seq_length]
    
    # Convert to tensors
    code_indices_tensor = torch.tensor(code_indices, dtype=torch.long)
    visit_indices_tensor = torch.tensor(visit_indices, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.float)
    
    return code_indices_tensor, visit_indices_tensor, attention_mask_tensor


def train_transformehr(model, data, code_to_idx, device, epochs=20, lr=1e-4, batch_size=32, max_seq_length=128):
    """
    Train the TransformEHR model.
    
    Args:
        model: TransformEHR model
        data: Training data
        code_to_idx: Mapping from code_id to index
        device: Device to use
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        max_seq_length: Maximum sequence length
        
    Returns:
        Trained model
    """
    logger.info("Training TransformEHR model...")
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Set up loss function based on task
    if model.task in ["mortality", "readmission"]:
        loss_fn = nn.BCELoss()
    elif model.task in ["los", "phenotype", "drugrec"]:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCELoss()
    
    # Calculate total steps
    total_steps = (len(data["train"]["patients"]) // batch_size) * epochs
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Process in batches
        for i in range(0, len(data["train"]["patients"]), batch_size):
            batch_indices = range(i, min(i + batch_size, len(data["train"]["patients"])))
            
            # Process patient data
            batch_code_indices = []
            batch_visit_indices = []
            batch_attention_masks = []
            batch_labels = []
            
            for idx in batch_indices:
                patient_data = {
                    "visits": data["train"]["visits"][idx]
                }
                
                code_indices, visit_indices, attention_mask = process_patient_data(
                    patient_data,
                    code_to_idx,
                    max_seq_length
                )
                
                batch_code_indices.append(code_indices)
                batch_visit_indices.append(visit_indices)
                batch_attention_masks.append(attention_mask)
                
                # Get label
                label = data["train"]["labels"][idx]
                batch_labels.append(label)
            
            # Convert to tensors
            batch_code_indices = torch.stack(batch_code_indices).to(device)
            batch_visit_indices = torch.stack(batch_visit_indices).to(device)
            batch_attention_masks = torch.stack(batch_attention_masks).to(device)
            
            if model.task in ["mortality", "readmission"]:
                batch_labels = torch.tensor(batch_labels, dtype=torch.float).to(device)
            else:
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(batch_code_indices, batch_visit_indices, batch_attention_masks)
            
            # Calculate loss
            if model.task in ["mortality", "readmission"]:
                loss = loss_fn(outputs.squeeze(), batch_labels)
            else:
                loss = loss_fn(outputs, batch_labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Calculate average loss
        avg_train_loss = train_loss / (len(data["train"]["patients"]) // batch_size)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(data["val"]["patients"]), batch_size):
                batch_indices = range(i, min(i + batch_size, len(data["val"]["patients"])))
                
                # Process patient data
                batch_code_indices = []
                batch_visit_indices = []
                batch_attention_masks = []
                batch_labels = []
                
                for idx in batch_indices:
                    patient_data = {
                        "visits": data["val"]["visits"][idx]
                    }
                    
                    code_indices, visit_indices, attention_mask = process_patient_data(
                        patient_data,
                        code_to_idx,
                        max_seq_length
                    )
                    
                    batch_code_indices.append(code_indices)
                    batch_visit_indices.append(visit_indices)
                    batch_attention_masks.append(attention_mask)
                    
                    # Get label
                    label = data["val"]["labels"][idx]
                    batch_labels.append(label)
                
                # Convert to tensors
                batch_code_indices = torch.stack(batch_code_indices).to(device)
                batch_visit_indices = torch.stack(batch_visit_indices).to(device)
                batch_attention_masks = torch.stack(batch_attention_masks).to(device)
                
                if model.task in ["mortality", "readmission"]:
                    batch_labels = torch.tensor(batch_labels, dtype=torch.float).to(device)
                else:
                    batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
                
                # Forward pass
                outputs = model(batch_code_indices, batch_visit_indices, batch_attention_masks)
                
                # Calculate loss
                if model.task in ["mortality", "readmission"]:
                    loss = loss_fn(outputs.squeeze(), batch_labels)
                else:
                    loss = loss_fn(outputs, batch_labels)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / (len(data["val"]["patients"]) // batch_size)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss
            }
            logger.info(f"New best model saved (val_loss: {avg_val_loss:.4f})")
    
    # Restore best model
    if best_model is not None:
        model.load_state_dict(best_model["model_state_dict"])
        logger.info(f"Restored best model from epoch {best_model['epoch']+1} with val_loss: {best_model['val_loss']:.4f}")
    
    logger.info("Training completed")
    return model


def evaluate_transformehr(model, data, code_to_idx, device, batch_size=32, max_seq_length=128):
    """
    Evaluate the TransformEHR model.
    
    Args:
        model: TransformEHR model
        data: Test data
        code_to_idx: Mapping from code_id to index
        device: Device to use
        batch_size: Batch size
        max_seq_length: Maximum sequence length
        
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating TransformEHR model...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(data["test"]["patients"]), batch_size):
            batch_indices = range(i, min(i + batch_size, len(data["test"]["patients"])))
            
            # Process patient data
            batch_code_indices = []
            batch_visit_indices = []
            batch_attention_masks = []
            batch_labels = []
            
            for idx in batch_indices:
                patient_data = {
                    "visits": data["test"]["visits"][idx]
                }
                
                code_indices, visit_indices, attention_mask = process_patient_data(
                    patient_data,
                    code_to_idx,
                    max_seq_length
                )
                
                batch_code_indices.append(code_indices)
                batch_visit_indices.append(visit_indices)
                batch_attention_masks.append(attention_mask)
                
                # Get label
                label = data["test"]["labels"][idx]
                batch_labels.append(label)
            
            # Convert to tensors
            batch_code_indices = torch.stack(batch_code_indices).to(device)
            batch_visit_indices = torch.stack(batch_visit_indices).to(device)
            batch_attention_masks = torch.stack(batch_attention_masks).to(device)
            
            # Forward pass
            outputs = model(batch_code_indices, batch_visit_indices, batch_attention_masks)
            
            # Store predictions and labels
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(batch_labels)
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics based on task
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
    
    metrics = {}
    
    if model.task in ["mortality", "readmission"]:
        # Binary classification
        try:
            metrics["auroc"] = roc_auc_score(all_labels, all_preds)
        except:
            metrics["auroc"] = 0.0
        
        try:
            metrics["auprc"] = average_precision_score(all_labels, all_preds)
        except:
            metrics["auprc"] = 0.0
        
        # Convert predictions to binary
        preds_binary = (all_preds > 0.5).astype(int)
        metrics["accuracy"] = accuracy_score(all_labels, preds_binary)
        metrics["f1"] = f1_score(all_labels, preds_binary)
    
    elif model.task in ["los", "phenotype", "drugrec"]:
        # Multi-class or multi-label classification
        if all_preds.shape[1] > 1:
            # Multi-class: convert to class indices
            preds_classes = np.argmax(all_preds, axis=1)
            metrics["accuracy"] = accuracy_score(all_labels, preds_classes)
            
            # Calculate macro-averaged metrics
            metrics["f1_macro"] = f1_score(all_labels, preds_classes, average="macro")
            
            # For multi-class ROC AUC, need to convert labels to one-hot encoding
            try:
                from sklearn.preprocessing import OneHotEncoder
                encoder = OneHotEncoder(sparse=False)
                labels_onehot = encoder.fit_transform(all_labels.reshape(-1, 1))
                metrics["auroc_macro"] = roc_auc_score(labels_onehot, all_preds, average="macro")
            except:
                metrics["auroc_macro"] = 0.0
        else:
            # Binary case
            try:
                metrics["auroc"] = roc_auc_score(all_labels, all_preds)
            except:
                metrics["auroc"] = 0.0
            
            try:
                metrics["auprc"] = average_precision_score(all_labels, all_preds)
            except:
                metrics["auprc"] = 0.0
    
    logger.info(f"Evaluation results: {metrics}")
    
    return metrics


def load_ehr_data(data_dir, task):
    """
    Load EHR data for the specified task.
    
    Args:
        data_dir: Directory containing data
        task: Task name
        
    Returns:
        Dictionary with patient data
    """
    logger.info(f"Loading EHR data for task: {task}")
    
    # For this example, we'll create placeholder data
    # In a real implementation, this would load actual data files
    
    data = {
        "train": {
            "patients": [f"patient_{i}" for i in range(100)],
            "visits": [],
            "labels": np.random.randint(0, 2, 100) if task in ["mortality", "readmission"] else np.random.randint(0, 10, 100)
        },
        "val": {
            "patients": [f"patient_{i}" for i in range(100, 150)],
            "visits": [],
            "labels": np.random.randint(0, 2, 50) if task in ["mortality", "readmission"] else np.random.randint(0, 10, 50)
        },
        "test": {
            "patients": [f"patient_{i}" for i in range(150, 200)],
            "visits": [],
            "labels": np.random.randint(0, 2, 50) if task in ["mortality", "readmission"] else np.random.randint(0, 10, 50)
        }
    }
    
    # Create dummy visits for each patient
    for split in ["train", "val", "test"]:
        for i in range(len(data[split]["patients"])):
            num_visits = np.random.randint(1, 5)  # 1-4 visits per patient
            visits = []
            
            for j in range(num_visits):
                num_codes = np.random.randint(3, 10)  # 3-9 codes per visit
                codes = []
                
                for k in range(num_codes):
                    # Generate random code
                    if np.random.random() < 0.7:
                        # ICD code
                        code_type = np.random.choice(["ICD9", "ICD10"])
                        if code_type == "ICD9":
                            code_id = f"{code_type}:{np.random.randint(1, 999)}.{np.random.randint(1, 9)}"
                        else:
                            code_id = f"{code_type}:{chr(65 + np.random.randint(0, 26))}{np.random.randint(1, 99)}.{np.random.randint(1, 9)}"
                        description = f"Diagnosis code for condition {code_id}"
                    else:
                        # Medication code
                        code_type = np.random.choice(["RxNorm", "ATC"])
                        code_id = f"{code_type}:{np.random.randint(10000, 99999)}"
                        description = f"Medication code for drug {code_id}"
                    
                    codes.append({
                        "code_id": code_id,
                        "description": description
                    })
                
                visits.append({
                    "visit_id": f"visit_{j}",
                    "codes": codes
                })
            
            data[split]["visits"].append(visits)
    
    logger.info(f"Loaded data with {len(data['train']['patients'])} train, {len(data['val']['patients'])} val, and {len(data['test']['patients'])} test patients")
    
    return data


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Load EHR data
    data = load_ehr_data(args.data_dir, args.task)
    
    # Determine graph directory
    graph_dir = args.graph_dir
    if graph_dir is None:
        graph_dir = os.path.join(args.data_dir, "graphs")
        if not os.path.exists(graph_dir):
            logger.error(f"Graph directory not found: {graph_dir}")
            logger.error("Please specify --graph_dir")
            return
    
    # Baseline TransformEHR (without MEDTOK)
    if args.baseline:
        logger.info("Running baseline TransformEHR (without MEDTOK)...")
        
        # Create vocabulary from data
        all_codes = set()
        for split in ["train", "val", "test"]:
            for patient_visits in data[split]["visits"]:
                for visit in patient_visits:
                    for code in visit["codes"]:
                        all_codes.add(code["code_id"])
        
        # Create mapping from code to index
        baseline_code_to_idx = {code: idx+1 for idx, code in enumerate(sorted(all_codes))}  # +1 to reserve 0 for padding
        
        # Create TransformEHR config
        baseline_config = {
            "vocab_size": len(baseline_code_to_idx) + 1,  # +1 for padding
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": 0.1,
            "max_seq_length": args.max_seq_length,
            "task": args.task
        }
        
        # Initialize or load TransformEHR model
        baseline_model = load_transformehr_model(
            model_path=args.transformehr_model,
            config=baseline_config,
            device=device
        )
        
        # Train baseline model
        baseline_model = train_transformehr(
            baseline_model,
            data,
            baseline_code_to_idx,
            device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length
        )
        
        # Save baseline model
        baseline_model_path = os.path.join(args.output_dir, "transformehr_baseline.pt")
        torch.save({
            "model_state_dict": baseline_model.state_dict(),
            "config": baseline_config
        }, baseline_model_path)
        logger.info(f"Baseline model saved to {baseline_model_path}")
        
        # Evaluate baseline model
        baseline_metrics = evaluate_transformehr(
            baseline_model,
            data,
            baseline_code_to_idx,
            device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length
        )
        
        # Save baseline metrics
        baseline_metrics_path = os.path.join(args.output_dir, "baseline_metrics.json")
        with open(baseline_metrics_path, "w") as f:
            json.dump(baseline_metrics, f, indent=2)
        
        logger.info(f"Baseline metrics saved to {baseline_metrics_path}")
    
    # TransformEHR with MEDTOK
    logger.info("Running TransformEHR with MEDTOK integration...")
    
    # Load MEDTOK model
    medtok_model = load_medtok_model(args.medtok_model, device)
    
    if medtok_model is None:
        logger.error("Failed to load MEDTOK model")
        return
    
    # Enhance TransformEHR model with MEDTOK
    # First, create initial TransformEHR model
    initial_config = {
        "vocab_size": 10000,  # Will be updated during enhancement
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": 0.1,
        "max_seq_length": args.max_seq_length,
        "task": args.task
    }
    
    transformehr_model = load_transformehr_model(
        model_path=None,  # Start with a fresh model for MEDTOK integration
        config=initial_config,
        device=device
    )
    
    # Enhance with MEDTOK
    enhanced_model, code_to_idx = enhance_transformehr_with_medtok(
        transformehr_model,
        medtok_model,
        data,
        graph_dir
    )
    
    # Update config with new vocabulary size
    enhanced_config = {
        "vocab_size": enhanced_model.vocab_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": 0.1,
        "max_seq_length": args.max_seq_length,
        "task": args.task
    }
    
    # Train enhanced model
    enhanced_model = train_transformehr(
        enhanced_model,
        data,
        code_to_idx,
        device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length
    )
    
    # Save enhanced model
    enhanced_model_path = os.path.join(args.output_dir, "transformehr_medtok.pt")
    torch.save({
        "model_state_dict": enhanced_model.state_dict(),
        "config": enhanced_config
    }, enhanced_model_path)
    logger.info(f"Enhanced model saved to {enhanced_model_path}")
    
    # Evaluate enhanced model
    enhanced_metrics = evaluate_transformehr(
        enhanced_model,
        data,
        code_to_idx,
        device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length
    )
    
    # Save enhanced metrics
    enhanced_metrics_path = os.path.join(args.output_dir, "medtok_metrics.json")
    with open(enhanced_metrics_path, "w") as f:
        json.dump(enhanced_metrics, f, indent=2)
    
    logger.info(f"Enhanced metrics saved to {enhanced_metrics_path}")
    
    # Compare results if baseline was run
    if args.baseline:
        # Create comparison report
        comparison = {
            "baseline": baseline_metrics,
            "medtok_enhanced": enhanced_metrics,
            "improvements": {}
        }
        
        # Calculate improvements
        for metric in baseline_metrics:
            if metric in enhanced_metrics:
                improvement = enhanced_metrics[metric] - baseline_metrics[metric]
                comparison["improvements"][metric] = improvement
                
                # Calculate relative improvement
                if baseline_metrics[metric] > 0:
                    rel_improvement = improvement / baseline_metrics[metric] * 100
                    comparison["improvements"][f"{metric}_relative"] = f"{rel_improvement:.2f}%"
        
        # Save comparison
        comparison_path = os.path.join(args.output_dir, "comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison saved to {comparison_path}")
        
        # Print summary
        logger.info("Comparison summary:")
        if "auprc" in comparison["improvements"]:
            logger.info(f"  AUPRC: {baseline_metrics['auprc']:.4f} → {enhanced_metrics['auprc']:.4f} " + 
                      f"(+{comparison['improvements']['auprc']:.4f}, {comparison['improvements']['auprc_relative']})")
        elif "auroc" in comparison["improvements"]:
            logger.info(f"  AUROC: {baseline_metrics['auroc']:.4f} → {enhanced_metrics['auroc']:.4f} " + 
                      f"(+{comparison['improvements']['auroc']:.4f}, {comparison['improvements']['auroc_relative']})")
        
        if "accuracy" in comparison["improvements"]:
            logger.info(f"  Accuracy: {baseline_metrics['accuracy']:.4f} → {enhanced_metrics['accuracy']:.4f} " + 
                      f"(+{comparison['improvements']['accuracy']:.4f}, {comparison['improvements']['accuracy_relative']})")
        
        if "f1" in comparison["improvements"]:
            logger.info(f"  F1 Score: {baseline_metrics['f1']:.4f} → {enhanced_metrics['f1']:.4f} " + 
                      f"(+{comparison['improvements']['f1']:.4f}, {comparison['improvements']['f1_relative']})")


if __name__ == "__main__":
    main()
