#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration script for combining MEDTOK with GT-BEHRT.

GT-BEHRT (Graph Transformer BEHRT) is a model that processes EHR data by first
modeling intra-visit dependencies as a graph, using a graph transformer to learn visit
representations, and then processing patient-level sequences with a transformer encoder.

This script demonstrates how to enhance GT-BEHRT by replacing its standard tokenization
with MEDTOK, improving performance on various tasks.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import logging
import networkx as nx
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from transformers import get_linear_schedule_with_warmup
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.medtok import MedTok
from utils.config import MedTokConfig
from utils.metrics import compute_medtok_metrics, evaluate_downstream_task


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Integrate MEDTOK with GT-BEHRT")
    
    parser.add_argument("--medtok_model", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--mimic_dir", type=str, required=True, 
                        help="Directory containing MIMIC data")
    parser.add_argument("--graph_dir", type=str, default=None, 
                        help="Directory containing graph files (required if not in mimic_dir)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for results")
    parser.add_argument("--task", type=str, default="mortality", 
                        help="Task to evaluate (mortality, readmission, los, phenotype, drugrec)")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=20, 
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, 
                        help="Warmup steps for learning rate scheduler")
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--hidden_size", type=int, default=256, 
                        help="Hidden size for GT-BEHRT")
    parser.add_argument("--num_heads", type=int, default=4, 
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, 
                        help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, 
                        help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--compare_baseline", action="store_true", 
                        help="Compare with baseline GT-BEHRT (without MEDTOK)")
    
    return parser.parse_args()


class VisitGraphTransformer(nn.Module):
    """
    Visit-level graph transformer for GT-BEHRT.
    
    This module processes each visit as a graph, modeling the dependencies
    between medical codes within a visit using graph neural networks.
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        """
        Initialize the visit graph transformer.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Graph neural network layers
        self.gat1 = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim, dropout=dropout)
        
        # Layer normalization and dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through the visit graph transformer.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity in COO format [2, num_edges]
            batch: Batch assignment for nodes in disjoint graphs [num_nodes]
            
        Returns:
            Graph-level representations [batch_size, hidden_dim]
        """
        # First GAT layer with residual connection
        h1 = F.relu(self.gat1(x, edge_index))
        h1 = self.dropout(h1)
        
        # Second GAT layer
        h2 = self.gat2(h1, edge_index)
        h2 = self.layer_norm2(h2)
        
        # Apply global pooling to get graph-level representations
        if batch is None:
            batch = torch.zeros(h2.size(0), dtype=torch.long, device=h2.device)
        
        graph_embedding = global_mean_pool(h2, batch)
        
        return graph_embedding


class PatientTransformer(nn.Module):
    """
    Patient-level transformer for GT-BEHRT.
    
    This module processes sequences of visit representations using a transformer encoder.
    """
    
    def __init__(self, hidden_dim, num_heads=4, num_layers=4, dropout=0.1):
        """
        Initialize the patient transformer.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Positional embedding
        self.position_embedding = nn.Parameter(torch.zeros(1, 512, hidden_dim))
        
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
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, attention_mask=None):
        """
        Forward pass through the patient transformer.
        
        Args:
            x: Visit representations [batch_size, seq_length, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Sequence of transformed representations [batch_size, seq_length, hidden_dim]
        """
        # Add positional embeddings
        seq_length = x.size(1)
        x = x + self.position_embedding[:, :seq_length, :]
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert attention mask to correct format for transformer
            attention_mask = attention_mask.bool()
            
            # Apply transformer encoder
            x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        else:
            # Apply transformer encoder without mask
            x = self.transformer_encoder(x)
        
        return x


class GTBEHRT(nn.Module):
    """
    GT-BEHRT (Graph Transformer BEHRT) model with MEDTOK integration.
    
    This model processes EHR data by first using MEDTOK for tokenization,
    then modeling intra-visit dependencies as a graph, and finally processing
    patient-level sequences with a transformer encoder.
    """
    
    def __init__(self, medtok_model, config):
        """
        Initialize the GT-BEHRT model.
        
        Args:
            medtok_model: MEDTOK model for tokenization
            config: Configuration object
        """
        super().__init__()
        
        self.medtok = medtok_model
        self.config = config
        
        # Freeze MEDTOK model
        for param in self.medtok.parameters():
            param.requires_grad = False
        
        # Embedding dimension from MEDTOK tokens
        self.token_embedding_dim = medtok_model.config.embedding_dim
        
        # Visit graph transformer
        self.visit_transformer = VisitGraphTransformer(
            input_dim=self.token_embedding_dim,
            hidden_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Patient-level transformer
        self.patient_transformer = PatientTransformer(
            hidden_dim=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
        # Task-specific head based on the task
        if config.task_type == "binary_classification":
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, 1)
            )
        elif config.task_type == "multiclass_classification":
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.num_classes)
            )
        elif config.task_type == "multilabel_classification":
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.num_labels)
            )
    
    def forward(self, patient_data, attention_mask=None):
        """
        Forward pass through the GT-BEHRT model.
        
        Args:
            patient_data: Dictionary containing patient visit data
            attention_mask: Attention mask for patient-level transformer
            
        Returns:
            Model predictions
        """
        # Extract patient visit data
        visits = patient_data["visits"]
        
        # Initialize list to store visit representations
        visit_embeddings = []
        
        # Process each visit
        for visit in visits:
            # Tokenize medical codes in the visit using MEDTOK
            token_embeddings = []
            
            for code_data in visit["codes"]:
                # Get token indices and embeddings
                token_indices = code_data["token_indices"]
                token_embedding = self.medtok.get_token_embedding(token_indices)
                token_embeddings.append(token_embedding)
            
            # Create graph from visit codes
            code_features = torch.cat(token_embeddings, dim=0)
            
            # Create edges based on code interactions within the visit
            # For simplicity, we connect all codes in the visit
            num_codes = len(visit["codes"])
            edge_index = []
            
            for i in range(num_codes):
                for j in range(num_codes):
                    if i != j:
                        edge_index.append([i, j])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Process visit as a graph
            visit_embedding = self.visit_transformer(code_features, edge_index)
            visit_embeddings.append(visit_embedding)
        
        # Stack visit embeddings
        if visit_embeddings:
            visit_embeddings = torch.stack(visit_embeddings, dim=0)
        else:
            # Handle empty sequence
            visit_embeddings = torch.zeros(1, self.config.hidden_size, device=self.medtok.config.device)
        
        # Add batch dimension if needed
        if visit_embeddings.dim() == 2:
            visit_embeddings = visit_embeddings.unsqueeze(0)
        
        # Apply patient-level transformer
        patient_embeddings = self.patient_transformer(visit_embeddings, attention_mask)
        
        # Get the representation for prediction (use the last visit)
        if attention_mask is not None:
            # Use the last valid position for each sequence
            last_positions = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(patient_embeddings.size(0), device=patient_embeddings.device)
            last_hidden = patient_embeddings[batch_indices, last_positions]
        else:
            # Use the last position
            last_hidden = patient_embeddings[:, -1]
        
        # Apply task-specific classifier
        logits = self.classifier(last_hidden)
        
        # Apply sigmoid or softmax based on task type
        if self.config.task_type == "binary_classification":
            predictions = torch.sigmoid(logits)
        elif self.config.task_type == "multiclass_classification":
            predictions = torch.softmax(logits, dim=-1)
        elif self.config.task_type == "multilabel_classification":
            predictions = torch.sigmoid(logits)
        
        return predictions


class BaselineGTBEHRT(nn.Module):
    """
    Baseline GT-BEHRT model without MEDTOK integration.
    
    This version uses standard embedding approach instead of MEDTOK tokenization.
    """
    
    def __init__(self, vocab_size, config):
        """
        Initialize the baseline GT-BEHRT model.
        
        Args:
            vocab_size: Size of the medical code vocabulary
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Embedding layer for medical codes
        self.code_embedding = nn.Embedding(vocab_size, config.hidden_size)
        
        # Visit graph transformer
        self.visit_transformer = VisitGraphTransformer(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Patient-level transformer
        self.patient_transformer = PatientTransformer(
            hidden_dim=config.hidden_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
        
        # Task-specific head based on the task
        if config.task_type == "binary_classification":
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, 1)
            )
        elif config.task_type == "multiclass_classification":
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.num_classes)
            )
        elif config.task_type == "multilabel_classification":
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.num_labels)
            )
    
    def forward(self, patient_data, attention_mask=None):
        """
        Forward pass through the baseline GT-BEHRT model.
        
        Args:
            patient_data: Dictionary containing patient visit data
            attention_mask: Attention mask for patient-level transformer
            
        Returns:
            Model predictions
        """
        # Extract patient visit data
        visits = patient_data["visits"]
        
        # Initialize list to store visit representations
        visit_embeddings = []
        
        # Process each visit
        for visit in visits:
            # Get code indices
            code_indices = visit["code_indices"]
            
            # Get code embeddings
            code_features = self.code_embedding(code_indices)
            
            # Create edges based on code interactions within the visit
            num_codes = len(code_indices)
            edge_index = []
            
            for i in range(num_codes):
                for j in range(num_codes):
                    if i != j:
                        edge_index.append([i, j])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Process visit as a graph
            visit_embedding = self.visit_transformer(code_features, edge_index)
            visit_embeddings.append(visit_embedding)
        
        # Stack visit embeddings
        if visit_embeddings:
            visit_embeddings = torch.stack(visit_embeddings, dim=0)
        else:
            # Handle empty sequence
            visit_embeddings = torch.zeros(1, self.config.hidden_size, device=next(self.parameters()).device)
        
        # Add batch dimension if needed
        if visit_embeddings.dim() == 2:
            visit_embeddings = visit_embeddings.unsqueeze(0)
        
        # Apply patient-level transformer
        patient_embeddings = self.patient_transformer(visit_embeddings, attention_mask)
        
        # Get the representation for prediction (use the last visit)
        if attention_mask is not None:
            # Use the last valid position for each sequence
            last_positions = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(patient_embeddings.size(0), device=patient_embeddings.device)
            last_hidden = patient_embeddings[batch_indices, last_positions]
        else:
            # Use the last position
            last_hidden = patient_embeddings[:, -1]
        
        # Apply task-specific classifier
        logits = self.classifier(last_hidden)
        
        # Apply sigmoid or softmax based on task type
        if self.config.task_type == "binary_classification":
            predictions = torch.sigmoid(logits)
        elif self.config.task_type == "multiclass_classification":
            predictions = torch.softmax(logits, dim=-1)
        elif self.config.task_type == "multilabel_classification":
            predictions = torch.sigmoid(logits)
        
        return predictions


class EHRDataset(Dataset):
    """
    Dataset for EHR data with MEDTOK tokenization.
    
    This dataset loads patient visit sequences with medical codes
    and tokenizes them using MEDTOK.
    """
    
    def __init__(self, data_file, medtok_model, graph_dir, max_seq_length=128, split="train"):
        """
        Initialize the EHR dataset.
        
        Args:
            data_file: Path to the data file
            medtok_model: MEDTOK model for tokenization
            graph_dir: Directory containing graph files
            max_seq_length: Maximum sequence length
            split: Data split (train, val, test)
        """
        self.max_seq_length = max_seq_length
        self.medtok_model = medtok_model
        self.graph_dir = graph_dir
        self.split = split
        
        # Load data
        self.data = pd.read_csv(data_file)
        
        # Filter by split
        self.data = self.data[self.data["split"] == split]
        
        # Get unique patient IDs
        self.patient_ids = self.data["patient_id"].unique()
        
        # Group by patient ID
        self.patient_groups = self.data.groupby("patient_id")
    
    def __len__(self):
        """Return number of patients."""
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        """
        Get patient data.
        
        Args:
            idx: Index of the patient
            
        Returns:
            Tokenized patient data and labels
        """
        patient_id = self.patient_ids[idx]
        patient_data = self.patient_groups.get_group(patient_id)
        
        # Sort by visit date
        patient_data = patient_data.sort_values("visit_date")
        
        # Get visits
        visits = []
        
        for _, visit_data in patient_data.iterrows():
            # Get codes and labels
            codes = eval(visit_data["codes"])  # Assumes codes are stored as a string representation of a list
            
            # Tokenize codes with MEDTOK
            tokenized_codes = []
            
            for code in codes:
                code_id = code["code_id"]
                description = code["description"]
                
                # Determine graph file path
                graph_file = os.path.join(self.graph_dir, f"{code_id}.json")
                
                # Tokenize with MEDTOK
                token_indices = self.tokenize_code(code_id, description, graph_file)
                
                tokenized_codes.append({
                    "code_id": code_id,
                    "description": description,
                    "token_indices": token_indices
                })
            
            # Create visit representation
            visit = {
                "visit_id": visit_data["visit_id"],
                "date": visit_data["visit_date"],
                "codes": tokenized_codes
            }
            
            visits.append(visit)
        
        # Truncate or pad to max_seq_length
        if len(visits) > self.max_seq_length:
            visits = visits[-self.max_seq_length:]  # Keep most recent visits
        
        # Create attention mask
        attention_mask = torch.ones(self.max_seq_length)
        attention_mask[len(visits):] = 0
        
        # Get labels based on the task
        if "mortality_label" in patient_data.columns:
            label = patient_data["mortality_label"].iloc[-1]
        elif "readmission_label" in patient_data.columns:
            label = patient_data["readmission_label"].iloc[-1]
        elif "los_label" in patient_data.columns:
            label = patient_data["los_label"].iloc[-1]
        else:
            # For multi-label tasks, get all label columns
            label_columns = [col for col in patient_data.columns if col.startswith("label_")]
            label = patient_data[label_columns].iloc[-1].values
        
        return {
            "patient_id": patient_id,
            "visits": visits,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.float if isinstance(label, (int, float)) else torch.long)
        }
    
    def tokenize_code(self, code_id, description, graph_file):
        """
        Tokenize a medical code using MEDTOK.
        
        Args:
            code_id: Medical code ID
            description: Code description
            graph_file: Path to the graph file
            
        Returns:
            Token indices
        """
        device = next(self.medtok_model.parameters()).device
        
        # Load graph file if it exists
        if os.path.exists(graph_file):
            # Load graph
            with open(graph_file, "r") as f:
                graph_data = json.load(f)
            
            # Convert to torch tensors
            G = nx.node_link_graph(graph_data)
            
            # Extract node features
            node_features = []
            for node in G.nodes:
                if "features" in G.nodes[node]:
                    node_features.append(G.nodes[node]["features"])
                else:
                    # Create default features
                    node_features.append([0.0] * self.medtok_model.config.node_feature_dim)
            
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
            node_features = torch.zeros((1, self.medtok_model.config.node_feature_dim), dtype=torch.float).to(device)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        
        # Tokenize text
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.medtok_model.config.text_encoder_model)
        
        encoded_text = tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Create batch tensor for graph
        graph_batch = torch.zeros(node_features.size(0), dtype=torch.long, device=device)
        
        # Tokenize with MEDTOK
        with torch.no_grad():
            token_indices = self.medtok_model.tokenize(
                encoded_text["input_ids"],
                node_features,
                edge_index,
                graph_batch
            )
        
        return token_indices[0].cpu()


class BaselineEHRDataset(Dataset):
    """
    Dataset for EHR data with baseline tokenization (without MEDTOK).
    
    This dataset loads patient visit sequences with medical codes
    and uses standard embedding approach.
    """
    
    def __init__(self, data_file, code_to_idx, max_seq_length=128, split="train"):
        """
        Initialize the baseline EHR dataset.
        
        Args:
            data_file: Path to the data file
            code_to_idx: Dictionary mapping medical codes to indices
            max_seq_length: Maximum sequence length
            split: Data split (train, val, test)
        """
        self.max_seq_length = max_seq_length
        self.code_to_idx = code_to_idx
        self.split = split
        
        # Load data
        self.data = pd.read_csv(data_file)
        
        # Filter by split
        self.data = self.data[self.data["split"] == split]
        
        # Get unique patient IDs
        self.patient_ids = self.data["patient_id"].unique()
        
        # Group by patient ID
        self.patient_groups = self.data.groupby("patient_id")
    
    def __len__(self):
        """Return number of patients."""
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        """
        Get patient data.
        
        Args:
            idx: Index of the patient
            
        Returns:
            Patient data and labels
        """
        patient_id = self.patient_ids[idx]
        patient_data = self.patient_groups.get_group(patient_id)
        
        # Sort by visit date
        patient_data = patient_data.sort_values("visit_date")
        
        # Get visits
        visits = []
        
        for _, visit_data in patient_data.iterrows():
            # Get codes and labels
            codes = eval(visit_data["codes"])  # Assumes codes are stored as a string representation of a list
            
            # Convert codes to indices
            code_indices = []
            
            for code in codes:
                code_id = code["code_id"]
                
                # Get index for the code
                code_idx = self.code_to_idx.get(code_id, 0)  # Use 0 for unknown codes
                code_indices.append(code_idx)
            
            # Create visit representation
            visit = {
                "visit_id": visit_data["visit_id"],
                "date": visit_data["visit_date"],
                "code_indices": torch.tensor(code_indices, dtype=torch.long)
            }
            
            visits.append(visit)
        
        # Truncate or pad to max_seq_length
        if len(visits) > self.max_seq_length:
            visits = visits[-self.max_seq_length:]  # Keep most recent visits
        
        # Create attention mask
        attention_mask = torch.ones(self.max_seq_length)
        attention_mask[len(visits):] = 0
        
        # Get labels based on the task
        if "mortality_label" in patient_data.columns:
            label = patient_data["mortality_label"].iloc[-1]
        elif "readmission_label" in patient_data.columns:
            label = patient_data["readmission_label"].iloc[-1]
        elif "los_label" in patient_data.columns:
            label = patient_data["los_label"].iloc[-1]
        else:
            # For multi-label tasks, get all label columns
            label_columns = [col for col in patient_data.columns if col.startswith("label_")]
            label = patient_data[label_columns].iloc[-1].values
        
        return {
            "patient_id": patient_id,
            "visits": visits,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.float if isinstance(label, (int, float)) else torch.long)
        }


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
    
    logger.info("MEDTOK model loaded successfully")
    
    return model


def collate_fn(batch):
    """
    Custom collate function for batching patient data.
    
    Args:
        batch: List of patient data samples
    
    Returns:
        Batched data
    """
    # Extract components
    patient_ids = [item["patient_id"] for item in batch]
    visits_list = [item["visits"] for item in batch]
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    
    # Handle labels based on shape
    labels = [item["label"] for item in batch]
    if isinstance(labels[0], torch.Tensor) and labels[0].dim() == 0:
        # Binary or multiclass classification
        labels = torch.stack(labels)
    elif isinstance(labels[0], torch.Tensor) and labels[0].dim() > 0:
        # Multilabel classification
        labels = torch.stack(labels)
    
    return {
        "patient_ids": patient_ids,
        "visits": visits_list,
        "attention_mask": attention_masks,
        "labels": labels
    }


def train_epoch(model, dataloader, optimizer, scheduler, config, device):
    """
    Train the model for one epoch.
    
    Args:
        model: GT-BEHRT model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration object
        device: Device to use
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    
    total_loss = 0.0
    
    # Define loss function based on task type
    if config.task_type == "binary_classification":
        criterion = nn.BCEWithLogitsLoss()
    elif config.task_type == "multiclass_classification":
        criterion = nn.CrossEntropyLoss()
    elif config.task_type == "multilabel_classification":
        criterion = nn.BCEWithLogitsLoss()
    
    progress_bar = tqdm(dataloader, desc=f"Training")
    
    for batch in progress_bar:
        # Move tensors to device
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model({"visits": batch["visits"]}, attention_mask)
        
        # Calculate loss
        if config.task_type == "binary_classification":
            loss = criterion(outputs.view(-1), labels)
        elif config.task_type == "multiclass_classification":
            loss = criterion(outputs, labels)
        elif config.task_type == "multilabel_classification":
            loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss


def evaluate(model, dataloader, config, device):
    """
    Evaluate the model.
    
    Args:
        model: GT-BEHRT model
        dataloader: Evaluation dataloader
        config: Configuration object
        device: Device to use
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move tensors to device
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model({"visits": batch["visits"]}, attention_mask)
            
            # Store predictions and labels
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())
    
    # Concatenate predictions and labels
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Calculate metrics based on task type
    metrics = {}
    
    if config.task_type == "binary_classification":
        # Binary classification
        auroc = roc_auc_score(all_labels, all_predictions)
        auprc = average_precision_score(all_labels, all_predictions)
        
        # Calculate F1 score at best threshold
        precision, recall, thresholds = precision_recall_curve(all_labels, all_predictions)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
        predictions_binary = (all_predictions >= best_threshold).astype(int)
        f1 = f1_score(all_labels, predictions_binary)
        
        metrics.update({
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
            "best_threshold": best_threshold
        })
    
    elif config.task_type == "multiclass_classification":
        # Multiclass classification
        predictions_classes = np.argmax(all_predictions, axis=1)
        accuracy = np.mean(predictions_classes == all_labels)
        
        metrics.update({
            "accuracy": accuracy
        })
    
    elif config.task_type == "multilabel_classification":
        # Multilabel classification
        auroc = roc_auc_score(all_labels, all_predictions, average="macro")
        auprc = average_precision_score(all_labels, all_predictions, average="macro")
        
        # Calculate F1 score at best threshold
        f1_scores = []
        
        for i in range(all_predictions.shape[1]):
            precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_predictions[:, i])
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            best_threshold_idx = np.argmax(f1)
            best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
            
            predictions_binary = (all_predictions[:, i] >= best_threshold).astype(int)
            f1_score_i = f1_score(all_labels[:, i], predictions_binary)
            f1_scores.append(f1_score_i)
        
        metrics.update({
            "macro_auroc": auroc,
            "macro_auprc": auprc,
            "macro_f1": np.mean(f1_scores)
        })
    
    return metrics


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
    np.random.seed(args.seed)
    
    # Load MEDTOK model
    medtok_model = load_medtok_model(args.medtok_model, device)
    
    if medtok_model is None:
        logger.error("Failed to load MEDTOK model")
        return
    
    # Determine graph directory
    graph_dir = args.graph_dir
    if graph_dir is None:
        graph_dir = os.path.join(args.mimic_dir, "graphs")
        if not os.path.exists(graph_dir):
            logger.error(f"Graph directory not found: {graph_dir}")
            logger.error("Please specify --graph_dir")
            return
    
    # Set task configuration
    task_config = {
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "task": args.task
    }
    
    # Determine task type and additional configuration
    if args.task == "mortality" or args.task == "readmission":
        task_config["task_type"] = "binary_classification"
    elif args.task == "los":
        task_config["task_type"] = "multiclass_classification"
        task_config["num_classes"] = 10  # As defined in the paper
    elif args.task == "phenotype" or args.task == "drugrec":
        task_config["task_type"] = "multilabel_classification"
        
        # Determine number of labels based on the task
        if args.task == "phenotype":
            task_config["num_labels"] = 25  # Number of phenotypes as defined in the paper
        elif args.task == "drugrec":
            task_config["num_labels"] = 5  # Number of drug candidates as defined in the paper
    
    # Create GT-BEHRT model with MEDTOK
    model = GTBEHRT(medtok_model, SimpleNamespace(**task_config)).to(device)
    
    # Create dataloaders
    train_dataset = EHRDataset(
        os.path.join(args.mimic_dir, f"{args.task}/{args.task}_data.csv"),
        medtok_model,
        graph_dir,
        max_seq_length=args.max_seq_length,
        split="train"
    )
    
    val_dataset = EHRDataset(
        os.path.join(args.mimic_dir, f"{args.task}/{args.task}_data.csv"),
        medtok_model,
        graph_dir,
        max_seq_length=args.max_seq_length,
        split="val"
    )
    
    test_dataset = EHRDataset(
        os.path.join(args.mimic_dir, f"{args.task}/{args.task}_data.csv"),
        medtok_model,
        graph_dir,
        max_seq_length=args.max_seq_length,
        split="test"
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Calculate total steps for scheduler
    total_steps = len(train_dataloader) * args.epochs
    
    # Create learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Train and evaluate
    logger.info("Starting training...")
    
    best_val_metric = 0.0
    best_model = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, SimpleNamespace(**task_config), device)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, SimpleNamespace(**task_config), device)
        
        # Determine validation metric based on task
        if task_config["task_type"] == "binary_classification":
            val_metric = val_metrics["auprc"]
            val_metric_name = "AUPRC"
        elif task_config["task_type"] == "multiclass_classification":
            val_metric = val_metrics["accuracy"]
            val_metric_name = "Accuracy"
        elif task_config["task_type"] == "multilabel_classification":
            val_metric = val_metrics["macro_auprc"]
            val_metric_name = "Macro AUPRC"
        
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val {val_metric_name} = {val_metric:.4f}")
        
        # Save best model
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_model = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics
            }
            
            logger.info(f"New best model with {val_metric_name} = {val_metric:.4f}")
    
    # Save best model
    if best_model:
        best_model_path = os.path.join(args.output_dir, f"gt_behrt_medtok_{args.task}_best.pt")
        torch.save(best_model, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
    
    # Load best model for final evaluation
    if best_model:
        model.load_state_dict(best_model["model_state_dict"])
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_dataloader, SimpleNamespace(**task_config), device)
    
    # Determine test metric based on task
    if task_config["task_type"] == "binary_classification":
        test_metric = test_metrics["auprc"]
        test_metric_name = "AUPRC"
    elif task_config["task_type"] == "multiclass_classification":
        test_metric = test_metrics["accuracy"]
        test_metric_name = "Accuracy"
    elif task_config["task_type"] == "multilabel_classification":
        test_metric = test_metrics["macro_auprc"]
        test_metric_name = "Macro AUPRC"
    
    logger.info(f"Test {test_metric_name} = {test_metric:.4f}")
    
    # Save test metrics
    test_metrics_path = os.path.join(args.output_dir, f"gt_behrt_medtok_{args.task}_test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"Test metrics saved to {test_metrics_path}")
    
    # Compare with baseline if requested
    if args.compare_baseline:
        logger.info("Comparing with baseline GT-BEHRT (without MEDTOK)...")
        
        # Create code-to-index mapping
        code_data = pd.read_csv(os.path.join(args.mimic_dir, "medical_codes.csv"))
        codes = code_data["code"].unique()
        code_to_idx = {code: idx + 1 for idx, code in enumerate(codes)}  # Start from 1, reserve 0 for padding
        
        # Create baseline model
        baseline_model = BaselineGTBEHRT(len(code_to_idx) + 1, SimpleNamespace(**task_config)).to(device)
        
        # Create baseline datasets
        baseline_train_dataset = BaselineEHRDataset(
            os.path.join(args.mimic_dir, f"{args.task}/{args.task}_data.csv"),
            code_to_idx,
            max_seq_length=args.max_seq_length,
            split="train"
        )
        
        baseline_val_dataset = BaselineEHRDataset(
            os.path.join(args.mimic_dir, f"{args.task}/{args.task}_data.csv"),
            code_to_idx,
            max_seq_length=args.max_seq_length,
            split="val"
        )
        
        baseline_test_dataset = BaselineEHRDataset(
            os.path.join(args.mimic_dir, f"{args.task}/{args.task}_data.csv"),
            code_to_idx,
            max_seq_length=args.max_seq_length,
            split="test"
        )
        
        # Create baseline dataloaders
        baseline_train_dataloader = DataLoader(
            baseline_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        baseline_val_dataloader = DataLoader(
            baseline_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        baseline_test_dataloader = DataLoader(
            baseline_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        # Create optimizer and scheduler for baseline model
        baseline_optimizer = optim.AdamW(
            baseline_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Calculate total steps for scheduler
        baseline_total_steps = len(baseline_train_dataloader) * args.epochs
        
        # Create learning rate scheduler with warmup
        baseline_scheduler = get_linear_schedule_with_warmup(
            baseline_optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=baseline_total_steps
        )
        
        # Train and evaluate baseline model
        logger.info("Starting baseline training...")
        
        baseline_best_val_metric = 0.0
        baseline_best_model = None
        
        for epoch in range(1, args.epochs + 1):
            # Train
            baseline_train_loss = train_epoch(
                baseline_model,
                baseline_train_dataloader,
                baseline_optimizer,
                baseline_scheduler,
                SimpleNamespace(**task_config),
                device
            )
            
            # Evaluate on validation set
            baseline_val_metrics = evaluate(
                baseline_model,
                baseline_val_dataloader,
                SimpleNamespace(**task_config),
                device
            )
            
            # Determine validation metric based on task
            if task_config["task_type"] == "binary_classification":
                baseline_val_metric = baseline_val_metrics["auprc"]
                val_metric_name = "AUPRC"
            elif task_config["task_type"] == "multiclass_classification":
                baseline_val_metric = baseline_val_metrics["accuracy"]
                val_metric_name = "Accuracy"
            elif task_config["task_type"] == "multilabel_classification":
                baseline_val_metric = baseline_val_metrics["macro_auprc"]
                val_metric_name = "Macro AUPRC"
            
            logger.info(f"Baseline Epoch {epoch}: Train Loss = {baseline_train_loss:.4f}, Val {val_metric_name} = {baseline_val_metric:.4f}")
            
            # Save best model
            if baseline_val_metric > baseline_best_val_metric:
                baseline_best_val_metric = baseline_val_metric
                baseline_best_model = {
                    "epoch": epoch,
                    "model_state_dict": baseline_model.state_dict(),
                    "optimizer_state_dict": baseline_optimizer.state_dict(),
                    "val_metrics": baseline_val_metrics
                }
                
                logger.info(f"New best baseline model with {val_metric_name} = {baseline_val_metric:.4f}")
        
        # Save best baseline model
        if baseline_best_model:
            baseline_best_model_path = os.path.join(args.output_dir, f"gt_behrt_baseline_{args.task}_best.pt")
            torch.save(baseline_best_model, baseline_best_model_path)
            logger.info(f"Best baseline model saved to {baseline_best_model_path}")
        
        # Load best baseline model for final evaluation
        if baseline_best_model:
            baseline_model.load_state_dict(baseline_best_model["model_state_dict"])
        
        # Evaluate baseline on test set
        baseline_test_metrics = evaluate(
            baseline_model,
            baseline_test_dataloader,
            SimpleNamespace(**task_config),
            device
        )
        
        # Determine test metric based on task
        if task_config["task_type"] == "binary_classification":
            baseline_test_metric = baseline_test_metrics["auprc"]
            test_metric_name = "AUPRC"
        elif task_config["task_type"] == "multiclass_classification":
            baseline_test_metric = baseline_test_metrics["accuracy"]
            test_metric_name = "Accuracy"
        elif task_config["task_type"] == "multilabel_classification":
            baseline_test_metric = baseline_test_metrics["macro_auprc"]
            test_metric_name = "Macro AUPRC"
        
        logger.info(f"Baseline Test {test_metric_name} = {baseline_test_metric:.4f}")
        
        # Save baseline test metrics
        baseline_test_metrics_path = os.path.join(args.output_dir, f"gt_behrt_baseline_{args.task}_test_metrics.json")
        with open(baseline_test_metrics_path, "w") as f:
            json.dump(baseline_test_metrics, f, indent=2)
        
        logger.info(f"Baseline test metrics saved to {baseline_test_metrics_path}")
        
        # Compare results
        improvement = test_metric - baseline_test_metric
        relative_improvement = improvement / baseline_test_metric * 100 if baseline_test_metric > 0 else 0
        
        logger.info(f"Comparison:")
        logger.info(f"  GT-BEHRT with MEDTOK: {test_metric:.4f}")
        logger.info(f"  Baseline GT-BEHRT: {baseline_test_metric:.4f}")
        logger.info(f"  Absolute improvement: {improvement:.4f}")
        logger.info(f"  Relative improvement: {relative_improvement:.2f}%")
        
        # Save comparison
        comparison = {
            "task": args.task,
            "metric": test_metric_name,
            "gt_behrt_medtok": test_metric,
            "gt_behrt_baseline": baseline_test_metric,
            "absolute_improvement": improvement,
            "relative_improvement": relative_improvement
        }
        
        comparison_path = os.path.join(args.output_dir, f"gt_behrt_{args.task}_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison saved to {comparison_path}")


if __name__ == "__main__":
    # Import additional modules needed in main
    from types import SimpleNamespace
    import torch.nn.functional as F
    
    main()
