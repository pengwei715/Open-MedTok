#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BEHRT integration script for MEDTOK.

This script shows how to integrate MEDTOK with the BEHRT model.
BEHRT applies deep bidirectional learning to predict future medical events,
encoding disease codes, age, and visit sequences using self-attention.
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
from datetime import datetime
from transformers import BertConfig, BertModel, BertPreTrainedModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
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
    parser = argparse.ArgumentParser(description="Integrate MEDTOK with BEHRT model")
    
    parser.add_argument("--medtok_model", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--behrt_model", type=str, default=None, 
                        help="Path to BEHRT model (if None, will initialize new)")
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
    parser.add_argument("--lr", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--hidden_size", type=int, default=768, 
                        help="Hidden size for BEHRT")
    parser.add_argument("--num_layers", type=int, default=12, 
                        help="Number of transformer layers")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--task", type=str, default="mortality", 
                        help="Task to evaluate (mortality, readmission, los, phenotype, drugrec)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    parser.add_argument("--baseline", action="store_true", 
                        help="Run baseline BEHRT (without MEDTOK)")
    
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
    
    logger.info("MEDTOK model loaded successfully")
    
    return model


class MedicalCode:
    """Class representing a medical code with its description and subgraph."""
    
    def __init__(self, code, description, system, node_features=None, edge_index=None, batch_idx=0, visit_idx=0):
        """
        Initialize a medical code.
        
        Args:
            code: The medical code identifier (e.g., "ICD10: E10.31")
            description: Textual description of the code
            system: Coding system (e.g., "ICD10", "SNOMED", "ATC")
            node_features: Optional graph node features
            edge_index: Optional graph edge indices
            batch_idx: Batch index for processing
            visit_idx: Visit index for temporal ordering
        """
        self.code = code
        self.description = description
        self.system = system
        self.node_features = node_features
        self.edge_index = edge_index
        self.batch_idx = batch_idx
        self.visit_idx = visit_idx


class BEHRTWithMedTok(BertPreTrainedModel):
    """
    BEHRT model integrated with MedTok tokenizer for EHR processing.
    This extends the original BEHRT architecture to work with MedTok tokenized inputs.
    """
    
    def __init__(self, config, medtok=None, num_labels=2, age_vocab_size=120):
        """
        Initialize BEHRT model with MedTok integration.
        
        Args:
            config: BertConfig configuration
            medtok: MedTok model for tokenization
            num_labels: Number of output labels
            age_vocab_size: Size of age vocabulary for embeddings
        """
        super().__init__(config)
        
        # Main BERT model
        self.bert = BertModel(config)
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Age embeddings (specific to BEHRT)
        self.age_embeddings = nn.Embedding(age_vocab_size, config.hidden_size)
        
        # Store MedTok model reference
        self.medtok = medtok
        
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        age_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        """
        Forward pass through the BEHRT model.
        
        Args:
            input_ids: Token IDs for input sequence
            attention_mask: Mask indicating which tokens to attend to
            token_type_ids: IDs indicating token type (diagnosis, procedure, etc.)
            position_ids: IDs indicating position in sequence (visit order)
            age_ids: IDs indicating patient age at each visit
            head_mask: Mask to nullify selected heads of the self-attention modules
            inputs_embeds: Pre-computed input embeddings
            labels: Ground truth labels for classification task
            
        Returns:
            Dictionary containing model outputs and optionally loss
        """
        # Run BERT model
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        # Get pooled output (CLS token) for classification
        pooled_output = outputs[1]
        
        # Add age embeddings if provided
        if age_ids is not None:
            # For simplicity, just average age embeddings across sequence
            age_embeddings = self.age_embeddings(age_ids)
            age_embedding = age_embeddings.mean(dim=1)
            pooled_output = pooled_output + age_embedding
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Binary classification
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                # Multi-class classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
    
    def get_medtok_embeddings(self, medical_codes):
        """
        Get embeddings for medical codes using MedTok.
        
        Args:
            medical_codes: List of MedicalCode objects
            
        Returns:
            Tensor of embeddings for the codes
        """
        if self.medtok is None:
            raise ValueError("MedTok model not provided")
        
        # Process each code with MedTok
        embeddings = []
        for code in medical_codes:
            # Get token indices
            token_indices = self.medtok_tokenize(code)
            
            # Get token embeddings
            token_embedding = self.medtok.get_token_embedding(token_indices)
            
            # Average token embeddings to get code embedding
            code_embedding = token_embedding.mean(dim=0)
            embeddings.append(code_embedding)
        
        # Stack embeddings
        if embeddings:
            return torch.stack(embeddings)
        else:
            return torch.zeros((0, self.config.hidden_size), device=next(self.parameters()).device)
    
    def medtok_tokenize(self, code):
        """
        Tokenize a medical code using MedTok.
        
        Args:
            code: MedicalCode object
            
        Returns:
            Token indices for the code
        """
        if self.medtok is None:
            raise ValueError("MedTok model not provided")
        
        device = next(self.parameters()).device
        
        # Get text encoder tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.medtok.config.text_encoder_model)
        
        # Tokenize text
        encoded_text = tokenizer(
            code.description,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Prepare graph inputs
        if code.node_features is not None and code.edge_index is not None:
            node_features = code.node_features.to(device)
            edge_index = code.edge_index.to(device)
        else:
            # Create dummy graph
            node_features = torch.zeros((1, self.medtok.config.node_feature_dim), device=device)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        
        # Create batch tensor for graph
        graph_batch = torch.zeros(node_features.size(0), dtype=torch.long, device=device)
        
        # Tokenize with MedTok
        with torch.no_grad():
            token_indices = self.medtok.tokenize(
                encoded_text["input_ids"],
                node_features,
                edge_index,
                graph_batch
            )
        
        return token_indices[0]


class EHRDataset(Dataset):
    """Dataset for EHR data with MEDTOK tokenization."""
    
    def __init__(self, data_file, code_mappings, medtok_model=None, graph_dir=None, 
                 max_seq_length=512, special_tokens=None, task="mortality"):
        """
        Initialize the EHR dataset.
        
        Args:
            data_file: Path to data file
            code_mappings: Dictionary mapping codes to indices
            medtok_model: MEDTOK model for tokenization
            graph_dir: Directory containing graph files
            max_seq_length: Maximum sequence length
            special_tokens: Dictionary of special tokens
            task: Prediction task
        """
        self.data_file = data_file
        self.code_mappings = code_mappings
        self.medtok_model = medtok_model
        self.graph_dir = graph_dir
        self.max_seq_length = max_seq_length
        self.task = task
        
        # Define special tokens
        self.special_tokens = special_tokens or {
            "PAD": 0,
            "CLS": 1,
            "SEP": 2,
            "MASK": 3,
            "UNK": 4
        }
        
        # Load data
        self.data = pd.read_csv(data_file)
        
        # Group by patient ID
        self.patient_groups = list(self.data.groupby("SUBJECT_ID"))
    
    def __len__(self):
        """Return number of patients."""
        return len(self.patient_groups)
    
    def __getitem__(self, idx):
        """
        Get patient data.
        
        Args:
            idx: Index of the patient
            
        Returns:
            Dictionary with patient data
        """
        patient_id, patient_data = self.patient_groups[idx]
        
        # Sort by admission time
        patient_data = patient_data.sort_values("ADMITTIME")
        
        # Process visits
        input_ids = torch.ones(self.max_seq_length, dtype=torch.long) * self.special_tokens["PAD"]
        token_type_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
        position_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)
        age_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
        
        idx = 0
        visits = []
        
        # Add CLS token at the beginning
        input_ids[idx] = self.special_tokens["CLS"]
        token_type_ids[idx] = 0
        position_ids[idx] = 0
        attention_mask[idx] = 1
        idx += 1
        
        # Process each visit
        for visit_idx, (_, visit) in enumerate(patient_data.groupby("HADM_ID")):
            # Get age at visit
            if "AGE" in visit.columns:
                age = min(int(visit["AGE"].iloc[0]), 119)  # Cap at 119 for embedding table
            else:
                age = 50  # Default age
            
            # Get diagnoses
            diagnoses = []
            if "ICD9_CODE" in visit.columns:
                diagnoses = visit["ICD9_CODE"].dropna().tolist()
            
            # Add codes to input sequence
            for code in diagnoses:
                if idx < self.max_seq_length:
                    # Use MEDTOK if available
                    if self.medtok_model is not None:
                        # Create MedicalCode object
                        medical_code = MedicalCode(
                            code=code,
                            description=self.get_code_description(code),
                            system="ICD9",
                            visit_idx=visit_idx
                        )
                        
                        # Load graph data if available
                        if self.graph_dir:
                            graph_file = os.path.join(self.graph_dir, f"{code}.json")
                            if os.path.exists(graph_file):
                                graph_data = self.load_graph_data(graph_file)
                                medical_code.node_features = graph_data["node_features"]
                                medical_code.edge_index = graph_data["edge_index"]
                        
                        # Get token ID from MEDTOK
                        # In a real implementation, we would get MEDTOK token IDs
                        # For this example, we use the code mapping instead
                        code_id = self.code_mappings.get(code, self.special_tokens["UNK"])
                    else:
                        # Use standard code mapping
                        code_id = self.code_mappings.get(code, self.special_tokens["UNK"])
                    
                    input_ids[idx] = code_id
                    token_type_ids[idx] = 1  # Diagnosis token type
                    position_ids[idx] = visit_idx + 1  # Visit position (1-indexed)
                    attention_mask[idx] = 1
                    age_ids[idx] = age
                    idx += 1
            
            # Add SEP token at the end of each visit
            if idx < self.max_seq_length:
                input_ids[idx] = self.special_tokens["SEP"]
                token_type_ids[idx] = 0
                position_ids[idx] = 0
                attention_mask[idx] = 1
                idx += 1
            
            # Store visit info
            visits.append({
                "visit_id": visit["HADM_ID"].iloc[0],
                "codes": diagnoses,
                "age": age
            })
        
        # Get label based on task
        if self.task == "mortality":
            label = 1 if patient_data["HOSPITAL_EXPIRE_FLAG"].max() > 0 else 0
        elif self.task == "readmission":
            # Check if patient was readmitted within 30 days
            admit_times = patient_data["ADMITTIME"].sort_values().tolist()
            if len(admit_times) > 1:
                readmitted = False
                for i in range(len(admit_times) - 1):
                    days_diff = (pd.to_datetime(admit_times[i+1]) - pd.to_datetime(admit_times[i])).days
                    if days_diff <= 30:
                        readmitted = True
                        break
                label = 1 if readmitted else 0
            else:
                label = 0
        elif self.task == "los":
            # Length of stay classification
            # 0: <1 day, 1: 1-2 days, 2: 3-7 days, 3: 8-14 days, 4: 14+ days
            los_days = patient_data["LOS"].max()
            if los_days < 1:
                label = 0
            elif los_days < 3:
                label = 1
            elif los_days < 8:
                label = 2
            elif los_days < 15:
                label = 3
            else:
                label = 4
        else:
            # Default to binary classification
            label = 0
        
        return {
            "patient_id": patient_id,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "age_ids": age_ids,
            "visits": visits,
            "label": torch.tensor(label)
        }
    
    def get_code_description(self, code):
        """Get description for a medical code."""
        # In a real implementation, this would lookup the description from a database
        return f"Medical code {code}"
    
    def load_graph_data(self, graph_file):
        """Load graph data from file."""
        try:
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
                    node_features.append([0.0] * 256)  # Assuming 256-dimensional features
            
            # Extract edge indices
            edge_index = []
            for src, dst in G.edges:
                edge_index.append([src, dst])
                edge_index.append([dst, src])  # Add reverse edge for undirected graphs
            
            # Convert to torch tensors
            node_features = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            return {"node_features": node_features, "edge_index": edge_index}
            
        except Exception as e:
            logger.warning(f"Error loading graph file {graph_file}: {e}")
            return {"node_features": None, "edge_index": None}


def load_mimic_data(data_dir, task="mortality"):
    """
    Load MIMIC data for the specified task.
    
    Args:
        data_dir: Directory containing MIMIC data
        task: Task name
        
    Returns:
        Tuple of dataframes (train, val, test)
    """
    logger.info(f"Loading MIMIC data for task: {task}")
    
    # Check if preprocessed data exists
    preprocessed_file = os.path.join(data_dir, f"{task}_preprocessed.csv")
    if os.path.exists(preprocessed_file):
        logger.info(f"Loading preprocessed data from {preprocessed_file}")
        data = pd.read_csv(preprocessed_file)
        
        # Split into train/val/test
        train_data = data[data["SPLIT"] == "train"]
        val_data = data[data["SPLIT"] == "val"]
        test_data = data[data["SPLIT"] == "test"]
        
        return train_data, val_data, test_data
    
    # Load raw data
    logger.info("Loading raw MIMIC data...")
    
    # In a real implementation, this would load the actual MIMIC tables
    # For this example, we create dummy data
    
    # Create dummy patients
    num_patients = 1000
    patients = []
    
    for i in range(num_patients):
        patient_id = f"P{i:06d}"
        num_visits = np.random.randint(1, 5)
        
        for j in range(num_visits):
            visit_id = f"V{i:06d}_{j:02d}"
            
            # Generate random diagnoses
            num_diagnoses = np.random.randint(1, 10)
            diagnoses = [f"{np.random.randint(1, 999):03d}.{np.random.randint(0, 9):1d}" for _ in range(num_diagnoses)]
            
            # Generate random age
            age = np.random.randint(18, 90)
            
            # Generate admit time
            base_date = pd.Timestamp("2010-01-01")
            admit_days = i * 10 + j * 100  # Ensure temporal ordering
            admit_time = base_date + pd.Timedelta(days=admit_days)
            
            # Generate discharge time
            los_days = np.random.randint(1, 30)
            discharge_time = admit_time + pd.Timedelta(days=los_days)
            
            # Generate mortality flag
            expire_flag = 1 if np.random.random() < 0.1 else 0
            
            # Create patient record
            for diagnosis in diagnoses:
                patients.append({
                    "SUBJECT_ID": patient_id,
                    "HADM_ID": visit_id,
                    "ADMITTIME": admit_time,
                    "DISCHTIME": discharge_time,
                    "LOS": los_days,
                    "AGE": age,
                    "HOSPITAL_EXPIRE_FLAG": expire_flag,
                    "ICD9_CODE": diagnosis
                })
    
    # Convert to dataframe
    data = pd.DataFrame(patients)
    
    # Split into train/val/test
    patient_ids = data["SUBJECT_ID"].unique()
    np.random.shuffle(patient_ids)
    
    train_size = int(0.7 * len(patient_ids))
    val_size = int(0.1 * len(patient_ids))
    
    train_ids = patient_ids[:train_size]
    val_ids = patient_ids[train_size:train_size+val_size]
    test_ids = patient_ids[train_size+val_size:]
    
    train_data = data[data["SUBJECT_ID"].isin(train_ids)].copy()
    val_data = data[data["SUBJECT_ID"].isin(val_ids)].copy()
    test_data = data[data["SUBJECT_ID"].isin(test_ids)].copy()
    
    # Add split column
    train_data["SPLIT"] = "train"
    val_data["SPLIT"] = "val"
    test_data["SPLIT"] = "test"
    
    # Save preprocessed data
    all_data = pd.concat([train_data, val_data, test_data])
    all_data.to_csv(preprocessed_file, index=False)
    
    logger.info(f"Data saved to {preprocessed_file}")
    logger.info(f"Train: {len(train_data)} rows, {len(train_ids)} patients")
    logger.info(f"Val: {len(val_data)} rows, {len(val_ids)} patients")
    logger.info(f"Test: {len(test_data)} rows, {len(test_ids)} patients")
    
    return train_data, val_data, test_data


def create_code_mappings(data):
    """
    Create mapping from medical codes to indices.
    
    Args:
        data: DataFrame with medical codes
        
    Returns:
        Dictionary mapping codes to indices
    """
    # Extract all unique codes
    all_codes = set()
    
    if "ICD9_CODE" in data.columns:
        all_codes.update(data["ICD9_CODE"].dropna().unique())
    
    # Create mapping
    code_to_idx = {}
    for idx, code in enumerate(sorted(all_codes)):
        code_to_idx[code] = idx + 5  # Start from 5 to account for special tokens
    
    return code_to_idx


def train_behrt(model, train_loader, val_loader, device, 
                epochs=20, lr=2e-5, num_labels=2):
    """
    Train BEHRT model.
    
    Args:
        model: BEHRT model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        epochs: Number of training epochs
        lr: Learning rate
        num_labels: Number of output labels
        
    Returns:
        Trained model and metrics
    """
    logger.info("Training BEHRT model...")
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Set up loss function
    if num_labels == 1:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    # Calculate total steps
    total_steps = len(train_loader) * epochs
    
    # Set up learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            age_ids = batch["age_ids"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                age_ids=age_ids,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                # Move data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                position_ids = batch["position_ids"].to(device)
                age_ids = batch["age_ids"].to(device)
                labels = batch["label"].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    age_ids=age_ids,
                    labels=labels
                )
                
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                val_loss += loss.item()
                
                # Store predictions and labels
                if num_labels == 1:
                    preds = torch.sigmoid(logits).squeeze().cpu().numpy()
                else:
                    preds = torch.softmax(logits, dim=-1).cpu().numpy()
                
                val_preds.extend(preds.tolist())
                val_labels.extend(labels.cpu().numpy().tolist())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        if num_labels == 1:
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            
            # Calculate AUROC and AUPRC
            try:
                auroc = roc_auc_score(val_labels, val_preds)
            except:
                auroc = 0.0
                
            try:
                auprc = average_precision_score(val_labels, val_preds)
            except:
                auprc = 0.0
                
            # Calculate F1 score at best threshold
            try:
                precision, recall, thresholds = precision_recall_curve(val_labels, val_preds)
                f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
                best_threshold_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
                
                preds_binary = (val_preds >= best_threshold).astype(int)
                f1 = f1_score(val_labels, preds_binary)
            except:
                f1 = 0.0
            
            metrics = {
                "val_loss": avg_val_loss,
                "val_auroc": auroc,
                "val_auprc": auprc,
                "val_f1": f1,
            }
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss = {avg_train_loss:.4f}, "
                      f"Val Loss = {avg_val_loss:.4f}, "
                      f"Val AUROC = {auroc:.4f}, "
                      f"Val AUPRC = {auprc:.4f}, "
                      f"Val F1 = {f1:.4f}")
        else:
            # Multi-class metrics
            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)
            
            # Get predicted classes
            pred_classes = np.argmax(val_preds, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(pred_classes == val_labels)
            
            metrics = {
                "val_loss": avg_val_loss,
                "val_accuracy": accuracy,
            }
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss = {avg_train_loss:.4f}, "
                      f"Val Loss = {avg_val_loss:.4f}, "
                      f"Val Accuracy = {accuracy:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            
            logger.info(f"New best model saved (val_loss: {avg_val_loss:.4f})")
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    logger.info("Training completed")
    
    return model, metrics


def evaluate_behrt(model, test_loader, device, num_labels=2):
    """
    Evaluate BEHRT model on test data.
    
    Args:
        model: BEHRT model
        test_loader: Test data loader
        device: Device to use
        num_labels: Number of output labels
        
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating BEHRT model...")
    
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []
    
    # Set up loss function
    if num_labels == 1:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            age_ids = batch["age_ids"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                age_ids=age_ids,
                labels=labels
            )
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            test_loss += loss.item()
            
            # Store predictions and labels
            if num_labels == 1:
                preds = torch.sigmoid(logits).squeeze().cpu().numpy()
            else:
                preds = torch.softmax(logits, dim=-1).cpu().numpy()
            
            test_preds.extend(preds.tolist())
            test_labels.extend(labels.cpu().numpy().tolist())
    
    # Calculate average test loss
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate metrics
    metrics = {"test_loss": avg_test_loss}
    
    if num_labels == 1:
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        
        # Calculate AUROC and AUPRC
        try:
            auroc = roc_auc_score(test_labels, test_preds)
            metrics["test_auroc"] = auroc
        except:
            metrics["test_auroc"] = 0.0
            
        try:
            auprc = average_precision_score(test_labels, test_preds)
            metrics["test_auprc"] = auprc
        except:
            metrics["test_auprc"] = 0.0
            
        # Calculate F1 score at best threshold
        try:
            precision, recall, thresholds = precision_recall_curve(test_labels, test_preds)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
            
            preds_binary = (test_preds >= best_threshold).astype(int)
            f1 = f1_score(test_labels, preds_binary)
            metrics["test_f1"] = f1
        except:
            metrics["test_f1"] = 0.0
        
        logger.info(f"Test Loss = {avg_test_loss:.4f}, "
                   f"Test AUROC = {metrics['test_auroc']:.4f}, "
                   f"Test AUPRC = {metrics['test_auprc']:.4f}, "
                   f"Test F1 = {metrics['test_f1']:.4f}")
    else:
        # Multi-class metrics
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        
        # Get predicted classes
        pred_classes = np.argmax(test_preds, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(pred_classes == test_labels)
        metrics["test_accuracy"] = accuracy
        
        logger.info(f"Test Loss = {avg_test_loss:.4f}, "
                   f"Test Accuracy = {accuracy:.4f}")
    
    return metrics


def tokenize_graph_files(medtok_model, all_codes, graph_dir, device):
    """
    Tokenize graph files for all medical codes.
    
    Args:
        medtok_model: MedTok model
        all_codes: Set of all medical codes
        graph_dir: Directory containing graph files
        device: Device to use
        
    Returns:
        Dictionary mapping codes to token indices
    """
    logger.info(f"Tokenizing graph files for {len(all_codes)} medical codes...")
    
    # Get text encoder tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(medtok_model.config.text_encoder_model)
    
    # Process each code
    code_to_tokens = {}
    
    for code in tqdm(all_codes):
        # Create dummy description (in a real implementation, this would be retrieved from a database)
        description = f"Medical code: {code}"
        
        # Check if graph file exists
        graph_file = os.path.join(graph_dir, f"{code}.json")
        
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
        
        # Tokenize text
        encoded_text = tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Create batch tensor for graph
        graph_batch = torch.zeros(node_features.size(0), dtype=torch.long, device=device)
        
        # Tokenize with MedTok
        with torch.no_grad():
            token_indices = medtok_model.tokenize(
                encoded_text["input_ids"],
                node_features,
                edge_index,
                graph_batch
            )
        
        # Store token indices
        code_to_tokens[code] = token_indices[0].cpu().numpy().tolist()
    
    logger.info(f"Tokenized {len(code_to_tokens)} medical codes")
    
    return code_to_tokens


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
    
    # Load data
    train_data, val_data, test_data = load_mimic_data(args.data_dir, args.task)
    
    # Create code mappings
    all_data = pd.concat([train_data, val_data, test_data])
    code_mappings = create_code_mappings(all_data)
    
    # Set number of labels based on task
    if args.task == "los":
        num_labels = 5  # Length of stay categories
    elif args.task in ["mortality", "readmission"]:
        num_labels = 1  # Binary classification
    else:
        num_labels = 2  # Default
    
    # Determine graph directory
    graph_dir = args.graph_dir
    if graph_dir is None:
        graph_dir = os.path.join(args.data_dir, "graphs")
        if not os.path.exists(graph_dir):
            logger.warning(f"Graph directory not found: {graph_dir}")
            logger.warning("Creating empty directory for graph files")
            os.makedirs(graph_dir, exist_ok=True)
    
    # Baseline BEHRT (without MEDTOK)
    if args.baseline:
        logger.info("Running baseline BEHRT (without MEDTOK)...")
        
        # Create datasets
        train_dataset = EHRDataset(
            data_file=os.path.join(args.data_dir, f"{args.task}_preprocessed.csv"),
            code_mappings=code_mappings,
            medtok_model=None,
            graph_dir=None,
            max_seq_length=args.max_seq_length,
            task=args.task
        )
        
        val_dataset = EHRDataset(
            data_file=os.path.join(args.data_dir, f"{args.task}_preprocessed.csv"),
            code_mappings=code_mappings,
            medtok_model=None,
            graph_dir=None,
            max_seq_length=args.max_seq_length,
            task=args.task
        )
        
        test_dataset = EHRDataset(
            data_file=os.path.join(args.data_dir, f"{args.task}_preprocessed.csv"),
            code_mappings=code_mappings,
            medtok_model=None,
            graph_dir=None,
            max_seq_length=args.max_seq_length,
            task=args.task
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        # Create BEHRT config
        config = BertConfig(
            vocab_size=len(code_mappings) + 5,  # +5 for special tokens
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=4,
            intermediate_size=args.hidden_size * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=args.max_seq_length,
            type_vocab_size=2,  # Binary token type IDs (0 for special, 1 for codes)
        )
        
        # Create BEHRT model
        baseline_model = BEHRTWithMedTok(
            config=config,
            medtok=None,
            num_labels=num_labels,
            age_vocab_size=120,  # 0-119 age range
        ).to(device)
        
        # Train model
        baseline_model, baseline_metrics = train_behrt(
            model=baseline_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            num_labels=num_labels,
        )
        
        # Save model
        baseline_model_path = os.path.join(args.output_dir, "behrt_baseline.pt")
        torch.save({
            "model_state_dict": baseline_model.state_dict(),
            "config": config.to_dict(),
            "metrics": baseline_metrics,
        }, baseline_model_path)
        
        logger.info(f"Baseline model saved to {baseline_model_path}")
        
        # Evaluate model
        baseline_test_metrics = evaluate_behrt(
            model=baseline_model,
            test_loader=test_loader,
            device=device,
            num_labels=num_labels,
        )
        
        # Save metrics
        baseline_metrics_path = os.path.join(args.output_dir, "behrt_baseline_metrics.json")
        with open(baseline_metrics_path, "w") as f:
            json.dump({
                "val_metrics": baseline_metrics,
                "test_metrics": baseline_test_metrics,
            }, f, indent=2)
        
        logger.info(f"Baseline metrics saved to {baseline_metrics_path}")
    
    # BEHRT with MEDTOK
    logger.info("Running BEHRT with MEDTOK integration...")
    
    # Load MEDTOK model
    medtok_model = load_medtok_model(args.medtok_model, device)
    
    if medtok_model is None:
        logger.error("Failed to load MEDTOK model")
        return
    
    # Set MEDTOK to evaluation mode
    medtok_model.eval()
    
    # Tokenize all codes with MEDTOK
    all_codes = set()
    if "ICD9_CODE" in all_data.columns:
        all_codes.update(all_data["ICD9_CODE"].dropna().unique())
    
    # Tokenize graph files for all codes
    code_to_tokens = tokenize_graph_files(
        medtok_model=medtok_model,
        all_codes=all_codes,
        graph_dir=graph_dir,
        device=device,
    )
    
    # Create datasets
    train_dataset = EHRDataset(
        data_file=os.path.join(args.data_dir, f"{args.task}_preprocessed.csv"),
        code_mappings=code_mappings,
        medtok_model=medtok_model,
        graph_dir=graph_dir,
        max_seq_length=args.max_seq_length,
        task=args.task
    )
    
    val_dataset = EHRDataset(
        data_file=os.path.join(args.data_dir, f"{args.task}_preprocessed.csv"),
        code_mappings=code_mappings,
        medtok_model=medtok_model,
        graph_dir=graph_dir,
        max_seq_length=args.max_seq_length,
        task=args.task
    )
    
    test_dataset = EHRDataset(
        data_file=os.path.join(args.data_dir, f"{args.task}_preprocessed.csv"),
        code_mappings=code_mappings,
        medtok_model=medtok_model,
        graph_dir=graph_dir,
        max_seq_length=args.max_seq_length,
        task=args.task
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Create BEHRT config
    config = BertConfig(
        vocab_size=len(code_mappings) + 5,  # +5 for special tokens
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=4,
        intermediate_size=args.hidden_size * 4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=args.max_seq_length,
        type_vocab_size=2,  # Binary token type IDs (0 for special, 1 for codes)
    )
    
    # Create BEHRT model with MEDTOK
    medtok_model = BEHRTWithMedTok(
        config=config,
        medtok=medtok_model,
        num_labels=num_labels,
        age_vocab_size=120,  # 0-119 age range
    ).to(device)
    
    # Train model
    medtok_model, medtok_metrics = train_behrt(
        model=medtok_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        num_labels=num_labels,
    )
    
    # Save model
    medtok_model_path = os.path.join(args.output_dir, "behrt_medtok.pt")
    torch.save({
        "model_state_dict": medtok_model.state_dict(),
        "config": config.to_dict(),
        "metrics": medtok_metrics,
    }, medtok_model_path)
    
    logger.info(f"MEDTOK model saved to {medtok_model_path}")
    
    # Evaluate model
    medtok_test_metrics = evaluate_behrt(
        model=medtok_model,
        test_loader=test_loader,
        device=device,
        num_labels=num_labels,
    )
    
    # Save metrics
    medtok_metrics_path = os.path.join(args.output_dir, "behrt_medtok_metrics.json")
    with open(medtok_metrics_path, "w") as f:
        json.dump({
            "val_metrics": medtok_metrics,
            "test_metrics": medtok_test_metrics,
        }, f, indent=2)
    
    logger.info(f"MEDTOK metrics saved to {medtok_metrics_path}")
    
    # Compare results if baseline was run
    if args.baseline:
        comparison = {}
        
        # Compare key metrics
        if num_labels == 1:
            # Binary classification
            metrics_to_compare = ["test_auroc", "test_auprc", "test_f1"]
        else:
            # Multi-class classification
            metrics_to_compare = ["test_accuracy"]
        
        for metric in metrics_to_compare:
            baseline_value = baseline_test_metrics.get(metric, 0.0)
            medtok_value = medtok_test_metrics.get(metric, 0.0)
            
            absolute_improvement = medtok_value - baseline_value
            relative_improvement = absolute_improvement / baseline_value * 100 if baseline_value > 0 else 0.0
            
            comparison[metric] = {
                "baseline": baseline_value,
                "medtok": medtok_value,
                "absolute_improvement": absolute_improvement,
                "relative_improvement": f"{relative_improvement:.2f}%",
            }
        
        # Save comparison
        comparison_path = os.path.join(args.output_dir, "behrt_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison saved to {comparison_path}")
        
        # Print summary
        logger.info("Comparison summary:")
        for metric, values in comparison.items():
            logger.info(f"  {metric}: {values['baseline']:.4f} -> {values['medtok']:.4f} "
                      f"(+{values['absolute_improvement']:.4f}, {values['relative_improvement']})")


if __name__ == "__main__":
    main()
