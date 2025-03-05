#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MulT-EHR integration script for MEDTOK.

This script shows how to integrate MEDTOK with the MulT-EHR model.
MulT-EHR uses multi-task heterogeneous graph learning with causal denoising 
to address data heterogeneity and confounding effects.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import logging
import networkx as nx

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
    parser = argparse.ArgumentParser(description="Integrate MEDTOK with MulT-EHR model")
    
    parser.add_argument("--medtok_model", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--mult_ehr_model", type=str, default=None, 
                        help="Path to MulT-EHR model (if None, will initialize new)")
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
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=256, 
                        help="Hidden size for MulT-EHR")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--task", type=str, default="mortality", 
                        help="Task to evaluate (mortality, readmission, los, etc.)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    parser.add_argument("--baseline", action="store_true", 
                        help="Run baseline MulT-EHR (without MEDTOK)")
    
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


def load_mult_ehr_model(model_path=None, hidden_size=256, device="cuda"):
    """
    Load or initialize MulT-EHR model.
    
    Args:
        model_path: Path to model (if None, will initialize new)
        hidden_size: Hidden size for the model
        device: Device to load model onto
        
    Returns:
        MulT-EHR model
    """
    logger.info("Loading/initializing MulT-EHR model...")
    
    try:
        # Import MulT-EHR module (assuming it's installed)
        import mult_ehr
        
        if model_path is not None:
            # Load pre-trained model
            mult_ehr_model = mult_ehr.model.load_from_checkpoint(model_path)
        else:
            # Initialize new model
            mult_ehr_model = mult_ehr.model.MultEHR(hidden_dim=hidden_size)
        
        # Move to device
        mult_ehr_model = mult_ehr_model.to(device)
        
        return mult_ehr_model
    
    except ImportError:
        logger.error("MulT-EHR module not found. Please install with 'pip install mult-ehr'.")
        
        # Create a simple placeholder model for demonstration
        logger.info("Creating a placeholder MulT-EHR model for demonstration...")
        
        class PlaceholderMultEHR(torch.nn.Module):
            def __init__(self, hidden_dim=256):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.node_embedding = torch.nn.Embedding(10000, hidden_dim)
                self.gnn = torch.nn.Linear(hidden_dim, hidden_dim)
                self.classifier = torch.nn.Linear(hidden_dim, 1)
            
            def forward(self, node_ids, edge_index, batch):
                x = self.node_embedding(node_ids)
                x = self.gnn(x)
                x = torch.mean(x, dim=0)
                return self.classifier(x)
        
        model = PlaceholderMultEHR(hidden_dim=hidden_size).to(device)
        return model
    
    except Exception as e:
        logger.error(f"Error loading MulT-EHR model: {e}")
        return None


def tokenize_medical_code(medtok_model, code, description, graph_file):
    """
    Tokenize a medical code using MEDTOK.
    
    Args:
        medtok_model: MEDTOK model
        code: Medical code ID
        description: Description of the code
        graph_file: Path to the graph file
        
    Returns:
        Token indices for the code
    """
    device = next(medtok_model.parameters()).device
    
    # Tokenize text
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(medtok_model.config.text_encoder_model)
    
    encoded_text = tokenizer(
        description,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Load graph
    if os.path.exists(graph_file):
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        
        # Convert to NetworkX graph
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


def enhance_mult_ehr_with_medtok(mult_ehr_model, medtok_model, data, graph_dir):
    """
    Enhance MulT-EHR model with MEDTOK embeddings.
    
    Args:
        mult_ehr_model: MulT-EHR model
        medtok_model: MEDTOK model
        data: EHR data
        graph_dir: Directory containing graph files
        
    Returns:
        Enhanced MulT-EHR model
    """
    logger.info("Enhancing MulT-EHR with MEDTOK embeddings...")
    
    # Process each node type in the heterogeneous graph
    for node_type in mult_ehr_model.node_types:
        if node_type in ["diagnosis", "procedure", "medication"]:
            logger.info(f"Processing node type: {node_type}")
            
            for node_id in tqdm(mult_ehr_model.nodes[node_type]):
                # Get code ID for the node
                code_id = mult_ehr_model.node_id_to_code[node_type][node_id]
                
                # Get description for the code
                description = data.code_descriptions.get(code_id, f"Medical code: {code_id}")
                
                # Tokenize with MEDTOK
                graph_file = os.path.join(graph_dir, f"{code_id}.json")
                token_indices = tokenize_medical_code(
                    medtok_model,
                    code_id,
                    description,
                    graph_file
                )
                
                # Get token embeddings from MEDTOK
                token_embedding = medtok_model.get_token_embedding(token_indices).mean(dim=1)
                
                # Replace or concat with original node features
                original_features = mult_ehr_model.node_features[node_type][node_id]
                enhanced_features = torch.cat([original_features, token_embedding], dim=-1)
                
                # Update node features in MulT-EHR
                mult_ehr_model.node_features[node_type][node_id] = enhanced_features
    
    # Update feature dimensions in the model
    mult_ehr_model.update_feature_dimensions()
    
    logger.info("MulT-EHR model enhanced with MEDTOK embeddings")
    
    return mult_ehr_model


def load_ehr_data(data_dir, task):
    """
    Load EHR data for the specified task.
    
    Args:
        data_dir: Directory containing the data
        task: Task name
        
    Returns:
        Loaded data
    """
    logger.info(f"Loading EHR data for task: {task}")
    
    # For this example, we'll create a simple placeholder data loader
    # In a real implementation, this would load actual MIMIC or EHRShot data
    
    class EHRData:
        def __init__(self, data_dir, task):
            self.data_dir = data_dir
            self.task = task
            self.code_descriptions = {}
            self.nodes = {}
            self.edges = {}
            
            # Load code descriptions
            try:
                codes_df = pd.read_csv(os.path.join(data_dir, "medical_codes_all.csv"))
                self.code_descriptions = {
                    row["code"]: row["description"] 
                    for _, row in codes_df.iterrows()
                }
            except:
                logger.warning("Could not load code descriptions")
            
            # Load task-specific data
            self.train_data = self._load_split("train")
            self.val_data = self._load_split("val")
            self.test_data = self._load_split("test")
        
        def _load_split(self, split):
            # In a real implementation, this would load actual data files
            return {
                "patients": list(range(100)),
                "visits": [{"codes": [f"code_{i}" for i in range(5)]} for _ in range(100)],
                "labels": np.random.randint(0, 2, 100)
            }
    
    return EHRData(data_dir, task)


def train_mult_ehr(model, data, device, epochs=20, lr=1e-4, batch_size=32):
    """
    Train the MulT-EHR model.
    
    Args:
        model: MulT-EHR model
        data: Training data
        device: Device to use
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        
    Returns:
        Trained model
    """
    logger.info("Training MulT-EHR model...")
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up loss function (binary cross entropy for classification tasks)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Process in batches
        for i in range(0, len(data.train_data["patients"]), batch_size):
            batch_patients = data.train_data["patients"][i:i+batch_size]
            batch_labels = torch.tensor(
                data.train_data["labels"][i:i+batch_size], 
                dtype=torch.float
            ).to(device)
            
            # In a real implementation, we would construct the actual graph here
            # For this example, we'll just use dummy inputs
            node_ids = torch.randint(0, 1000, (len(batch_patients), 10)).to(device)
            edge_index = torch.randint(0, 10, (2, 20)).to(device)
            batch = torch.zeros(10, dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(node_ids, edge_index, batch)
            loss = loss_fn(outputs.squeeze(), batch_labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average loss
        avg_train_loss = train_loss / (len(data.train_data["patients"]) // batch_size)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(data.val_data["patients"]), batch_size):
                batch_patients = data.val_data["patients"][i:i+batch_size]
                batch_labels = torch.tensor(
                    data.val_data["labels"][i:i+batch_size], 
                    dtype=torch.float
                ).to(device)
                
                # Dummy inputs for demonstration
                node_ids = torch.randint(0, 1000, (len(batch_patients), 10)).to(device)
                edge_index = torch.randint(0, 10, (2, 20)).to(device)
                batch = torch.zeros(10, dtype=torch.long).to(device)
                
                # Forward pass
                outputs = model(node_ids, edge_index, batch)
                loss = loss_fn(outputs.squeeze(), batch_labels)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / (len(data.val_data["patients"]) // batch_size)
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    logger.info("Training completed")
    return model


def evaluate_mult_ehr(model, data, device, batch_size=32):
    """
    Evaluate the MulT-EHR model.
    
    Args:
        model: MulT-EHR model
        data: Test data
        device: Device to use
        batch_size: Batch size
        
    Returns:
        Evaluation metrics
    """
    logger.info("Evaluating MulT-EHR model...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(data.test_data["patients"]), batch_size):
            batch_patients = data.test_data["patients"][i:i+batch_size]
            batch_labels = torch.tensor(
                data.test_data["labels"][i:i+batch_size], 
                dtype=torch.float
            ).to(device)
            
            # Dummy inputs for demonstration
            node_ids = torch.randint(0, 1000, (len(batch_patients), 10)).to(device)
            edge_index = torch.randint(0, 10, (2, 20)).to(device)
            batch = torch.zeros(10, dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(node_ids, edge_index, batch)
            
            # Store predictions and labels
            all_preds.append(outputs.sigmoid().cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    
    try:
        auroc = roc_auc_score(all_labels, all_preds)
    except:
        auroc = 0.0
    
    try:
        auprc = average_precision_score(all_labels, all_preds)
    except:
        auprc = 0.0
    
    # Calculate F1 score at threshold 0.5
    preds_binary = (all_preds > 0.5).astype(int)
    f1 = f1_score(all_labels, preds_binary)
    
    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1
    }
    
    logger.info(f"Evaluation results: AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}, F1 = {f1:.4f}")
    
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
    
    # Load data
    data = load_ehr_data(args.data_dir, args.task)
    
    # Determine graph directory
    graph_dir = args.graph_dir
    if graph_dir is None:
        graph_dir = os.path.join(args.data_dir, "graphs")
        if not os.path.exists(graph_dir):
            logger.error(f"Graph directory not found: {graph_dir}")
            logger.error("Please specify --graph_dir")
            return
    
    # Baseline evaluation (without MEDTOK)
    if args.baseline:
        logger.info("Running baseline MulT-EHR (without MEDTOK)...")
        
        # Load/initialize MulT-EHR
        baseline_model = load_mult_ehr_model(
            model_path=args.mult_ehr_model,
            hidden_size=args.hidden_size,
            device=device
        )
        
        if baseline_model is None:
            logger.error("Failed to load/initialize baseline MulT-EHR model")
            return
        
        # Train baseline model
        baseline_model = train_mult_ehr(
            baseline_model,
            data,
            device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size
        )
        
        # Evaluate baseline model
        baseline_metrics = evaluate_mult_ehr(
            baseline_model,
            data,
            device,
            batch_size=args.batch_size
        )
        
        # Save baseline metrics
        baseline_metrics_path = os.path.join(args.output_dir, "baseline_metrics.json")
        with open(baseline_metrics_path, "w") as f:
            json.dump(baseline_metrics, f, indent=2)
        
        logger.info(f"Baseline metrics saved to {baseline_metrics_path}")
    
    # MEDTOK-enhanced evaluation
    logger.info("Running MulT-EHR enhanced with MEDTOK...")
    
    # Load MEDTOK model
    medtok_model = load_medtok_model(args.medtok_model, device)
    
    if medtok_model is None:
        logger.error("Failed to load MEDTOK model")
        return
    
    # Load/initialize MulT-EHR
    mult_ehr_model = load_mult_ehr_model(
        model_path=args.mult_ehr_model,
        hidden_size=args.hidden_size,
        device=device
    )
    
    if mult_ehr_model is None:
        logger.error("Failed to load/initialize MulT-EHR model")
        return
    
    # Enhance MulT-EHR with MEDTOK
    enhanced_model = enhance_mult_ehr_with_medtok(
        mult_ehr_model,
        medtok_model,
        data,
        graph_dir
    )
    
    # Train enhanced model
    enhanced_model = train_mult_ehr(
        enhanced_model,
        data,
        device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    # Save enhanced model
    enhanced_model_path = os.path.join(args.output_dir, "mult_ehr_medtok.pt")
    torch.save(enhanced_model.state_dict(), enhanced_model_path)
    logger.info(f"Enhanced model saved to {enhanced_model_path}")
    
    # Evaluate enhanced model
    enhanced_metrics = evaluate_mult_ehr(
        enhanced_model,
        data,
        device,
        batch_size=args.batch_size
    )
    
    # Save enhanced metrics
    enhanced_metrics_path = os.path.join(args.output_dir, "medtok_metrics.json")
    with open(enhanced_metrics_path, "w") as f:
        json.dump(enhanced_metrics, f, indent=2)
    
    logger.info(f"Enhanced metrics saved to {enhanced_metrics_path}")
    
    # Compare metrics if baseline was run
    if args.baseline:
        comparison = {
            "baseline": baseline_metrics,
            "medtok_enhanced": enhanced_metrics,
            "improvements": {
                metric: enhanced_metrics[metric] - baseline_metrics[metric]
                for metric in enhanced_metrics
            }
        }
        
        # Calculate relative improvements
        for metric in comparison["improvements"]:
            if baseline_metrics[metric] > 0:
                rel_improvement = comparison["improvements"][metric] / baseline_metrics[metric] * 100
                comparison["relative_improvements"] = {
                    metric: f"{rel_improvement:.2f}%"
                }
        
        # Save comparison
        comparison_path = os.path.join(args.output_dir, "comparison.json")
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison saved to {comparison_path}")
        
        # Print summary
        logger.info("Evaluation summary:")
        logger.info(f"  Baseline AUPRC: {baseline_metrics['auprc']:.4f}")
        logger.info(f"  MEDTOK-enhanced AUPRC: {enhanced_metrics['auprc']:.4f}")
        improvement = enhanced_metrics['auprc'] - baseline_metrics['auprc']
        rel_improvement = improvement / baseline_metrics['auprc'] * 100 if baseline_metrics['auprc'] > 0 else 0
        logger.info(f"  Absolute improvement: {improvement:.4f}")
        logger.info(f"  Relative improvement: {rel_improvement:.2f}%")


if __name__ == "__main__":
    main()

