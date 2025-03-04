#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for evaluating MEDTOK on MIMIC-III tasks.

This script evaluates the MEDTOK tokenizer on various MIMIC-III tasks,
including mortality prediction, readmission prediction, length-of-stay prediction,
phenotype prediction, and drug recommendation.
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
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score
from collections import defaultdict

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
    parser = argparse.ArgumentParser(description="Evaluate MEDTOK on MIMIC-III tasks")
    
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--mimic_dir", type=str, required=True, 
                        help="Directory containing MIMIC-III data")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for results")
    parser.add_argument("--graph_dir", type=str, default=None, 
                        help="Directory containing graph files (required if not in mimic_dir)")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--tasks", type=str, default="all", 
                        help="Tasks to evaluate (all, mortality, readmission, los, phenotype, drugrec)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--baseline", type=str, default=None, 
                        help="Baseline tokenizer to compare with (bert, vqgraph)")
    
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


def load_baseline_tokenizer(baseline_type, device):
    """
    Load a baseline tokenizer.
    
    Args:
        baseline_type: Type of baseline tokenizer
        device: Device to load the tokenizer onto
    
    Returns:
        Baseline tokenizer
    """
    logger.info(f"Loading {baseline_type} tokenizer...")
    
    if baseline_type.lower() == "bert":
        from transformers import BertTokenizer, BertModel
        
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
        
        # Wrap in a class with similar interface to MEDTOK
        class BertTokenizerWrapper:
            def __init__(self, tokenizer, model, device):
                self.tokenizer = tokenizer
                self.model = model
                self.device = device
                self.model.eval()
            
            def tokenize(self, text, *args, **kwargs):
                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get [CLS] token embeddings
                token_embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Convert to token indices (simulating MEDTOK output)
                # Since BERT doesn't have the same discrete tokens,
                # we'll return the indices of the closest vectors in a synthetic codebook
                num_vectors = 12000  # Same as MEDTOK codebook size
                synthetic_codebook = torch.randn(num_vectors, token_embeddings.shape[1], device=self.device)
                
                # Compute distances to codebook vectors
                expanded_embeddings = token_embeddings.unsqueeze(1)  # (batch_size, 1, dim)
                expanded_codebook = synthetic_codebook.unsqueeze(0)  # (1, num_vectors, dim)
                distances = torch.norm(expanded_embeddings - expanded_codebook, dim=2)  # (batch_size, num_vectors)
                
                # Get indices of closest vectors
                _, indices = torch.topk(distances, k=4, dim=1, largest=False)  # (batch_size, 4)
                
                return indices
        
        return BertTokenizerWrapper(tokenizer, model, device)
    
    elif baseline_type.lower() == "vqgraph":
        # For VQGraph, we need to implement its interface
        # This is a simplified version - in practice you would load the actual VQGraph model
        from torch_geometric.nn import GCNConv
        
        class VQGraphTokenizer(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device
                self.gcn1 = GCNConv(128, 256).to(device)
                self.gcn2 = GCNConv(256, 256).to(device)
                self.codebook = torch.nn.Parameter(torch.randn(12000, 256).to(device))
                
                # Set to evaluation mode
                self.eval()
            
            def tokenize(self, _, graph_features, graph_edge_index, graph_batch=None):
                with torch.no_grad():
                    # Apply GCN layers
                    x = torch.relu(self.gcn1(graph_features, graph_edge_index))
                    x = self.gcn2(x, graph_edge_index)
                    
                    # Pool to graph-level representations
                    if graph_batch is None:
                        graph_batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                    
                    from torch_geometric.nn import global_mean_pool
                    graph_embedding = global_mean_pool(x, graph_batch)
                    
                    # Quantize
                    expanded_embeddings = graph_embedding.unsqueeze(1)  # (batch_size, 1, dim)
                    expanded_codebook = self.codebook.unsqueeze(0)  # (1, num_vectors, dim)
                    distances = torch.norm(expanded_embeddings - expanded_codebook, dim=2)  # (batch_size, num_vectors)
                    
                    # Get indices of closest vectors
                    _, indices = torch.topk(distances, k=4, dim=1, largest=False)  # (batch_size, 4)
                    
                    return indices
        
        return VQGraphTokenizer(device)
    
    else:
        logger.error(f"Unknown baseline tokenizer type: {baseline_type}")
        return None


def load_mimic_data(mimic_dir, task):
    """
    Load MIMIC-III data for a specific task.
    
    Args:
        mimic_dir: Directory containing MIMIC-III data
        task: Task to load data for
    
    Returns:
        Dictionary containing data for the task
    """
    logger.info(f"Loading MIMIC-III data for {task} task...")
    
    if task == "mortality":
        # Load mortality data
        data_file = os.path.join(mimic_dir, "mortality", "mortality_data.csv")
        
        if not os.path.exists(data_file):
            logger.error(f"Mortality data file not found: {data_file}")
            return None
        
        data = pd.read_csv(data_file)
        
        return {
            "data": data,
            "task_type": "binary_classification",
            "target_column": "mortality_label",
            "metric": "auprc"
        }
    
    elif task == "readmission":
        # Load readmission data
        data_file = os.path.join(mimic_dir, "readmission", "readmission_data.csv")
        
        if not os.path.exists(data_file):
            logger.error(f"Readmission data file not found: {data_file}")
            return None
        
        data = pd.read_csv(data_file)
        
        return {
            "data": data,
            "task_type": "binary_classification",
            "target_column": "readmission_label",
            "metric": "auprc"
        }
    
    elif task == "los":
        # Load length-of-stay data
        data_file = os.path.join(mimic_dir, "los", "los_data.csv")
        
        if not os.path.exists(data_file):
            logger.error(f"Length-of-stay data file not found: {data_file}")
            return None
        
        data = pd.read_csv(data_file)
        
        return {
            "data": data,
            "task_type": "multiclass_classification",
            "target_column": "los_label",
            "metric": "accuracy",
            "num_classes": 10  # As defined in the paper
        }
    
    elif task == "phenotype":
        # Load phenotype data
        data_file = os.path.join(mimic_dir, "phenotype", "phenotype_data.csv")
        
        if not os.path.exists(data_file):
            logger.error(f"Phenotype data file not found: {data_file}")
            return None
        
        data = pd.read_csv(data_file)
        
        # Get phenotype columns
        phenotype_columns = [col for col in data.columns if col.startswith("phenotype_")]
        
        return {
            "data": data,
            "task_type": "multilabel_classification",
            "target_columns": phenotype_columns,
            "metric": "macro_auprc"
        }
    
    elif task == "drugrec":
        # Load drug recommendation data
        data_file = os.path.join(mimic_dir, "drugrec", "drugrec_data.csv")
        
        if not os.path.exists(data_file):
            logger.error(f"Drug recommendation data file not found: {data_file}")
            return None
        
        data = pd.read_csv(data_file)
        
        # Get drug columns
        drug_columns = [col for col in data.columns if col.startswith("drug_")]
        
        return {
            "data": data,
            "task_type": "multilabel_classification",
            "target_columns": drug_columns,
            "metric": "macro_auprc"
        }
    
    else:
        logger.error(f"Unknown task: {task}")
        return None


def tokenize_codes(model, data, graph_dir, batch_size=32):
    """
    Tokenize medical codes in the data.
    
    Args:
        model: MEDTOK model
        data: DataFrame containing medical codes
        graph_dir: Directory containing graph files
        batch_size: Batch size for tokenization
    
    Returns:
        DataFrame with tokenized codes
    """
    logger.info("Tokenizing medical codes...")
    
    # Check if code and description columns exist
    if "code" not in data.columns or "description" not in data.columns:
        logger.error("Data must contain 'code' and 'description' columns")
        return None
    
    # Create device
    device = next(model.parameters()).device
    
    # Initialize list to store tokenized codes
    tokenized_rows = []
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Tokenizing"):
        batch = data.iloc[i:i+batch_size]
        
        # Get codes and descriptions
        codes = batch["code"].tolist()
        descriptions = batch["description"].tolist()
        
        # Load graph files
        graph_features_list = []
        graph_edge_index_list = []
        
        for code in codes:
            graph_file = os.path.join(graph_dir, f"{code}.json")
            
            if os.path.exists(graph_file):
                # Load graph
                with open(graph_file, "r") as f:
                    graph_data = json.load(f)
                
                # Convert to torch tensors
                import networkx as nx
                G = nx.node_link_graph(graph_data)
                
                # Extract node features
                node_features = []
                for node in G.nodes:
                    if "features" in G.nodes[node]:
                        node_features.append(G.nodes[node]["features"])
                    else:
                        # Create default features
                        node_features.append([0.0] * model.config.node_feature_dim)
                
                # Extract edge indices
                edge_index = []
                for src, dst in G.edges:
                    edge_index.append([src, dst])
                    edge_index.append([dst, src])  # Add reverse edge for undirected graphs
                
                # Convert to torch tensors
                node_features = torch.tensor(node_features, dtype=torch.float).to(device)
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
                
                graph_features_list.append(node_features)
                graph_edge_index_list.append(edge_index)
            else:
                # Create dummy graph
                node_features = torch.zeros((1, model.config.node_feature_dim), dtype=torch.float).to(device)
                edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
                
                graph_features_list.append(node_features)
                graph_edge_index_list.append(edge_index)
        
        # Tokenize descriptions
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        
        encoded_texts = tokenizer(
            descriptions,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Tokenize in batches
        for j in range(len(codes)):
            # Get text and graph data for this code
            input_ids = encoded_texts["input_ids"][j:j+1]
            attention_mask = encoded_texts["attention_mask"][j:j+1]
            graph_features = graph_features_list[j]
            graph_edge_index = graph_edge_index_list[j]
            
            # Create batch tensor for graph
            graph_batch = torch.zeros(graph_features.size(0), dtype=torch.long, device=device)
            
            # Tokenize
            with torch.no_grad():
                token_indices = model.tokenize(
                    input_ids,
                    graph_features,
                    graph_edge_index,
                    graph_batch
                )
            
            # Add to result
            tokenized_row = batch.iloc[j].copy()
            tokenized_row["tokens"] = token_indices[0].cpu().numpy().tolist()
            tokenized_rows.append(tokenized_row)
    
    # Convert to DataFrame
    tokenized_df = pd.DataFrame(tokenized_rows)
    
    logger.info(f"Tokenized {len(tokenized_df)} codes")
    
    return tokenized_df


def evaluate_task(tokenized_data, task_info, output_dir, model_name="medtok"):
    """
    Evaluate a task with tokenized codes.
    
    Args:
        tokenized_data: DataFrame with tokenized codes
        task_info: Dictionary with task information
        output_dir: Output directory for results
        model_name: Name of the model (for saving results)
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {task_info['task_type']} task...")
    
    # Get task type and target columns
    task_type = task_info["task_type"]
    
    if task_type == "binary_classification":
        target_column = task_info["target_column"]
        
        # Split data
        train_data = tokenized_data[tokenized_data["split"] == "train"]
        val_data = tokenized_data[tokenized_data["split"] == "val"]
        test_data = tokenized_data[tokenized_data["split"] == "test"]
        
        # Train a simple logistic regression model
        from sklearn.linear_model import LogisticRegression
        
        # Convert tokens to features
        def tokens_to_features(tokens_list):
            # For simplicity, we'll just use the sum of token indices as features
            return [sum(tokens) for tokens in tokens_list]
        
        X_train = tokens_to_features(train_data["tokens"].tolist())
        y_train = train_data[target_column].values
        
        X_val = tokens_to_features(val_data["tokens"].tolist())
        y_val = val_data[target_column].values
        
        X_test = tokens_to_features(test_data["tokens"].tolist())
        y_test = test_data[target_column].values
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_val = model.predict_proba(X_val)[:, 1]
        y_pred_test = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        val_auprc = average_precision_score(y_val, y_pred_val)
        test_auprc = average_precision_score(y_test, y_pred_test)
        
        val_auroc = roc_auc_score(y_val, y_pred_val)
        test_auroc = roc_auc_score(y_test, y_pred_test)
        
        # Find best threshold on validation set
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_val)
        fscore = (2 * precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(fscore)]
        
        # Apply best threshold on test set
        y_pred_test_binary = (y_pred_test >= best_threshold).astype(int)
        test_f1 = f1_score(y_test, y_pred_test_binary)
        
        metrics = {
            "val_auprc": val_auprc,
            "test_auprc": test_auprc,
            "val_auroc": val_auroc,
            "test_auroc": test_auroc,
            "test_f1": test_f1,
            "best_threshold": best_threshold
        }
    
    elif task_type == "multiclass_classification":
        target_column = task_info["target_column"]
        num_classes = task_info["num_classes"]
        
        # Split data
        train_data = tokenized_data[tokenized_data["split"] == "train"]
        val_data = tokenized_data[tokenized_data["split"] == "val"]
        test_data = tokenized_data[tokenized_data["split"] == "test"]
        
        # Train a simple multinomial logistic regression model
        from sklearn.linear_model import LogisticRegression
        
        # Convert tokens to features
        def tokens_to_features(tokens_list):
            # For simplicity, we'll just use the sum of token indices as features
            return [sum(tokens) for tokens in tokens_list]
        
        X_train = tokens_to_features(train_data["tokens"].tolist())
        y_train = train_data[target_column].values
        
        X_val = tokens_to_features(val_data["tokens"].tolist())
        y_val = val_data[target_column].values
        
        X_test = tokens_to_features(test_data["tokens"].tolist())
        y_test = test_data[target_column].values
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000, multi_class="multinomial")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        
        val_accuracy = accuracy_score(y_val, y_pred_val)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        val_balanced_accuracy = balanced_accuracy_score(y_val, y_pred_val)
        test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_test)
        
        metrics = {
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "val_balanced_accuracy": val_balanced_accuracy,
            "test_balanced_accuracy": test_balanced_accuracy
        }
    
    elif task_type == "multilabel_classification":
        target_columns = task_info["target_columns"]
        
        # Split data
        train_data = tokenized_data[tokenized_data["split"] == "train"]
        val_data = tokenized_data[tokenized_data["split"] == "val"]
        test_data = tokenized_data[tokenized_data["split"] == "test"]
        
        # Train a separate logistic regression model for each target
        from sklearn.linear_model import LogisticRegression
        
        # Convert tokens to features
        def tokens_to_features(tokens_list):
            # For simplicity, we'll just use the sum of token indices as features
            return [sum(tokens) for tokens in tokens_list]
        
        X_train = tokens_to_features(train_data["tokens"].tolist())
        X_val = tokens_to_features(val_data["tokens"].tolist())
        X_test = tokens_to_features(test_data["tokens"].tolist())
        
        # Initialize metrics
        val_auprcs = []
        test_auprcs = []
        val_aurocs = []
        test_aurocs = []
        test_f1s = []
        
        # Train and evaluate for each target
        for target_column in target_columns:
            y_train = train_data[target_column].values
            y_val = val_data[target_column].values
            y_test = test_data[target_column].values
            
            # Skip if all targets are the same (no positive samples)
            if np.all(y_train == 0) or np.all(y_train == 1):
                continue
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_val = model.predict_proba(X_val)[:, 1]
            y_pred_test = model.predict_proba(X_test)[:, 1]
            
            # Compute metrics
            try:
                val_auprc = average_precision_score(y_val, y_pred_val)
                test_auprc = average_precision_score(y_test, y_pred_test)
                
                val_auroc = roc_auc_score(y_val, y_pred_val)
                test_auroc = roc_auc_score(y_test, y_pred_test)
                
                # Find best threshold on validation set
                precision, recall, thresholds = precision_recall_curve(y_val, y_pred_val)
                fscore = (2 * precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(fscore)
                best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                
                # Apply best threshold on test set
                y_pred_test_binary = (y_pred_test >= best_threshold).astype(int)
                test_f1 = f1_score(y_test, y_pred_test_binary)
                
                val_auprcs.append(val_auprc)
                test_auprcs.append(test_auprc)
                val_aurocs.append(val_auroc)
                test_aurocs.append(test_auroc)
                test_f1s.append(test_f1)
            except:
                # Skip targets with errors (e.g., only one class present)
                continue
        
        # Compute macro-averaged metrics
        metrics = {
            "val_macro_auprc": np.mean(val_auprcs),
            "test_macro_auprc": np.mean(test_auprcs),
            "val_macro_auroc": np.mean(val_aurocs),
            "test_macro_auroc": np.mean(test_aurocs),
            "test_macro_f1": np.mean(test_f1s),
            "num_targets": len(val_auprcs)
        }
    
    else:
        logger.error(f"Unknown task type: {task_type}")
        return None
    
    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, f"{model_name}_{task_info['metric']}.json")
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_file}")
    
    return metrics


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load MEDTOK model
    model = load_medtok_model(args.model_path, device)
    
    if model is None:
        logger.error("Failed to load MEDTOK model")
        return
    
    # Load baseline tokenizer if specified
    baseline_model = None
    if args.baseline:
        baseline_model = load_baseline_tokenizer(args.baseline, device)
        
        if baseline_model is None:
            logger.error(f"Failed to load {args.baseline} tokenizer")
            return
    
    # Determine graph directory
    graph_dir = args.graph_dir
    if graph_dir is None:
        graph_dir = os.path.join(args.mimic_dir, "graphs")
        if not os.path.exists(graph_dir):
            logger.error(f"Graph directory not found: {graph_dir}")
            logger.error("Please specify --graph_dir")
            return
    
    # Determine tasks to evaluate
    tasks = []
    if args.tasks.lower() == "all":
        tasks = ["mortality", "readmission", "los", "phenotype", "drugrec"]
    else:
        tasks = args.tasks.lower().split(",")
    
    # Evaluate each task
    results = {}
    
    for task in tasks:
        logger.info(f"Evaluating task: {task}")
        
        # Load task data
        task_data = load_mimic_data(args.mimic_dir, task)
        
        if task_data is None:
            logger.error(f"Failed to load data for task: {task}")
            continue
        
        # Create task output directory
        task_output_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # Tokenize codes with MEDTOK
        medtok_tokenized = tokenize_codes(model, task_data["data"], graph_dir, args.batch_size)
        
        if medtok_tokenized is None:
            logger.error(f"Failed to tokenize codes for task: {task}")
            continue
        
        # Evaluate MEDTOK
        medtok_metrics = evaluate_task(medtok_tokenized, task_data, task_output_dir)
        
        # Tokenize codes with baseline if specified
        if baseline_model is not None:
            baseline_tokenized = tokenize_codes(baseline_model, task_data["data"], graph_dir, args.batch_size)
            
            if baseline_tokenized is None:
                logger.error(f"Failed to tokenize codes with {args.baseline} for task: {task}")
            else:
                # Evaluate baseline
                baseline_metrics = evaluate_task(baseline_tokenized, task_data, task_output_dir, args.baseline)
                
                # Compare results
                if medtok_metrics and baseline_metrics:
                    metric_name = f"test_{task_data['metric']}"
                    
                    if metric_name in medtok_metrics and metric_name in baseline_metrics:
                        medtok_value = medtok_metrics[metric_name]
                        baseline_value = baseline_metrics[metric_name]
                        
                        improvement = medtok_value - baseline_value
                        relative_improvement = improvement / baseline_value * 100
                        
                        comparison = {
                            "medtok": medtok_value,
                            "baseline": baseline_value,
                            "absolute_improvement": improvement,
                            "relative_improvement": relative_improvement
                        }
                        
                        # Save comparison
                        comparison_file = os.path.join(task_output_dir, f"comparison_{args.baseline}.json")
                        
                        with open(comparison_file, "w") as f:
                            json.dump(comparison, f, indent=2)
                        
                        logger.info(f"Comparison saved to {comparison_file}")
                        
                        # Add to results
                        results[task] = comparison
        
        else:
            # Add MEDTOK results only
            metric_name = f"test_{task_data['metric']}"
            
            if medtok_metrics and metric_name in medtok_metrics:
                results[task] = {
                    "medtok": medtok_metrics[metric_name]
                }
    
    # Save overall results
    results_file = os.path.join(args.output_dir, "results.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    logger.info("Evaluation summary:")
    
    for task, metrics in results.items():
        if "baseline" in metrics:
            logger.info(f"Task: {task}")
            logger.info(f"  MEDTOK: {metrics['medtok']:.4f}")
            logger.info(f"  {args.baseline.upper()}: {metrics['baseline']:.4f}")
            logger.info(f"  Improvement: {metrics['absolute_improvement']:.4f} ({metrics['relative_improvement']:.2f}%)")
        else:
            logger.info(f"Task: {task}")
            logger.info(f"  MEDTOK: {metrics['medtok']:.4f}")


if __name__ == "__main__":
    main()
