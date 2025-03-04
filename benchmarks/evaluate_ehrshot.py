#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for evaluating MEDTOK on EHRShot tasks.

This script evaluates the MEDTOK tokenizer on EHRShot tasks,
including operational outcomes and new diagnosis assignments.
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
    parser = argparse.ArgumentParser(description="Evaluate MEDTOK on EHRShot tasks")
    
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to trained MEDTOK model")
    parser.add_argument("--ehrshot_dir", type=str, required=True, 
                        help="Directory containing EHRShot data")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for results")
    parser.add_argument("--graph_dir", type=str, default=None, 
                        help="Directory containing graph files (required if not in ehrshot_dir)")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--tasks", type=str, default="all", 
                        help="Tasks to evaluate (all, oo, nd)")
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


def load_ehrshot_data(ehrshot_dir, task):
    """
    Load EHRShot data for a specific task.
    
    Args:
        ehrshot_dir: Directory containing EHRShot data
        task: Task to load data for
    
    Returns:
        Dictionary containing data for the task
    """
    logger.info(f"Loading EHRShot data for {task} task...")
    
    if task == "oo":
        # Load operational outcomes data
        
        # Define the three operational outcome tasks
        subtasks = ["long_los", "readmission_15", "mortality"]
        task_data = {}
        
        for subtask in subtasks:
            data_file = os.path.join(ehrshot_dir, "operational_outcomes", f"{subtask}.csv")
            
            if not os.path.exists(data_file):
                logger.error(f"Operational outcomes data file not found: {data_file}")
                continue
            
            data = pd.read_csv(data_file)
            
            task_data[subtask] = {
                "data": data,
                "task_type": "binary_classification",
                "target_column": "label",
                "metric": "auprc"
            }
        
        if not task_data:
            logger.error("No operational outcomes data found")
            return None
        
        return {
            "subtasks": task_data,
            "task_type": "multitask",
            "metric": "auprc"
        }
    
    elif task == "nd":
        # Load new diagnosis assignments data
        
        # Define the four new diagnosis tasks from the paper
        subtasks = ["hypertension", "hyperlipidemia", "pancreatic_cancer", "acute_mi"]
        task_data = {}
        
        for subtask in subtasks:
            data_file = os.path.join(ehrshot_dir, "new_diagnosis", f"{subtask}.csv")
            
            if not os.path.exists(data_file):
                logger.error(f"New diagnosis data file not found: {data_file}")
                continue
            
            data = pd.read_csv(data_file)
            
            task_data[subtask] = {
                "data": data,
                "task_type": "binary_classification",
                "target_column": "label",
                "metric": "auprc"
            }
        
        if not task_data:
            logger.error("No new diagnosis data found")
            return None
        
        return {
            "subtasks": task_data,
            "task_type": "multitask",
            "metric": "auprc"
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
    
    elif task_type == "multitask":
        subtasks = task_info["subtasks"]
        all_metrics = {}
        
        for subtask_name, subtask_info in subtasks.items():
            logger.info(f"Evaluating subtask: {subtask_name}")
            
            # Evaluate subtask
            subtask_metrics = evaluate_task(
                tokenized_data,
                subtask_info,
                os.path.join(output_dir, subtask_name),
                model_name
            )
            
            # Add to all metrics
            if subtask_metrics:
                all_metrics[subtask_name] = subtask_metrics
        
        # Compute average metrics across subtasks
        if all_metrics:
            avg_metrics = {}
            
            # Get common metric names
            common_metrics = set.intersection(*[set(m.keys()) for m in all_metrics.values()])
            
            # Compute averages
            for metric in common_metrics:
                avg_metrics[f"avg_{metric}"] = np.mean([m[metric] for m in all_metrics.values()])
            
            # Add to all metrics
            all_metrics["average"] = avg_metrics
            
            # Save average metrics
            avg_metrics_file = os.path.join(output_dir, f"{model_name}_average.json")
            
            with open(avg_metrics_file, "w") as f:
                json.dump(avg_metrics, f, indent=2)
            
            logger.info(f"Average metrics saved to {avg_metrics_file}")
        
        metrics = all_metrics
    
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
        graph_dir = os.path.join(args.ehrshot_dir, "graphs")
        if not os.path.exists(graph_dir):
            logger.error(f"Graph directory not found: {graph_dir}")
            logger.error("Please specify --graph_dir")
            return
    
    # Determine tasks to evaluate
    tasks = []
    if args.tasks.lower() == "all":
        tasks = ["oo", "nd"]
    else:
        tasks = args.tasks.lower().split(",")
    
    # Evaluate each task
    results = {}
    
    for task in tasks:
        logger.info(f"Evaluating task: {task}")
        
        # Load task data
        task_data = load_ehrshot_data(args.ehrshot_dir, task)
        
        if task_data is None:
            logger.error(f"Failed to load data for task: {task}")
            continue
        
        # Create task output directory
        task_output_dir = os.path.join(args.output_dir, task)
        os.makedirs(task_output_dir, exist_ok=True)
        
        # If this is a multitask setup, we need to process each subtask separately
        if task_data["task_type"] == "multitask":
            subtasks_results = {}
            
            for subtask_name, subtask_info in task_data["subtasks"].items():
                logger.info(f"Processing subtask: {subtask_name}")
                
                # Create subtask output directory
                subtask_output_dir = os.path.join(task_output_dir, subtask_name)
                os.makedirs(subtask_output_dir, exist_ok=True)
                
                # Tokenize codes with MEDTOK
                medtok_tokenized = tokenize_codes(model, subtask_info["data"], graph_dir, args.batch_size)
                
                if medtok_tokenized is None:
                    logger.error(f"Failed to tokenize codes for subtask: {subtask_name}")
                    continue
                
                # Evaluate MEDTOK
                medtok_metrics = evaluate_task(medtok_tokenized, subtask_info, subtask_output_dir)
                
                # Tokenize codes with baseline if specified
                if baseline_model is not None:
                    baseline_tokenized = tokenize_codes(baseline_model, subtask_info["data"], graph_dir, args.batch_size)
                    
                    if baseline_tokenized is None:
                        logger.error(f"Failed to tokenize codes with {args.baseline} for subtask: {subtask_name}")
                    else:
                        # Evaluate baseline
                        baseline_metrics = evaluate_task(baseline_tokenized, subtask_info, subtask_output_dir, args.baseline)
                        
                        # Compare results
                        if medtok_metrics and baseline_metrics:
                            metric_name = f"test_{subtask_info['metric']}"
                            
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
                                comparison_file = os.path.join(subtask_output_dir, f"comparison_{args.baseline}.json")
                                
                                with open(comparison_file, "w") as f:
                                    json.dump(comparison, f, indent=2)
                                
                                logger.info(f"Comparison saved to {comparison_file}")
                                
                                # Add to results
                                subtasks_results[subtask_name] = comparison
                
                else:
                    # Add MEDTOK results only
                    metric_name = f"test_{subtask_info['metric']}"
                    
                    if medtok_metrics and metric_name in medtok_metrics:
                        subtasks_results[subtask_name] = {
                            "medtok": medtok_metrics[metric_name]
                        }
            
            # Compute average improvement across subtasks
            if subtasks_results:
                if args.baseline:
                    # If baseline is specified, compute average improvement
                    improvements = [result["relative_improvement"] for result in subtasks_results.values() if "relative_improvement" in result]
                    
                    if improvements:
                        avg_improvement = np.mean(improvements)
                        
                        # Add to results
                        results[task] = {
                            "subtasks": subtasks_results,
                            "average_improvement": avg_improvement
                        }
                else:
                    # Otherwise, just add the subtask results
                    results[task] = {"subtasks": subtasks_results}
        
        else:
            # Process as a single task
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
    
    for task, task_results in results.items():
        logger.info(f"Task: {task}")
        
        if "subtasks" in task_results:
            # Multitask results
            logger.info(f"  Subtasks:")
            
            for subtask, metrics in task_results["subtasks"].items():
                if "baseline" in metrics:
                    logger.info(f"    {subtask}:")
                    logger.info(f"      MEDTOK: {metrics['medtok']:.4f}")
                    logger.info(f"      {args.baseline.upper()}: {metrics['baseline']:.4f}")
                    logger.info(f"      Improvement: {metrics['absolute_improvement']:.4f} ({metrics['relative_improvement']:.2f}%)")
                else:
                    logger.info(f"    {subtask}:")
                    logger.info(f"      MEDTOK: {metrics['medtok']:.4f}")
            
            if "average_improvement" in task_results:
                logger.info(f"  Average improvement: {task_results['average_improvement']:.2f}%")
        
        elif "baseline" in task_results:
            # Single task with baseline comparison
            logger.info(f"  MEDTOK: {task_results['medtok']:.4f}")
            logger.info(f"  {args.baseline.upper()}: {task_results['baseline']:.4f}")
            logger.info(f"  Improvement: {task_results['absolute_improvement']:.4f} ({task_results['relative_improvement']:.2f}%)")
        
        else:
            # Single task without baseline
            logger.info(f"  MEDTOK: {task_results['medtok']:.4f}")


if __name__ == "__main__":
    main()
