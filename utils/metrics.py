import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def compute_medtok_metrics(model, dataloader, config):
    """
    Compute metrics for evaluating the MEDTOK tokenizer.
    
    Args:
        model: The MEDTOK model
        dataloader: Evaluation dataloader
        config: Configuration object
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Collect embeddings and tokens
    text_embeddings = []
    graph_embeddings = []
    e_text_s = []
    e_graph_s = []
    e_text_c = []
    e_graph_c = []
    z_text_s = []
    z_graph_s = []
    z_text_c = []
    z_graph_c = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            graph_features = batch["graph_features"].to(config.device)
            graph_edge_index = batch["graph_edge_index"].to(config.device)
            graph_batch = batch["graph_batch"].to(config.device)
            
            # Forward pass
            outputs = model(
                input_ids, 
                graph_features, 
                graph_edge_index, 
                graph_batch
            )
            
            # Collect embeddings
            text_embeddings.append(model.encode_text(input_ids).detach().cpu())
            graph_embeddings.append(model.encode_graph(graph_features, graph_edge_index, graph_batch).detach().cpu())
            
            # Collect modality-specific and cross-modality embeddings
            e_text_s.append(outputs["e_text_s"].detach().cpu())
            e_graph_s.append(outputs["e_graph_s"].detach().cpu())
            e_text_c.append(outputs["e_text_c"].detach().cpu())
            e_graph_c.append(outputs["e_graph_c"].detach().cpu())
            
            # Collect quantized embeddings
            z_text_s.append(outputs["z_text_s"].detach().cpu())
            z_graph_s.append(outputs["z_graph_s"].detach().cpu())
            z_text_c.append(outputs["z_text_c"].detach().cpu())
            z_graph_c.append(outputs["z_graph_c"].detach().cpu())
    
    # Concatenate embeddings
    text_embeddings = torch.cat(text_embeddings, dim=0)
    graph_embeddings = torch.cat(graph_embeddings, dim=0)
    e_text_s = torch.cat(e_text_s, dim=0)
    e_graph_s = torch.cat(e_graph_s, dim=0)
    e_text_c = torch.cat(e_text_c, dim=0)
    e_graph_c = torch.cat(e_graph_c, dim=0)
    z_text_s = torch.cat(z_text_s, dim=0)
    z_graph_s = torch.cat(z_graph_s, dim=0)
    z_text_c = torch.cat(z_text_c, dim=0)
    z_graph_c = torch.cat(z_graph_c, dim=0)
    
    # Compute metrics
    metrics = {}
    
    # Reconstruction error
    metrics["text_s_recon_error"] = compute_reconstruction_error(e_text_s, z_text_s)
    metrics["graph_s_recon_error"] = compute_reconstruction_error(e_graph_s, z_graph_s)
    metrics["text_c_recon_error"] = compute_reconstruction_error(e_text_c, z_text_c)
    metrics["graph_c_recon_error"] = compute_reconstruction_error(e_graph_c, z_graph_c)
    metrics["avg_recon_error"] = (metrics["text_s_recon_error"] + 
                                metrics["graph_s_recon_error"] + 
                                metrics["text_c_recon_error"] + 
                                metrics["graph_c_recon_error"]) / 4
    
    # Codebook utilization
    token_indices = torch.cat([
        model.codebook.text_specific_quantizer.get_token_indices(model.codebook.text_specific_quantizer.compute_distances(e_text_s)),
        model.codebook.graph_specific_quantizer.get_token_indices(model.codebook.graph_specific_quantizer.compute_distances(e_graph_s)),
        model.codebook.shared_quantizer.get_token_indices(model.codebook.shared_quantizer.compute_distances(e_text_c)),
        model.codebook.shared_quantizer.get_token_indices(model.codebook.shared_quantizer.compute_distances(e_graph_c))
    ], dim=1).view(-1).cpu().numpy()
    
    metrics["codebook_utilization"] = compute_codebook_utilization(token_indices, config.codebook_size)
    
    # Modality alignment
    metrics["text_graph_alignment"] = compute_modality_alignment(e_text_c, e_graph_c)
    
    # Information preservation
    metrics["text_info_preservation"] = compute_information_preservation(text_embeddings, z_text_s, z_text_c)
    metrics["graph_info_preservation"] = compute_information_preservation(graph_embeddings, z_graph_s, z_graph_c)
    metrics["avg_info_preservation"] = (metrics["text_info_preservation"] + metrics["graph_info_preservation"]) / 2
    
    return metrics


def compute_reconstruction_error(original, reconstructed):
    """
    Compute the mean squared error between original and reconstructed embeddings.
    
    Args:
        original: Original embeddings
        reconstructed: Reconstructed embeddings
    
    Returns:
        Mean squared error
    """
    return torch.mean((original - reconstructed) ** 2).item()


def compute_codebook_utilization(token_indices, codebook_size):
    """
    Compute the percentage of codebook vectors that are used.
    
    Args:
        token_indices: Indices of tokens used
        codebook_size: Total size of the codebook
    
    Returns:
        Percentage of codebook utilization
    """
    unique_tokens = np.unique(token_indices)
    return len(unique_tokens) / codebook_size


def compute_modality_alignment(text_embeddings, graph_embeddings):
    """
    Compute the alignment between text and graph embeddings.
    
    Args:
        text_embeddings: Text embeddings
        graph_embeddings: Graph embeddings
    
    Returns:
        Alignment score
    """
    # Normalize embeddings
    text_norm = torch.nn.functional.normalize(text_embeddings, p=2, dim=1)
    graph_norm = torch.nn.functional.normalize(graph_embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.matmul(text_norm, graph_norm.t())
    
    # Compute average similarity for corresponding pairs
    diagonal = torch.diagonal(similarity)
    alignment = torch.mean(diagonal).item()
    
    return alignment


def compute_information_preservation(original, specific, common):
    """
    Compute how much information from the original embeddings is preserved.
    
    Args:
        original: Original embeddings
        specific: Modality-specific embeddings
        common: Cross-modality embeddings
    
    Returns:
        Information preservation score
    """
    # Normalize embeddings
    original_norm = torch.nn.functional.normalize(original, p=2, dim=1)
    specific_norm = torch.nn.functional.normalize(specific, p=2, dim=1)
    common_norm = torch.nn.functional.normalize(common, p=2, dim=1)
    
    # Compute cosine similarity
    specific_sim = torch.matmul(original_norm, specific_norm.t())
    common_sim = torch.matmul(original_norm, common_norm.t())
    
    # Compute diagonal similarities
    specific_diag = torch.diagonal(specific_sim)
    common_diag = torch.diagonal(common_sim)
    
    # Combine similarities (weighted sum)
    combined_sim = 0.5 * specific_diag + 0.5 * common_diag
    
    # Compute average
    info_preservation = torch.mean(combined_sim).item()
    
    return info_preservation


def evaluate_downstream_task(model, dataloader, task_type="classification"):
    """
    Evaluate the MEDTOK model on a downstream task.
    
    Args:
        model: Model for the downstream task
        dataloader: Evaluation dataloader
        task_type: Type of downstream task ("classification" or "regression")
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Get predictions
            outputs = model(**batch)
            predictions = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Store predictions and labels
            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute metrics based on task type
    metrics = {}
    
    if task_type == "classification":
        # Binary classification
        if all_preds.shape[1] == 1 or (all_preds.shape[1] == 2 and np.all(all_labels < 2)):
            if all_preds.shape[1] == 2:
                all_preds = all_preds[:, 1]  # Probability of the positive class
            
            # Compute AUROC
            try:
                metrics["auroc"] = roc_auc_score(all_labels, all_preds)
            except:
                metrics["auroc"] = 0.0
            
            # Compute AUPRC
            try:
                metrics["auprc"] = average_precision_score(all_labels, all_preds)
            except:
                metrics["auprc"] = 0.0
            
            # Compute precision and recall at different thresholds
            precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
            
            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            metrics["best_threshold"] = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            metrics["best_precision"] = precision[best_idx]
            metrics["best_recall"] = recall[best_idx]
            metrics["best_f1"] = f1_scores[best_idx]
        
        # Multi-class classification
        else:
            # Convert predictions to class indices
            pred_classes = np.argmax(all_preds, axis=1)
            
            # Compute accuracy
            metrics["accuracy"] = np.mean(pred_classes == all_labels)
            
            # Compute macro-averaged AUROC
            try:
                if all_preds.shape[1] > 2:
                    metrics["macro_auroc"] = roc_auc_score(all_labels, all_preds, average='macro', multi_class='ovr')
                else:
                    metrics["macro_auroc"] = roc_auc_score(all_labels, all_preds[:, 1])
            except:
                metrics["macro_auroc"] = 0.0
    
    elif task_type == "regression":
        # Compute mean squared error
        metrics["mse"] = np.mean((all_preds - all_labels) ** 2)
        
        # Compute mean absolute error
        metrics["mae"] = np.mean(np.abs(all_preds - all_labels))
        
        # Compute R-squared
        ss_total = np.sum((all_labels - np.mean(all_labels)) ** 2)
        ss_residual = np.sum((all_labels - all_preds) ** 2)
        metrics["r2"] = 1 - (ss_residual / ss_total)
    
    return metrics


def compute_token_similarity(token_indices1, token_indices2, codebook):
    """
    Compute similarity between tokenized medical codes.
    
    Args:
        token_indices1: Token indices for the first code
        token_indices2: Token indices for the second code
        codebook: Codebook embeddings
    
    Returns:
        Similarity score
    """
    # Get token embeddings
    embeddings1 = codebook[token_indices1]
    embeddings2 = codebook[token_indices2]
    
    # Compute pairwise similarities
    similarities = torch.matmul(embeddings1, embeddings2.t())
    
    # Return maximum similarity
    return torch.max(similarities).item()


def compute_token_coverage(model, dataset):
    """
    Compute the coverage of tokens for medical codes.
    
    Args:
        model: The MEDTOK model
        dataset: Dataset of medical codes
    
    Returns:
        Token coverage statistics
    """
    model.eval()
    
    # Count token usage
    token_counts = {}
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            
            # Move to device
            input_ids = sample["input_ids"].unsqueeze(0).to(model.config.device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(model.config.device)
            graph_features = sample["graph_features"].to(model.config.device)
            graph_edge_index = sample["graph_edge_index"].to(model.config.device)
            
            # Create batch tensor for graph
            graph_batch = torch.zeros(graph_features.size(0), dtype=torch.long, device=model.config.device)
            
            # Tokenize
            token_indices = model.tokenize(
                input_ids,
                graph_features,
                graph_edge_index,
                graph_batch
            )
            
            # Count tokens
            for token_idx in token_indices[0].cpu().numpy():
                if token_idx not in token_counts:
                    token_counts[token_idx] = 0
                token_counts[token_idx] += 1
    
    # Compute statistics
    total_tokens = model.config.codebook_size
    used_tokens = len(token_counts)
    coverage = used_tokens / total_tokens
    
    # Compute token distribution statistics
    counts = np.array(list(token_counts.values()))
    mean_count = np.mean(counts)
    median_count = np.median(counts)
    max_count = np.max(counts)
    min_count = np.min(counts)
    
    return {
        "total_tokens": total_tokens,
        "used_tokens": used_tokens,
        "coverage": coverage,
        "mean_count": mean_count,
        "median_count": median_count,
        "max_count": max_count,
        "min_count": min_count
    }
