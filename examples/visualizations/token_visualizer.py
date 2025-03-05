#!/usr/bin/env python
"""
Token Visualization Tool for MedTok

This script visualizes medical code tokenization results from MedTok,
showing both text-specific and graph-specific tokens and their relationships.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.medtok import MedTok

# Create custom colormaps for visualization
text_cmap = LinearSegmentedColormap.from_list("text_cmap", ["#2979FF", "#00BCD4"])
graph_cmap = LinearSegmentedColormap.from_list("graph_cmap", ["#F44336", "#FF9800"])
shared_cmap = LinearSegmentedColormap.from_list("shared_cmap", ["#9C27B0", "#E91E63"])


def load_model(model_path):
    """Load a MedTok model from file"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    
    try:
        model = MedTok.load_from_checkpoint(model_path)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def tokenize_codes(model, codes_file):
    """Tokenize a set of medical codes using the provided model"""
    if not os.path.exists(codes_file):
        print(f"Error: Codes file not found: {codes_file}")
        return None
    
    try:
        with open(codes_file, 'r') as f:
            codes_data = json.load(f)
        
        results = []
        for item in codes_data:
            code = item['code']
            description = item.get('description', '')
            graph_path = item.get('graph_path', None)
            
            # Load graph if provided
            graph = None
            if graph_path and os.path.exists(graph_path):
                try:
                    with open(graph_path, 'r') as f:
                        graph = json.load(f)
                except:
                    print(f"Warning: Could not load graph for {code} from {graph_path}")
            
            # Tokenize with model
            tokens = model.tokenize(code, description, graph)
            
            # Get token embeddings
            token_embeddings = []
            for token_id in tokens:
                token_embeddings.append(model.get_token_embedding(token_id).detach().numpy())
            
            results.append({
                'code': code,
                'description': description,
                'tokens': tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens,
                'token_embeddings': token_embeddings,
                'modality_info': model.get_token_modality_info(tokens)
            })
        
        return results
    
    except Exception as e:
        print(f"Error tokenizing codes: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def visualize_tokens_2d(tokenization_results, output_dir, method="tsne"):
    """Create 2D visualizations of token embeddings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all token embeddings
    all_embeddings = []
    all_modalities = []
    code_labels = []
    
    for result in tokenization_results:
        code = result['code']
        modality_info = result['modality_info']
        
        for i, embedding in enumerate(result['token_embeddings']):
            all_embeddings.append(embedding)
            all_modalities.append(modality_info[i]['type'])
            code_labels.append(code)
    
    all_embeddings = np.array(all_embeddings)
    
    # Reduce dimensionality
    if method == "tsne":
        print("Applying t-SNE for dimensionality reduction...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    else:
        print("Applying PCA for dimensionality reduction...")
        reducer = PCA(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot points by modality
    for modality in ['text_specific', 'graph_specific', 'shared']:
        mask = np.array([m == modality for m in all_modalities])
        if not any(mask):
            continue
            
        points = reduced_embeddings[mask]
        labels = np.array(code_labels)[mask]
        
        # Choose color map based on modality
        if modality == 'text_specific':
            cmap = text_cmap
            marker = 'o'
            alpha = 0.8
            label = 'Text-specific tokens'
        elif modality == 'graph_specific':
            cmap = graph_cmap
            marker = 's'  # square
            alpha = 0.8
            label = 'Graph-specific tokens'
        else:  # shared
            cmap = shared_cmap
            marker = '^'  # triangle
            alpha = 0.9
            label = 'Shared tokens'
        
        plt.scatter(
            points[:, 0], points[:, 1],
            c=np.arange(len(points)),
            cmap=cmap,
            marker=marker,
            s=100,
            alpha=alpha,
            label=label
        )
        
        # Add code labels to some points
        for i, (x, y) in enumerate(points):
            if np.random.random() < 0.3:  # Only label a subset to avoid clutter
                txt = plt.text(x, y, labels[i], fontsize=8)
                txt.set_path_effects([
                    PathEffects.withStroke(linewidth=3, foreground='white')
                ])
    
    plt.legend(fontsize=12)
    plt.title(f'2D Token Visualization ({method.upper()})', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'token_visualization_{method}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    return output_path


def visualize_code_token_distribution(tokenization_results, output_dir):
    """Visualize token distribution across modalities for each code"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    codes = []
    text_tokens = []
    graph_tokens = []
    shared_tokens = []
    
    for result in tokenization_results:
        code = result['code']
        modality_info = result['modality_info']
        
        # Count tokens by modality
        text_count = sum(1 for item in modality_info if item['type'] == 'text_specific')
        graph_count = sum(1 for item in modality_info if item['type'] == 'graph_specific')
        shared_count = sum(1 for item in modality_info if item['type'] == 'shared')
        
        codes.append(code)
        text_tokens.append(text_count)
        graph_tokens.append(graph_count)
        shared_tokens.append(shared_count)
    
    # Create stacked bar chart
    fig, ax = plt.figure(figsize=(12, len(codes)*0.5 + 2)), plt.subplot(111)
    
    width = 0.8
    ind = np.arange(len(codes))
    
    p1 = ax.barh(ind, text_tokens, width, color='#2979FF', label='Text-specific')
    p2 = ax.barh(ind, graph_tokens, width, left=text_tokens, color='#F44336', 
                 label='Graph-specific')
    p3 = ax.barh(ind, shared_tokens, width, 
                 left=np.array(text_tokens) + np.array(graph_tokens), 
                 color='#9C27B0', label='Shared')
    
    ax.set_yticks(ind)
    ax.set_yticklabels(codes)
    ax.set_xlabel('Token Count')
    ax.set_title('Token Distribution by Modality for Each Medical Code')
    ax.legend()
    
    # Add count labels
    for i, (x1, x2, x3) in enumerate(zip(text_tokens, graph_tokens, shared_tokens)):
        if x1 > 0:
            ax.text(x1/2, i, str(x1), ha='center', va='center', color='white')
        if x2 > 0:
            ax.text(x1 + x2/2, i, str(x2), ha='center', va='center', color='white')
        if x3 > 0:
            ax.text(x1 + x2 + x3/2, i, str(x3), ha='center', va='center', color='white')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'token_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Token distribution visualization saved to {output_path}")
    return output_path


def create_html_report(tokenization_results, visualization_paths, output_dir):
    """Create an HTML report with all visualizations and token details"""
    os.makedirs(output_dir, exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedTok Tokenization Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #333; }
        .visualization { margin: 30px 0; text-align: center; }
        .visualization img { max-width: 100%; height: auto; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .text-token { background-color: rgba(41, 121, 255, 0.2); }
        .graph-token { background-color: rgba(244, 67, 54, 0.2); }
        .shared-token { background-color: rgba(156, 39, 176, 0.2); }
        .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>MedTok Tokenization Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>This report shows tokenization results for ${num_codes} medical codes using the MedTok model.</p>
    </div>
    
    <div class="visualizations">
        <h2>Visualizations</h2>
"""
    
    # Add visualizations
    for path, title in visualization_paths:
        rel_path = os.path.basename(path)
        html_content += f"""
        <div class="visualization">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
        </div>
"""
    
    # Add token details
    html_content += """
    <h2>Token Details by Medical Code</h2>
    <table>
        <tr>
            <th>Code</th>
            <th>Description</th>
            <th>Tokens (Token ID: Modality)</th>
            <th>Total Token Count</th>
        </tr>
"""
    
    # Add rows for each code
    for result in tokenization_results:
        code = result['code']
        description = result['description']
        tokens = result['tokens']
        modality_info = result['modality_info']
        
        # Format tokens with their modality
        token_display = []
        for i, token_id in enumerate(tokens):
            modality = modality_info[i]['type']
            css_class = {
                'text_specific': 'text-token',
                'graph_specific': 'graph-token',
                'shared': 'shared-token'
            }.get(modality, '')
            
            token_display.append(f'<span class="{css_class}">{token_id}: {modality}</span>')
        
        tokens_str = ', '.join(token_display)
        
        html_content += f"""
        <tr>
            <td>{code}</td>
            <td>{description}</td>
            <td>{tokens_str}</td>
            <td>{len(tokens)}</td>
        </tr>
"""
    
    # Close HTML
    html_content += """
    </table>
</body>
</html>
"""
    
    # Replace placeholder with actual count
    html_content = html_content.replace('${num_codes}', str(len(tokenization_results)))
    
    # Write HTML file
    output_path = os.path.join(output_dir, 'tokenization_report.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    # Copy visualization files to output directory if they're not already there
    for path, _ in visualization_paths:
        if os.path.dirname(path) != output_dir:
            import shutil
            shutil.copy2(path, output_dir)
    
    print(f"HTML report generated at {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize MedTok tokenization results")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to trained MedTok model")
    parser.add_argument("--codes", type=str, required=True, 
                        help="JSON file containing medical codes to tokenize")
    parser.add_argument("--output", type=str, default="visualization_output", 
                        help="Output directory for visualizations")
    parser.add_argument("--method", type=str, choices=["tsne", "pca"], default="tsne",
                        help="Dimensionality reduction method")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    if model is None:
        return
    
    # Tokenize codes
    print(f"Tokenizing codes from {args.codes}...")
    results = tokenize_codes(model, args.codes)
    if results is None:
        return
    
    print(f"Tokenized {len(results)} medical codes")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create visualizations
    visualizations = []
    
    # 2D visualization of token embeddings
    print("Creating 2D token visualization...")
    vis_path = visualize_tokens_2d(results, args.output, method=args.method)
    visualizations.append((vis_path, f"2D Token Visualization ({args.method.upper()})"))
    
    # Token distribution by modality
    print("Creating token distribution visualization...")
    dist_path = visualize_code_token_distribution(results, args.output)
    visualizations.append((dist_path, "Token Distribution by Modality"))
    
    # Generate HTML report
    print("Generating HTML report...")
    report_path = create_html_report(results, visualizations, args.output)
    
    print(f"\nVisualization complete! Open {report_path} to view the results.")


if __name__ == "__main__":
    main()