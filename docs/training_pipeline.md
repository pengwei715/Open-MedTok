# MedTok Training Pipeline

This document explains the entire training pipeline for MedTok, from data preparation to model evaluation and testing.

## Overview

The MedTok training pipeline consists of several sequential stages:

1. Data preparation
2. Model configuration
3. Training
4. Evaluation
5. Inference

This guide will walk you through each stage, explaining the necessary commands and expected outcomes.

## Hardware Requirements

Before starting, ensure your hardware meets these minimum specifications:

- **CPU**: 8+ cores recommended
- **RAM**: 32GB+ recommended (16GB minimum)
- **GPU**: NVIDIA GPU with 8GB+ VRAM for training
- **Storage**: 100GB+ for datasets and model checkpoints

Training times vary based on your hardware:
- Full training on 1x A100 GPU: ~12 hours
- Full training on 1x V100 GPU: ~24 hours
- Full training on 1x RTX 3090: ~30 hours

## 1. Data Preparation

### 1.1 Unified Preprocessing Pipeline

The simplest approach is to use our unified preprocessing pipeline:

```bash
# Run the complete preprocessing pipeline
python scripts/preprocess_pipeline.py --output data/ --full
```

This will:
- Download medical code databases
- Download the PrimeKG knowledge graph
- Process medical code descriptions
- Generate subgraphs for each medical code
- Create training, validation, and test splits

### 1.2 Manual Step-by-Step Preprocessing

If you prefer more control, you can run each step individually:

```bash
# 1. Download medical codes
python scripts/download_medical_codes.py --output data/medical_codes

# 2. Download PrimeKG knowledge graph
python scripts/download_primekg.py --output data/primekg

# 3. Process medical codes
python scripts/process_medical_codes.py --input data/medical_codes --output data/processed_codes

# 4. Process text descriptions
python data/preprocessing/text_processor.py \
    --input data/processed_codes/codes.csv \
    --output data/processed_descriptions.csv

# 5. Process graph data
python data/preprocessing/graph_processor.py \
    --input data/processed_codes/codes.csv \
    --kg_path data/primekg/kg.json \
    --output_dir data/graphs/
```

### 1.3 MIMIC Datasets (Optional)

If you want to evaluate on MIMIC datasets, you'll need PhysioNet credentials:

```bash
# Download and prepare MIMIC-III
python scripts/download_mimic.py --version mimic3 --username YOUR_USERNAME --prepare

# Download and prepare MIMIC-IV
python scripts/download_mimic.py --version mimic4 --username YOUR_USERNAME --prepare
```

This requires approved access to PhysioNet and may take several hours.

### 1.4 Verifying Data

Check your data preparation results:

```bash
# Check number of processed codes
wc -l data/processed_codes/codes.csv

# Check number of descriptions
wc -l data/processed_descriptions.csv

# Check number of graph files
find data/graphs -name "*.json" | wc -l

# Check dataset splits
head -n 5 data/dataset/train.json
head -n 5 data/dataset/val.json
head -n 5 data/dataset/test.json
```

Expected output:
- ~600,000 processed medical codes
- ~600,000 text descriptions
- ~600,000 graph files
- Dataset split in approximately 80/10/10 ratio

## 2. Model Configuration

MedTok has multiple configuration options. The default configuration is in `configs/default_config.json`:

```json
{
  "codebook_size": 12000,
  "embedding_dim": 768,
  "text_encoder_model": "bert-base-uncased",
  "text_encoder_type": "weighted_pooling",
  "graph_encoder_type": "hierarchical",
  "num_top_k_tokens": 4,
  "text_specific_ratio": 0.3,
  "graph_specific_ratio": 0.3,
  "shared_ratio": 0.4,
  "alpha": 0.25,
  "beta": 0.2,
  "lambda_val": 0.1
}
```

### 2.1 Key Configuration Parameters

- **codebook_size**: Total vocabulary size (12,000 recommended, 21,000 for larger datasets)
- **embedding_dim**: Dimension of token embeddings (768 recommended)
- **text_encoder_model**: Pre-trained text encoder model (bert-base-uncased recommended)
- **text_encoder_type**: Type of text encoder ("weighted_pooling" or "default")
- **graph_encoder_type**: Type of graph encoder ("hierarchical", "gat", or "default")
- **num_top_k_tokens**: Number of tokens to select per modality (4 recommended)
- **text_specific_ratio**: Portion of codebook dedicated to text-specific tokens
- **graph_specific_ratio**: Portion of codebook dedicated to graph-specific tokens
- **shared_ratio**: Portion of codebook dedicated to shared tokens
- **alpha**: Weight for vector quantization loss
- **beta**: Weight for token diversity loss
- **lambda_val**: Weight for token specificity loss

You can create custom configurations by modifying this file or passing parameters directly to the training script.

## 3. Training

### 3.1 Basic Training Command

```bash
python train.py \
    --data_dir data/dataset \
    --output_dir output/ \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --codebook_size 12000 \
    --embedding_dim 768 \
    --text_encoder_model "bert-base-uncased" \
    --text_encoder_type "weighted_pooling" \
    --graph_encoder_type "hierarchical"
```

### 3.2 Multi-GPU Training

For faster training with multiple GPUs:

```bash
python train.py \
    --data_dir data/dataset \
    --output_dir output/ \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --codebook_size 12000 \
    --embedding_dim 768 \
    --distributed \
    --num_gpus 4
```

### 3.3 Training with Smaller Datasets

For testing or systems with limited resources:

```bash
python train.py \
    --data_dir data/dataset \
    --output_dir output/ \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 5 \
    --codebook_size 1000 \
    --embedding_dim 64 \
    --sample_size 10000  # Use only 10K samples
```

### 3.4 Resuming Training

To resume from a checkpoint:

```bash
python train.py \
    --data_dir data/dataset \
    --output_dir output/ \
    --resume_from output/checkpoints/last_model.pt \
    --epochs 100  # Will train for 100 more epochs
```

### 3.5 Monitoring Training

Training progress is logged using TensorBoard:

```bash
# View training metrics
tensorboard --logdir output/logs
```

Key metrics to monitor:
- **total_loss**: Overall training loss
- **quantization_loss**: Vector quantization loss
- **packing_loss**: Token packing loss
- **kl_loss**: KL divergence loss
- **codebook_utilization**: Percentage of codebook being used
- **reconstruction_error**: Error in reconstructing original embeddings

## 4. Evaluation

### 4.1 Basic Evaluation

Evaluate your trained model:

```bash
# Evaluate tokenization quality
python evaluate.py \
    --model_path output/checkpoints/best_model.pt \
    --test_data data/dataset/test.json \
    --output_dir results/
```

### 4.2 Benchmarking on MIMIC

If you downloaded MIMIC datasets:

```bash
# Evaluate on MIMIC-III
python benchmarks/evaluate_mimic3.py \
    --model_path output/checkpoints/best_model.pt \
    --mimic_dir data/mimic3 \
    --output_dir results/mimic3

# Evaluate on MIMIC-IV  
python benchmarks/evaluate_mimic4.py \
    --model_path output/checkpoints/best_model.pt \
    --mimic_dir data/mimic4 \
    --output_dir results/mimic4
```

### 4.3 Benchmarking on EHRShot

```bash
python benchmarks/evaluate_ehrshot.py \
    --model_path output/checkpoints/best_model.pt \
    --ehrshot_dir data/ehrshot \
    --output_dir results/ehrshot
```

### 4.4 Comparing with Baselines

To compare with baseline tokenizers:

```bash
# Compare with BERT tokenizer  
python benchmarks/compare_tokenizers.py \
    --medtok_path output/checkpoints/best_model.pt \
    --baseline bert \
    --dataset mimic3 \
    --task mortality

# Compare with VQGraph
python benchmarks/compare_tokenizers.py \
    --medtok_path output/checkpoints/best_model.pt \
    --baseline vqgraph \
    --dataset ehrshot \
    --task operational_outcomes
```

## 5. Inference and Tokenization

### 5.1 Basic Tokenization

Tokenize individual medical codes:

```bash
python examples/tokenize_example.py \
    --model output/checkpoints/best_model.pt \
    --code "E11.9" \
    --desc "Type 2 diabetes mellitus without complications" \
    --verbose
```

### 5.2 Batch Tokenization

Tokenize a batch of medical codes:

```bash
python infer.py \
    --model_path output/checkpoints/best_model.pt \
    --input_file data/test_codes.csv \
    --output_file tokenized_codes.json \
    --graph_dir data/graphs \
    --batch_size 16
```

### 5.3 Visualization

Visualize tokenization results:

```bash
python examples/visualizations/token_visualizer.py \
    --model output/checkpoints/best_model.pt \
    --codes examples/sample_codes.json \
    --output visualization_output \
    --method tsne
```

This generates an HTML report with interactive visualizations of token distributions and embeddings.

## 6. Integration with EHR Models

MedTok can be integrated with various EHR models:

```bash
# Integrate with ETHOS
python integrations/ethos_integration.py \
    --medtok_model output/checkpoints/best_model.pt \
    --mimic_dir data/mimic-iv \
    --output_dir results/ethos_medtok

# Integrate with GT-BEHRT
python integrations/gt_behrt_integration.py \
    --medtok_model output/checkpoints/best_model.pt \
    --mimic_dir data/mimic-iii \
    --output_dir results/gt_behrt_medtok
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size (`--batch_size 8` or lower)
   - Reduce embedding dimension (`--embedding_dim 256` or lower)
   - Reduce codebook size (`--codebook_size 5000`)
   - Use gradient accumulation (`--grad_accum_steps 4`)

2. **Slow Training**
   - Use a smaller subset of data for testing (`--sample_size 10000`)
   - Reduce number of epochs (`--epochs 5`)
   - Use mixed precision training (`--mixed_precision`)

3. **Poor Performance**
   - Check if all data was processed correctly
   - Increase training epochs (`--epochs 100`)
   - Try different hyperparameters (`--lr 5e-5` or `--codebook_size 21000`)
   - Use weighted pooling for text encoder (`--text_encoder_type weighted_pooling`)
   - Use hierarchical graph encoder (`--graph_encoder_type hierarchical`)

## Advanced Configuration

For advanced users, you can modify the source files:
- `model/medtok.py`: Main MedTok model implementation
- `model/text_encoder.py`: Text encoder implementations
- `model/graph_encoder.py`: Graph encoder implementations
- `model/vector_quantizer.py`: Vector quantizer implementation
- `model/token_packer.py`: Token packing implementation

## Next Steps

Once you have a trained MedTok model, you can:
1. Use it to tokenize medical codes in your EHR pipeline
2. Integrate it with transformer-based EHR models
3. Analyze token distributions and patterns
4. Fine-tune it for specific medical specialties or tasks

For more specific use cases, refer to our other documentation:
- [Integration Guide](integration_guide.md) for integrating with existing EHR systems
- [Tokenization Examples](../examples/README.md) for code examples