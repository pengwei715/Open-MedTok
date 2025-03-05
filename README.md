# MEDTOK: Multimodal Medical Code Tokenizer

MEDTOK is a multimodal tokenizer for medical codes that improves encoding of Electronic Health Records (EHRs) by combining textual descriptions with graph-based representations from biomedical knowledge graphs.

## Overview

Foundation models trained on patient electronic health records (EHRs) require tokenizing medical data into sequences of discrete vocabulary items. Existing tokenizers treat medical codes from EHRs as isolated textual tokens. However, each medical code is defined by its textual description, its position in ontological hierarchies, and its relationships to other codes, such as disease co-occurrences and drug-treatment associations.

MEDTOK processes text using a language model encoder and encodes the relational structure with a graph encoder. It then quantizes both modalities into a unified token space, preserving modality-specific and cross-modality information.

## Features

- **Multimodal Tokenization**: Combines text descriptions and graph-based relationships for rich code representations
- **Vector Quantization**: Maps continuous embeddings to discrete tokens for use with transformer models
- **Cross-Modality Learning**: Preserves both modality-specific and shared information
- **Token Packing**: Optimizes token representations to balance representational power and efficiency
- **Integration-Ready**: Easily integrates with existing EHR models and systems

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- Transformers
- NetworkX
- Pandas
- NumPy
- tqdm

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medtok.git
cd medtok
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

## Getting Started

### Quick Start

1. Prepare the data (small sample for testing):
   ```bash
   python scripts/preprocess_pipeline.py --output data/ --codes_only
   ```

2. Train a basic model (small scale for testing):
   ```bash
   python train.py \
       --data_dir data/ \
       --output_dir output/ \
       --batch_size 8 \
       --epochs 5 \
       --codebook_size 1000 \
       --embedding_dim 64
   ```

3. Tokenize a medical code with your trained model:
   ```bash
   python examples/tokenize_example.py \
       --model output/checkpoints/last_model.pt \
       --code "E11.9" \
       --desc "Type 2 diabetes mellitus without complications"
   ```

## Hardware Requirements

### Minimum Requirements
- CPU: 8+ cores recommended
- RAM: 16GB+ (32GB+ recommended for large datasets)
- Storage: 100GB+ for datasets and model checkpoints
- GPU: NVIDIA GPU with 8GB+ VRAM (for training)

### Recommended Specifications
- GPU: NVIDIA A100, V100, or RTX 3090/4090 (16GB+ VRAM)
- RAM: 64GB+ for processing large biomedical knowledge graphs
- Storage: 500GB+ SSD for datasets, knowledge graphs, and model checkpoints

### Training Time Estimates
- Full training on 1x A100 GPU: ~12 hours
- Full training on 1x V100 GPU: ~24 hours
- Full training on 1x RTX 3090: ~30 hours
- Inference time per code: ~50ms on GPU, ~200ms on CPU

## Data Requirements

### Required Datasets

To fully reproduce the results from the paper, you'll need access to the following datasets:

1. **Medical Code Databases**
   - ICD-9-CM and ICD-10-CM codes: Available from [CMS.gov](https://www.cms.gov/Medicare/Coding/ICD10)
   - SNOMED CT: Requires license from [SNOMED International](https://www.snomed.org/)
   - RxNorm: Available from [NLM](https://www.nlm.nih.gov/research/umls/rxnorm/index.html)
   - ATC: Available from [WHO Collaborating Centre](https://www.whocc.no/atc_ddd_index/)
   - CPT: Requires license from the American Medical Association
   - NDC: Available from [FDA NDC Directory](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory)

2. **Biomedical Knowledge Graph**
   - PrimeKG: Download from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM)
   - Or use alternative biomedical knowledge graphs like UMLS (requires license)

3. **EHR Datasets for Evaluation**
   - MIMIC-III: Available from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) (requires credentialing)
   - MIMIC-IV: Available from [PhysioNet](https://physionet.org/content/mimiciv/1.0/) (requires credentialing)
   - EHRShot: Available from [EHRShot GitHub](https://github.com/som-shahlab/ehrshot-benchmark)

4. **Medical QA Datasets (for additional evaluation)**
   - MedDDx dataset: Used in the paper for medical QA evaluation

### Dataset Sizes
- PrimeKG: ~9GB
- MIMIC-III: ~6GB
- MIMIC-IV: ~14GB
- EHRShot: ~2GB
- Medical code databases: ~500MB combined

### Data Download Instructions

1. **MIMIC Datasets**:
   - Register on [PhysioNet](https://physionet.org/) and complete the required training
   - Request access to the MIMIC datasets
   - Download using the provided scripts:
     ```bash
     python scripts/download_mimic.py --version mimic3 --username YOUR_USERNAME --prepare
     ```

2. **PrimeKG**:
   - Download from Harvard Dataverse:
     ```bash
     python scripts/download_primekg.py --output data/primekg
     ```

3. **Medical Codes**:
   - Use the provided download scripts:
     ```bash
     python scripts/download_medical_codes.py --output data/medical_codes
     ```
     
4. **Training Your Own Models**:
   - As this is a recent research paper implementation, you'll need to train models from scratch:
     ```bash
     # Follow the training instructions below after data preparation
     python train.py --data_dir data/dataset --output_dir output/
     ```

## Data Preparation

MEDTOK requires two types of data for each medical code:

1. **Text Descriptions**: Textual definitions of each medical code
2. **Graph Representations**: Subgraphs from biomedical knowledge graphs

### Option 1: Unified Preprocessing Pipeline

The simplest approach is to use the unified preprocessing pipeline:

```bash
# Run the complete preprocessing pipeline
python scripts/preprocess_pipeline.py --output data/ --full

# Or just process medical codes only
python scripts/preprocess_pipeline.py --output data/ --codes_only
```

### Option 2: Step-by-Step Preprocessing

If you prefer manual control over the preprocessing steps:

```bash
# Process medical codes
python scripts/process_medical_codes.py --input data/medical_codes --output data/processed_codes

# Prepare text descriptions
python data/preprocessing/text_processor.py --input data/processed_codes/codes.csv --output data/processed_descriptions.csv

# Prepare graph data
python data/preprocessing/graph_processor.py --input data/processed_codes/codes.csv --kg_path data/primekg/kg.json --output_dir data/graphs/
```

## Training

To train the MEDTOK tokenizer:

```bash
python train.py \
    --data_dir data/ \
    --output_dir output/ \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --codebook_size 12000 \
    --embedding_dim 64 \
    --text_encoder_model "bert-base-uncased" \
    --text_encoder_type "weighted_pooling" \
    --graph_encoder_type "hierarchical"
```

### Training Arguments

- `--data_dir`: Directory containing the dataset files
- `--output_dir`: Directory to save model checkpoints and logs
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--codebook_size`: Size of the token vocabulary
- `--embedding_dim`: Dimension of token embeddings
- `--text_encoder_model`: Pretrained text encoder model
- `--text_encoder_type`: Type of text encoder ("default" or "weighted_pooling")
- `--graph_encoder_type`: Type of graph encoder ("default", "gat", or "hierarchical")
- `--num_workers`: Number of dataloader workers
- `--device`: Device to use ("cuda" or "cpu")

## Inference

To tokenize medical codes using a trained MEDTOK model:

```bash
python infer.py \
    --model_path output/checkpoints/best_model.pt \
    --input_file data/test_codes.csv \
    --output_file tokenized_codes.json \
    --graph_dir data/graphs \
    --batch_size 16
```

### Inference Arguments

- `--model_path`: Path to the trained model
- `--input_file`: Input file with medical codes (CSV or JSON)
- `--output_file`: Output file to save tokenized codes (CSV or JSON)
- `--graph_dir`: Directory with graph files
- `--batch_size`: Batch size for inference
- `--device`: Device to use ("cuda" or "cpu")

## Visualization

MedTok includes visualization tools to better understand token representations:

```bash
# Visualize tokenization results
python examples/visualizations/token_visualizer.py \
    --model output/checkpoints/best_model.pt \
    --codes examples/sample_codes.json \
    --output visualization_output \
    --method tsne
```

This generates an HTML report with interactive visualizations of token distributions and embeddings, making it easier to interpret the multimodal token representations.

## Benchmarking and Evaluation

To reproduce the results from the paper, you can use the provided evaluation scripts:

```bash
# Evaluate on MIMIC-III
python benchmarks/evaluate_mimic3.py \
    --model_path output/checkpoints/best_model.pt \
    --mimic_dir data/mimic-iii \
    --output_dir results/mimic3

# Evaluate on MIMIC-IV
python benchmarks/evaluate_mimic4.py \
    --model_path output/checkpoints/best_model.pt \
    --mimic_dir data/mimic-iv \
    --output_dir results/mimic4

# Evaluate on EHRShot
python benchmarks/evaluate_ehrshot.py \
    --model_path output/checkpoints/best_model.pt \
    --ehrshot_dir data/ehrshot \
    --output_dir results/ehrshot
```

### Evaluation Tasks

The benchmark scripts evaluate MEDTOK on the following tasks:

1. **MIMIC-III & MIMIC-IV Tasks**:
   - Mortality Prediction (MT)
   - Readmission Prediction (RA, <15 days)
   - Length-of-Stay Prediction (LOS)
   - Phenotype Prediction
   - Drug Recommendation

2. **EHRShot Tasks**:
   - Operational Outcomes (Long LOS, RA <15 days, MT)
   - Assignment of New Diagnoses (Hypertension, Hyperlipidemia, Pancreatic Cancer, Acute MI)

### Comparing with Baselines

To compare MEDTOK with baseline tokenizers:

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

### Expected Results

The expected performance improvements (AUPRC) compared to baseline EHR models:

- **MIMIC-III**: +4.10% improvement
- **MIMIC-IV**: +4.78% improvement
- **EHRShot**: +11.30% improvement

## Integration with EHR Models

MEDTOK can be integrated with various EHR models. The repository includes integration examples for:

- ETHOS
- GT-BEHRT
- MulT-EHR 
- TransformEHR
- BEHRT

Example integration:

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

See the `examples/ehr_integration_example.py` script for a detailed demonstration of how to integrate MEDTOK with a simple transformer-based EHR model.

## Medical QA Integration

To evaluate MEDTOK's performance on medical question answering tasks:

```bash
python benchmarks/evaluate_medqa.py \
    --medtok_model output/checkpoints/best_model.pt \
    --llm_model "llama3.1-8b" \
    --dataset medddx \
    --output_dir results/medqa
```

## Citation

```
@article{medtok2024,
  title={Multimodal Medical Code Tokenizer},
  author={Su, Xiaorui and Messica, Shvat and Huang, Yepeng and Johnson, Ruth and Fesser, Lukas and Gao, Shanghua and Sahneh, Faryad and Zitnik, Marinka},
  journal={arXiv preprint arXiv:2502.04397},
  year={2024}
}
```

## Testing

MedTok includes comprehensive test suites to ensure code quality and correct functionality:

```bash
# Run all tests with pytest
pytest tests/

# Run only unit tests
pytest tests/ -k "not integration"

# Run only integration tests
pytest tests/ -m "integration"

# Run with coverage reporting
pytest tests/ --cov=model --cov=data

# Or use the test runner script
python scripts/run_all_tests.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This implementation is based on the paper "Multimodal Medical Code Tokenizer" by Su et al. (2024). We thank the authors for their research that made this implementation possible.
