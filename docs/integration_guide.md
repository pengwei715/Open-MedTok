# MEDTOK Integration Guide

This guide explains how to integrate the MEDTOK tokenizer with various Electronic Health Record (EHR) models and systems.

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Integration Pattern](#basic-integration-pattern)
3. [Integration with EHR Models](#integration-with-ehr-models)
   - [ETHOS Integration](#ethos-integration)
   - [GT-BEHRT Integration](#gt-behrt-integration)
   - [MulT-EHR Integration](#mult-ehr-integration)
   - [TransformEHR Integration](#transformehr-integration)
   - [BEHRT Integration](#behrt-integration)
4. [Integration with Medical QA Systems](#integration-with-medical-qa-systems)
5. [Using MEDTOK with Custom Models](#using-medtok-with-custom-models)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Introduction

MEDTOK is a multimodal medical code tokenizer that integrates textual descriptions and graph-based representations from biomedical knowledge graphs. It can be used to enhance various EHR models by replacing their standard tokenization approaches with MEDTOK's more informative token representations.

The key advantages of using MEDTOK include:

1. Better representation of medical codes through multimodal integration
2. Preservation of ontological relationships between codes
3. Enhanced semantic understanding of medical concepts
4. Improved performance on downstream tasks like mortality prediction, readmission prediction, etc.

## Basic Integration Pattern

The basic pattern for integrating MEDTOK with any model involves:

1. **Initialize MEDTOK**: Load a pre-trained MEDTOK model.
2. **Process Medical Codes**: Tokenize medical codes in the patient records.
3. **Replace Embeddings**: Use MEDTOK token embeddings in place of standard embeddings.
4. **Fine-tune or Train**: Fine-tune the integrated model on the downstream task.

Here's a general code snippet for this pattern:

```python
import torch
from model.medtok import MedTok
from utils.config import MedTokConfig

# Step 1: Load MEDTOK
model_path = "path/to/trained/medtok.pt"
checkpoint = torch.load(model_path, map_location="cuda")
config = checkpoint["config"]
medtok = MedTok(config)
medtok.load_state_dict(checkpoint["model_state_dict"])
medtok.eval()

# Step 2: Process Medical Codes
def process_patient_record(patient_record, medtok):
    processed_visits = []
    
    for visit in patient_record:
        processed_codes = []
        
        for code in visit["codes"]:
            # Tokenize the code
            token_indices = tokenize_medical_code(
                medtok,
                code["code_id"],
                code["description"],
                code["graph_file"]
            )
            
            # Get token embeddings
            token_embeddings = medtok.get_token_embedding(token_indices)
            
            processed_codes.append({
                "code_id": code["code_id"],
                "token_indices": token_indices,
                "token_embeddings": token_embeddings
            })
        
        processed_visits.append({
            "visit_id": visit["visit_id"],
            "date": visit["date"],
            "codes": processed_codes
        })
    
    return processed_visits

# Step 3: Replace standard embeddings in your model
# (This depends on the specific model being integrated)
```

## Integration with EHR Models

### ETHOS Integration

[ETHOS](https://github.com/rencp/ethos) is a transformer-based model for tokenizing patient health timelines. To integrate MEDTOK with ETHOS:

```python
from ethos.model import ETHOS
from ethos.data import load_data

# Load ETHOS model
ethos_model = ETHOS.from_pretrained("ethos-base")

# Replace ETHOS code embeddings with MEDTOK
def integrate_medtok_with_ethos(ethos_model, medtok, data):
    # Process data with MEDTOK
    medtok_embeddings = {}
    
    for patient_id, patient_data in data.items():
        for visit in patient_data["visits"]:
            for code in visit["codes"]:
                if code["code_id"] not in medtok_embeddings:
                    # Tokenize code with MEDTOK
                    token_indices = tokenize_medical_code(
                        medtok,
                        code["code_id"],
                        code["description"],
                        f"data/graphs/{code['code_id']}.json"
                    )
                    
                    # Get token embeddings (average across tokens)
                    token_embedding = medtok.get_token_embedding(token_indices).mean(dim=1)
                    
                    medtok_embeddings[code["code_id"]] = token_embedding
    
    # Replace ETHOS code embeddings with MEDTOK embeddings
    ethos_model.update_code_embeddings(medtok_embeddings)
    
    return ethos_model
```

See `integrations/ethos_integration.py` for a complete implementation.

### GT-BEHRT Integration

[GT-BEHRT](https://github.com/bhanushashank/GT-BEHRT) (Graph Transformer Bidirectional Encoder Representations from Transformers) models intra-visit dependencies as a graph. To integrate MEDTOK with GT-BEHRT:

```python
# Assuming GT-BEHRT model is already loaded
from gt_behrt.model import GTBEHRT

class MedtokGTBEHRT(GTBEHRT):
    def __init__(self, medtok_model, config):
        super().__init__(config)
        
        # Store MEDTOK model
        self.medtok = medtok_model
        
        # Freeze MEDTOK
        for param in self.medtok.parameters():
            param.requires_grad = False
    
    def forward(self, patient_data, attention_mask=None):
        # Process visits with MEDTOK
        visit_embeddings = []
        
        for visit in patient_data["visits"]:
            # Tokenize codes in the visit using MEDTOK
            token_embeddings = []
            
            for code_data in visit["codes"]:
                # Get token indices and embeddings
                token_indices = code_data["token_indices"]
                token_embedding = self.medtok.get_token_embedding(token_indices)
                token_embeddings.append(token_embedding)
            
            # Create graph from visit codes
            # (This follows GT-BEHRT's approach for building a visit graph)
            # ...
            
            # Apply visit-level transformer
            visit_embedding = self.visit_transformer(token_embeddings, visit_graph)
            visit_embeddings.append(visit_embedding)
        
        # Continue with standard GT-BEHRT forward pass
        # ...
        
        return predictions
```

See `integrations/gt_behrt_integration.py` for a complete implementation.

### MulT-EHR Integration

[MulT-EHR](https://github.com/tong-wu-umn/MulT-EHR) uses multi-task heterogeneous graph learning. To integrate MEDTOK:

```python
# Add MEDTOK embeddings to MulT-EHR's node features
def enhance_mult_ehr_with_medtok(mult_ehr, medtok, data):
    # For each node in the heterogeneous graph
    for node_type in mult_ehr.node_types:
        if node_type in ["diagnosis", "procedure", "medication"]:
            for node_id in mult_ehr.nodes[node_type]:
                code_id = mult_ehr.node_id_to_code[node_type][node_id]
                
                # Tokenize with MEDTOK
                token_indices = tokenize_medical_code(
                    medtok,
                    code_id,
                    mult_ehr.code_descriptions[code_id],
                    f"data/graphs/{code_id}.json"
                )
                
                # Get token embeddings
                token_embedding = medtok.get_token_embedding(token_indices).mean(dim=1)
                
                # Replace or concatenate with original node features
                original_features = mult_ehr.node_features[node_type][node_id]
                enhanced_features = torch.cat([original_features, token_embedding], dim=-1)
                
                mult_ehr.node_features[node_type][node_id] = enhanced_features
    
    # Adjust node feature dimensions in the model
    mult_ehr.update_feature_dimensions()
    
    return mult_ehr
```

### TransformEHR Integration

[TransformEHR](https://github.com/zyin3/transformehr) adopts an encoder-decoder transformer with visit-level masking. To integrate MEDTOK:

```python
# Replace TransformEHR's code embeddings with MEDTOK
def enhance_transformehr_with_medtok(transformehr, medtok, data):
    # Create a mapping from code IDs to MEDTOK embeddings
    code_to_embedding = {}
    
    for patient in data:
        for visit in patient["visits"]:
            for code in visit["codes"]:
                if code["code_id"] not in code_to_embedding:
                    # Tokenize with MEDTOK
                    token_indices = tokenize_medical_code(
                        medtok,
                        code["code_id"],
                        code["description"],
                        f"data/graphs/{code['code_id']}.json"
                    )
                    
                    # Get token embeddings (average across tokens)
                    token_embedding = medtok.get_token_embedding(token_indices).mean(dim=1)
                    
                    code_to_embedding[code["code_id"]] = token_embedding
    
    # Replace code embeddings in TransformEHR
    transformehr.code_embeddings.weight.data = torch.stack(
        [code_to_embedding[code_id] for code_id in transformehr.code_to_idx]
    )
    
    return transformehr
```

### BEHRT Integration

[BEHRT](https://github.com/deepmedicine/BEHRT) applies deep bidirectional learning for EHR data. To integrate MEDTOK:

```python
# Replace BEHRT's code embeddings with MEDTOK
def enhance_behrt_with_medtok(behrt, medtok, data):
    # Create a mapping from code IDs to MEDTOK embeddings
    code_to_embedding = {}
    
    for patient in data:
        for visit in patient["visits"]:
            for code in visit["codes"]:
                if code["code_id"] not in code_to_embedding:
                    # Tokenize with MEDTOK
                    token_indices = tokenize_medical_code(
                        medtok,
                        code["code_id"],
                        code["description"],
                        f"data/graphs/{code['code_id']}.json"
                    )
                    
                    # Get token embeddings (average across tokens)
                    token_embedding = medtok.get_token_embedding(token_indices).mean(dim=1)
                    
                    code_to_embedding[code["code_id"]] = token_embedding
    
    # Replace code embeddings in BEHRT
    behrt.embeddings.word_embeddings.weight.data[:len(behrt.code_to_idx)] = torch.stack(
        [code_to_embedding.get(behrt.idx_to_code.get(i, "UNK"), 
                              behrt.embeddings.word_embeddings.weight.data[i]) 
         for i in range(len(behrt.code_to_idx))]
    )
    
    return behrt
```

## Integration with Medical QA Systems

MEDTOK can be integrated with medical question-answering systems to enhance their understanding of medical codes. Here's how to integrate MEDTOK with an LLM-based medical QA system:

```python
# Add MEDTOK prefix tokens to LLM prompts
def enhance_llm_with_medtok(llm_model, medtok, question, context):
    # Extract medical codes from the question and context
    codes = extract_medical_codes(question, context)
    
    # Generate MEDTOK tokens for each code
    medtok_tokens = []
    
    for code in codes:
        token_indices = tokenize_medical_code(
            medtok,
            code["code_id"],
            code["description"],
            f"data/graphs/{code['code_id']}.json"
        )
        
        medtok_tokens.append({
            "code_id": code["code_id"],
            "tokens": token_indices
        })
    
    # Create prompt with MEDTOK context
    medtok_context = "Medical context:\n"
    for code_tokens in medtok_tokens:
        medtok_context += f"- Code {code_tokens['code_id']} has been tokenized with MEDTOK.\n"
    
    enhanced_prompt = f"{medtok_context}\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:"
    
    # Generate response with LLM
    response = llm_model.generate(enhanced_prompt)
    
    return response
```

See `benchmarks/evaluate_medqa.py` for a complete implementation.

## Using MEDTOK with Custom Models

To integrate MEDTOK with a custom EHR model:

1. **Tokenize Medical Codes**: Use MEDTOK to tokenize all medical codes in your dataset.

```python
from scripts.tokenize_code import tokenize_medical_code

def tokenize_dataset_codes(data, medtok):
    tokenized_codes = {}
    
    for code_id, description in data["codes"].items():
        graph_file = f"data/graphs/{code_id}.json"
        
        # Tokenize the code
        token_indices = tokenize_medical_code(
            medtok,
            code_id,
            description,
            graph_file
        )
        
        tokenized_codes[code_id] = token_indices
    
    return tokenized_codes
```

2. **Integrate Tokenized Codes**: Replace standard embedding lookup in your model with MEDTOK token embeddings.

```python
class CustomEHRModel(nn.Module):
    def __init__(self, medtok, config):
        super().__init__()
        
        self.medtok = medtok
        self.config = config
        
        # Define your model architecture
        self.transformer = nn.TransformerEncoder(...)
        self.classifier = nn.Linear(...)
    
    def forward(self, patient_data):
        # Use MEDTOK token embeddings
        visit_embeddings = []
        
        for visit in patient_data:
            code_embeddings = []
            
            for code in visit["codes"]:
                # Get token embeddings
                token_indices = code["token_indices"]
                token_embeddings = self.medtok.get_token_embedding(token_indices)
                
                # Aggregate token embeddings for the code
                code_embedding = token_embeddings.mean(dim=1)
                code_embeddings.append(code_embedding)
            
            # Aggregate code embeddings for the visit
            visit_embedding = torch.stack(code_embeddings).mean(dim=0)
            visit_embeddings.append(visit_embedding)
        
        # Process visit embeddings with transformer
        sequence = torch.stack(visit_embeddings)
        transformed = self.transformer(sequence)
        
        # Make prediction
        prediction = self.classifier(transformed.mean(dim=0))
        
        return prediction
```

3. **Train or Fine-tune**: Train your custom model with the MEDTOK-enhanced representations.

## Best Practices

When integrating MEDTOK with EHR models, consider these best practices:

1. **Freeze MEDTOK**: In most cases, you should freeze the MEDTOK model to preserve its learned representations.

2. **Aggregation Strategy**: Choose an appropriate strategy for aggregating token embeddings:
   - Mean pooling: Simple but effective for many cases
   - Attention-weighted pooling: Better when some tokens are more important
   - Hierarchical aggregation: For complex structures (e.g., visits → codes → tokens)

3. **Dimensionality Alignment**: Ensure that the dimensions of MEDTOK embeddings are compatible with your model's expected input dimensions.

4. **Preprocessing**: Preprocess all medical codes in your dataset with MEDTOK before training to improve efficiency.

5. **Memory Efficiency**: For large datasets, consider storing tokenized representations on disk to avoid repeating tokenization.

6. **Handling Missing Graphs**: Implement a fallback strategy for codes without associated graph files.

## Troubleshooting

Common issues and solutions when integrating MEDTOK:

1. **Memory Issues**: 
   - Problem: MEDTOK tokenization requires loading both text and graph data, which can be memory-intensive.
   - Solution: Process codes in smaller batches and use memory-efficient data loaders.

2. **Missing Graph Files**:
   - Problem: Some medical codes might not have corresponding graph files.
   - Solution: Create dummy graph representations or fall back to text-only tokenization.

3. **Dimension Mismatch**:
   - Problem: MEDTOK embeddings might have different dimensions than the original model's embeddings.
   - Solution: Add projection layers to align dimensions or redesign the model's architecture to accommodate the new dimensions.

4. **Performance Degradation**:
   - Problem: In some cases, integrating MEDTOK might initially degrade performance.
   - Solution: Try different aggregation strategies, adjust learning rates, or use curriculum learning to gradually incorporate MEDTOK embeddings.

5. **Integration Complexity**:
   - Problem: Some models have complex architecture that makes integration difficult.
   - Solution: Consider creating a wrapper model that uses MEDTOK as a preprocessing step, rather than modifying the model's internal architecture.
