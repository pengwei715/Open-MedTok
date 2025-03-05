# MedTok Integration Guide

This guide explains how to integrate MedTok with existing EHR systems and transformer-based models.

## Table of Contents
- [Overview](#overview)
- [Integration with Transformer-Based EHR Models](#integration-with-transformer-based-ehr-models)
  - [BEHRT Integration](#behrt-integration)
  - [GT-BEHRT Integration](#gt-behrt-integration)
  - [TransformEHR Integration](#transformehr-integration)
  - [ETHOS Integration](#ethos-integration)
  - [Mult-EHR Integration](#mult-ehr-integration)
- [Integration with Medical Question-Answering Systems](#integration-with-medical-question-answering-systems)
- [Custom Integration](#custom-integration)
- [Performance Considerations](#performance-considerations)
- [Example Applications](#example-applications)

## Overview

MedTok is designed to be a drop-in replacement for traditional tokenizers in EHR systems. It produces token sequences that can be consumed by any model that works with discrete tokens, particularly transformer-based architectures.

The key advantages of integrating MedTok into your EHR system include:

1. **Richer Code Representations**: Combines textual and graph-based representations
2. **Better Handling of Rare Codes**: Rare codes benefit from shared code knowledge
3. **Cross-System Compatibility**: Bridges semantic gaps between different coding systems
4. **Performance Improvements**: Typically improves AUPRC by 4-11% on downstream tasks

## Integration with Transformer-Based EHR Models

### BEHRT Integration

[BEHRT](https://github.com/deepmedicine/BEHRT) is a transformer-based model for EHR modeling based on BERT architecture.

#### Basic Integration

```python
from model.medtok import MedTok
from integrations.behrt_integration import integrate_medtok_with_behrt

# Load pre-trained MedTok model
medtok_model = MedTok.load_from_checkpoint("path/to/medtok_model.pt")

# Configure BEHRT model
behrt_config = {
    "vocab_size": 30000,  # Will be overridden by MedTok's token space
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 512
}

# Integrate MedTok with BEHRT
integrated_model = integrate_medtok_with_behrt(medtok_model, behrt_config)

# Use the integrated model
inputs = {
    "codes": ["E11.9", "I10", "R94.31"],
    "descriptions": [
        "Type 2 diabetes mellitus without complications",
        "Essential (primary) hypertension",
        "Abnormal electrocardiogram"
    ],
    "graphs": [...],  # Graph representations for codes
    "age": 65,
    "gender": "M",
    "visit_segment_ids": [0, 0, 1]  # Visit segmentation
}

# Get predictions
outputs = integrated_model(**inputs)
```

#### Detailed Example

For a complete working example, see `integrations/behrt_integration.py` and `examples/behrt_integration_example.py`.

### GT-BEHRT Integration

GT-BEHRT extends BEHRT with graph transformer capabilities.

```python
from model.medtok import MedTok
from integrations.gt_behrt_integration import integrate_medtok_with_gt_behrt

# Load pre-trained MedTok model
medtok_model = MedTok.load_from_checkpoint("path/to/medtok_model.pt")

# Configure GT-BEHRT model
gt_behrt_config = {
    "vocab_size": 30000,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "graph_hidden_size": 128,
    "graph_num_layers": 3
}

# Integrate MedTok with GT-BEHRT
integrated_model = integrate_medtok_with_gt_behrt(medtok_model, gt_behrt_config)

# Use the integrated model (similar to BEHRT but with graph inputs)
```

### TransformEHR Integration

TransformEHR uses an encoder-decoder transformer architecture to model EHR data.

```python
from model.medtok import MedTok
from integrations.transformehr_integration import integrate_medtok_with_transformehr

# Load pre-trained MedTok model
medtok_model = MedTok.load_from_checkpoint("path/to/medtok_model.pt")

# Configure TransformEHR model
transformehr_config = {
    "vocab_size": 30000,
    "d_model": 768,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dim_feedforward": 2048,
    "dropout": 0.1
}

# Integrate MedTok with TransformEHR
integrated_model = integrate_medtok_with_transformehr(medtok_model, transformehr_config)
```

### ETHOS Integration

ETHOS tokenizes patient health timelines for transformer-based pretraining.

```python
from model.medtok import MedTok
from integrations.ethos_integration import integrate_medtok_with_ethos

# Load pre-trained MedTok model
medtok_model = MedTok.load_from_checkpoint("path/to/medtok_model.pt")

# Integrate MedTok with ETHOS
integrated_model = integrate_medtok_with_ethos(medtok_model)
```

### Mult-EHR Integration

Mult-EHR uses multi-task heterogeneous graph learning with causal denoising.

```python
from model.medtok import MedTok
from integrations.mult_ehr_integration import integrate_medtok_with_mult_ehr

# Load pre-trained MedTok model
medtok_model = MedTok.load_from_checkpoint("path/to/medtok_model.pt")

# Integrate MedTok with Mult-EHR
integrated_model = integrate_medtok_with_mult_ehr(medtok_model)
```

## Integration with Medical Question-Answering Systems

MedTok can enhance medical QA systems by providing better tokenization for medical codes:

```python
from model.medtok import MedTok
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load pre-trained MedTok model
medtok_model = MedTok.load_from_checkpoint("path/to/medtok_model.pt")

# Load QA model
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def process_qa_with_medtok(question, context):
    # Extract medical codes from context
    medical_codes = extract_medical_codes(context)
    
    # Tokenize medical codes with MedTok
    medtok_tokens = []
    for code, desc in medical_codes:
        code_tokens = medtok_model.tokenize(
            code=code,
            description=desc,
            graph=load_graph_for_code(code)
        )
        medtok_tokens.extend(code_tokens)
    
    # Create enhanced context
    enhanced_context = context + " [MED_TOKENS] " + " ".join(medtok_tokens)
    
    # Tokenize with regular tokenizer
    inputs = tokenizer(
        f"question: {question} context: {enhanced_context}",
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    # Generate answer
    outputs = qa_model.generate(**inputs, max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer
```

## Custom Integration

For custom integrations, you can access MedTok's functionality through the following interfaces:

### 1. Tokenizing Medical Codes

```python
# Single code tokenization
tokens = medtok_model.tokenize(
    code="E11.9",
    description="Type 2 diabetes mellitus without complications",
    graph=graph_data
)

# Batch tokenization
batch_tokens = medtok_model.tokenize_batch(
    codes=["E11.9", "I10"],
    descriptions=[
        "Type 2 diabetes mellitus without complications",
        "Essential (primary) hypertension"
    ],
    graphs=[graph_data_1, graph_data_2]
)
```

### 2. Getting Token Embeddings

```python
# Get token embeddings for downstream use
token_embeddings = medtok_model.get_token_embedding(tokens)
```

### 3. Creating Custom Preprocessing Pipeline

```python
def preprocess_patient_record(patient_data, medtok_model):
    # Extract medical codes and descriptions
    codes = [event["code"] for visit in patient_data["visits"] for event in visit["events"]]
    descriptions = [event["description"] for visit in patient_data["visits"] for event in visit["events"]]
    
    # Load graphs for each code
    graphs = [load_graph_for_code(code) for code in codes]
    
    # Tokenize all codes
    tokenized_codes = medtok_model.tokenize_batch(codes, descriptions, graphs)
    
    # Format for your model
    formatted_input = format_for_model(tokenized_codes, patient_data)
    
    return formatted_input
```

## Performance Considerations

When integrating MedTok, consider the following performance aspects:

### Computational Overhead

- **Inference Time**: MedTok adds approximately 50-100ms per code tokenization on CPU
- **Memory Usage**: The full model requires ~1-2GB RAM depending on configuration
- **GPU Acceleration**: Graph processing benefits significantly from GPU acceleration

### Optimization Techniques

1. **Batch Processing**: Always use batch tokenization for multiple codes
2. **Caching**: Cache tokenization results for frequently used codes
3. **Quantization**: Consider int8 quantization for deployment
4. **Sparse Integration**: In some cases, only tokenize important codes

Example caching implementation:

```python
class CachedMedTok:
    def __init__(self, medtok_model, cache_size=10000):
        self.medtok_model = medtok_model
        self.cache = {}
        self.cache_size = cache_size
    
    def tokenize(self, code, description, graph):
        # Create cache key
        cache_key = f"{code}_{description}_{hash(str(graph))}"
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Tokenize
        tokens = self.medtok_model.tokenize(code, description, graph)
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[cache_key] = tokens
        return tokens
```

## Example Applications

### 1. Enhanced EHR Prediction Model

```python
from model.medtok import MedTok
import torch.nn as nn

class EnhancedEHRModel(nn.Module):
    def __init__(self, medtok_model, hidden_size=768, num_classes=10):
        super().__init__()
        self.medtok = medtok_model
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size*4
            ),
            num_layers=6
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, patient_data):
        # Extract data
        codes = patient_data["codes"]
        descriptions = patient_data["descriptions"]
        graphs = patient_data["graphs"]
        
        # Tokenize all codes
        tokens = self.medtok.tokenize_batch(codes, descriptions, graphs)
        
        # Get token embeddings
        token_embeddings = self.medtok.get_token_embedding(tokens)
        
        # Apply transformer
        transformer_output = self.transformer(token_embeddings)
        
        # Global average pooling
        pooled = transformer_output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
```

### 2. Clinical Decision Support System

```python
def clinical_decision_support(patient_record, medtok_model, ehr_model):
    # Preprocess patient record using MedTok
    processed_input = preprocess_patient_record(patient_record, medtok_model)
    
    # Make predictions
    predictions = ehr_model(processed_input)
    
    # Generate clinical recommendations
    recommendations = generate_recommendations(predictions)
    
    return recommendations
```

### 3. Cohort Identification System

```python
def identify_cohort(patient_records, inclusion_criteria, medtok_model):
    matching_patients = []
    
    for patient_id, record in patient_records.items():
        # Tokenize patient record
        tokenized_record = preprocess_patient_record(record, medtok_model)
        
        # Check if patient meets criteria
        if matches_criteria(tokenized_record, inclusion_criteria):
            matching_patients.append(patient_id)
    
    return matching_patients
```

For more examples and detailed implementation guides, see the `examples/` directory.