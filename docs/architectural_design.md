# MedTok Architectural Design

This document explains the architectural design of MedTok, including its components, data flow, and internal mechanisms.

## System Overview

MedTok is a multimodal tokenizer for medical codes that combines text descriptions with graph-based representations from biomedical knowledge graphs. It enables more effective encoding of Electronic Health Records (EHRs) by preserving both textual and relational information about medical codes.

### Key Features

- **Multimodal Encoding**: Combines text and graph modalities
- **Vector Quantization**: Maps continuous embeddings to discrete tokens
- **Cross-modality Learning**: Preserves both modality-specific and shared information
- **Token Packing**: Optimizes token representations for efficiency and expressivity

## System Architecture

![MedTok Architecture](../assets/medtok_architecture.png)

MedTok consists of five main components:

1. **Text Encoder**: Processes medical code text descriptions
2. **Graph Encoder**: Processes code relationships in knowledge graphs
3. **Vector Quantizer**: Maps embeddings to discrete token space
4. **Token Packer**: Optimizes token representations
5. **Integration Interfaces**: Connects with EHR models and systems

### 1. Text Encoder

The text encoder processes medical code descriptions using pre-trained language models.

**Implementation**: `model/text_encoder.py`

#### Variants:
- **Default Text Encoder**: Uses BERT's [CLS] token for sentence representation
- **Weighted Pooling Text Encoder**: Uses attention-weighted token representations

```python
# Example architecture (simplified)
class TextEncoder(nn.Module):
    def __init__(self, model_name, embedding_dim, pooling_type):
        self.transformer = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_size, embedding_dim)
        self.pooling_type = pooling_type
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(input_ids, attention_mask)
        
        # Apply appropriate pooling
        if self.pooling_type == "cls":
            pooled = outputs.last_hidden_state[:, 0]
        elif self.pooling_type == "mean":
            pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling_type == "weighted":
            pooled = weighted_pooling(outputs.last_hidden_state, attention_mask)
            
        # Project to desired dimension
        projected = self.projection(pooled)
        return projected
```

### 2. Graph Encoder

The graph encoder processes the biomedical knowledge graph context of each medical code.

**Implementation**: `model/graph_encoder.py`

#### Variants:
- **Default Graph Encoder**: Uses GCN for encoding graph structure
- **GAT Graph Encoder**: Uses Graph Attention Networks for edge-weighted encoding
- **Hierarchical Graph Encoder**: Uses hierarchical pooling for multi-scale representations

```python
# Example architecture (simplified)
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            
        self.layers.append(GCNConv(hidden_dim, output_dim))
        self.pool = global_mean_pool
        
    def forward(self, x, edge_index, batch):
        # Apply GCN layers
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=0.1, training=self.training)
            
        x = self.layers[-1](x, edge_index)
        
        # Pool to graph-level representations
        pooled = self.pool(x, batch)
        return pooled
```

### 3. Vector Quantizer

The vector quantizer maps continuous embeddings to discrete tokens using a learnable codebook.

**Implementation**: `model/vector_quantizer.py`

#### Key Components:
- **Multi-part Codebook**: Divides the codebook into text-specific, graph-specific, and shared regions
- **Top-K Selection**: Selects top-K nearest codebook vectors for each embedding
- **Straight-through Estimator**: Enables gradient flow through the discrete bottleneck

```python
# Example architecture (simplified)
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim, num_codebooks=1):
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.num_codebooks = num_codebooks
        
    def forward(self, z):
        # Calculate distances to all codebook vectors
        distances = torch.cdist(z, self.codebook.weight)
        
        # Get indices of nearest codebook vectors
        _, indices = torch.topk(distances, k=self.num_codebooks, largest=False)
        
        # Get quantized vectors
        z_q = self.codebook(indices)
        
        # Compute loss
        q_loss = F.mse_loss(z_q, z.detach())
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, indices, q_loss
```

### 4. Token Packer

The token packer optimizes token representations by balancing modality-specific and shared information.

**Implementation**: `model/token_packer.py`

#### Key Functions:
- **Information Preservation**: Ensures tokens preserve modality-specific information
- **Cross-modality Learning**: Enhances shared information between modalities
- **Redundancy Reduction**: Minimizes redundant information between tokens

```python
# Example architecture (simplified)
class TokenPacker(nn.Module):
    def __init__(self, embedding_dim, beta=0.2):
        self.beta = beta
        
    def forward(self, text_embed, graph_embed, e_text_s, e_graph_s, e_text_c, e_graph_c):
        # Compute KL divergence for cross-attention
        kl_loss = compute_kl_divergence(e_text_c, e_graph_c)
        
        # Compute common token packing loss
        common_loss = compute_common_loss(e_text_c, e_graph_c, self.beta)
        
        # Compute specific token packing loss
        specific_loss = compute_specific_loss(e_text_s, e_graph_s, e_text_c, e_graph_c)
        
        # Total packing loss
        packing_loss = common_loss + specific_loss + kl_loss
        
        return {
            'token_packing_loss': packing_loss,
            'common_loss': common_loss,
            'specific_loss': specific_loss,
            'kl_loss': kl_loss
        }
```

### 5. MedTok Main Model

The main MedTok model integrates all components and provides the tokenization interface.

**Implementation**: `model/medtok.py`

```python
# Example architecture (simplified)
class MedTok(nn.Module):
    def __init__(self, text_encoder, graph_encoder, vector_quantizer, token_packer):
        self.text_encoder = text_encoder
        self.graph_encoder = graph_encoder
        self.vector_quantizer = vector_quantizer
        self.token_packer = token_packer
        
    def forward(self, text_data, graph_data):
        # 1. Extract embeddings from text and graph modalities
        text_embedding = self.text_encoder(*text_data)
        graph_embedding = self.graph_encoder(*graph_data)
        
        # 2. Generate modality-specific and cross-modality embeddings
        e_text_s, e_graph_s, e_text_c, e_graph_c = self.generate_embeddings(
            text_embedding, graph_embedding)
        
        # 3. Quantize embeddings to tokens
        z_text_s, indices_text_s, loss_text_s = self.vector_quantizer(e_text_s)
        z_graph_s, indices_graph_s, loss_graph_s = self.vector_quantizer(e_graph_s)
        z_text_c, indices_text_c, loss_text_c = self.vector_quantizer(e_text_c)
        z_graph_c, indices_graph_c, loss_graph_c = self.vector_quantizer(e_graph_c)
        
        # 4. Compute token packing loss
        packing_losses = self.token_packer(
            text_embedding, graph_embedding, 
            e_text_s, e_graph_s, e_text_c, e_graph_c)
        
        # 5. Combine token indices
        token_indices = torch.cat([
            indices_text_s, indices_graph_s, indices_text_c, indices_graph_c
        ], dim=1)
        
        # 6. Compute total loss
        quantization_loss = loss_text_s + loss_graph_s + loss_text_c + loss_graph_c
        packing_loss = packing_losses['token_packing_loss']
        total_loss = quantization_loss + packing_loss
        
        return token_indices, {
            'quantization_loss': quantization_loss,
            'packing_loss': packing_loss,
            'total_loss': total_loss
        }
        
    def tokenize(self, code, description, graph):
        # Process inputs and run forward pass in inference mode
        # Return token indices
        ...
        
    def get_token_embedding(self, token_id):
        # Retrieve token embedding from codebook
        return self.vector_quantizer.codebook(token_id)
```

## Data Flow

![MedTok Data Flow](../assets/medtok_data_flow.png)

### Training Flow

1. **Input Processing**:
   - Text descriptions are tokenized using the BERT tokenizer
   - Graphs are formatted as node features and edge indices

2. **Forward Pass**:
   - Text descriptions → Text Encoder → Text Embeddings
   - Graph data → Graph Encoder → Graph Embeddings
   - Text/Graph Embeddings → Cross-attention → Modality-specific & Shared Embeddings
   - Embeddings → Vector Quantizer → Quantized Vectors & Token Indices
   - All Embeddings → Token Packer → Packing Loss
   - All Losses → Combined → Total Loss

3. **Backward Pass**:
   - Total Loss → Backward → Parameter Updates
   - Straight-through estimator enables gradient flow through discrete bottleneck

### Inference Flow

1. **Input Processing**:
   - Process text description and graph for the medical code

2. **Encoding & Tokenization**:
   - Text descriptions → Text Encoder → Text Embeddings
   - Graph data → Graph Encoder → Graph Embeddings
   - Text/Graph Embeddings → Cross-attention → Modality-specific & Shared Embeddings
   - Embeddings → Vector Quantizer → Token Indices
   - Combine token indices from all modalities → Final Token Sequence

3. **Token Utilization**:
   - Token sequences can be used by downstream EHR models
   - Token embeddings can be retrieved from the codebook for continuous representations

## Modal Interactions

MedTok facilitates three types of modal interactions:

1. **Text-specific**: Captures information unique to the text description
2. **Graph-specific**: Captures information unique to the relational context
3. **Cross-modal**: Captures information shared between text and graph modalities

### Cross-attention Mechanism

Cross-attention allows each modality to attend to the other:

```
e_text_c = softmax(W_q * text_embed * (W_k * graph_embed)^T / sqrt(d)) * (W_v * graph_embed)
e_graph_c = softmax(W_q * graph_embed * (W_k * text_embed)^T / sqrt(d)) * (W_v * text_embed)
```

This enables the model to extract information that is common across modalities.

## Loss Functions

MedTok uses multiple loss functions:

1. **Codebook Loss**: Ensures embeddings are close to codebook vectors
   ```
   L_codebook = ||sg[e_hat] - e||^2 + β||e_hat - sg[e]||^2
   ```

2. **KL Divergence Loss**: Aligns text and graph token distributions
   ```
   L_KL = D_KL(softmax(-dist(e_t^c, C)) || softmax(-dist(e_g^c, C)))
   ```

3. **Token Packing Loss**: Balances modality-specific and shared information
   ```
   L_token = L_token^c + L_token^s
   ```

4. **Total Loss**: Combines all losses for optimization
   ```
   L = L_codebook + L_KL + L_token
   ```

## Integration with EHR Models

MedTok can be integrated with various EHR models:

1. **Tokenization Interface**:
   ```python
   tokens = medtok.tokenize(code, description, graph)
   ```

2. **Token Embedding Retrieval**:
   ```python
   token_embeddings = medtok.get_token_embedding(token_ids)
   ```

3. **Batch Processing**:
   ```python
   batch_tokens = medtok.tokenize_batch(codes, descriptions, graphs)
   ```

Integration examples are provided for multiple EHR models, including BEHRT, TransformEHR, GT-BEHRT, ETHOS, and Mult-EHR.

## References

- MedTok Paper: "Multimodal Medical Code Tokenizer" by Su et al.
- Vector Quantization: "Neural Discrete Representation Learning" by van den Oord et al.
- Cross-modal Learning: "Learning Transferable Visual Models From Natural Language Supervision" by Radford et al.
- Tokenization: "Improving Language Understanding by Generative Pre-Training" by Radford et al.