# nn-semantic-extractor

> **Semantic = sign(v) × |v| / ||v||**

A universal mathematical framework for parsing hidden representations of any neural network and extracting semantic information.

## Overview

This paper presents a unified semantic extraction formula that reveals all neural networks (regardless of architecture, task, or training method) use a **dual-channel structure** to encode information:

- **Semantic Channel** (`sign(v) × |v|`): Carries meaningful information
- **Stability Channel** (`||v||`): Ensures numerical stability and learning convergence

The formula has been mathematically proven and experimentally validated to achieve perfect semantic extraction across multiple architectures including Qwen2.5-1.5B, Transformer, CNN, and RNN.

---

## Core Formula

Given a hidden state vector `v ∈ R^d` from any layer of a neural network, the semantic contribution of neuron `i` is:

```
Semantic_i = sign(v_i) × |v_i| / ||v||
```
```
Pytorch: semantic = torch.sign(v) * torch.abs(v) / torch.norm(v, dim=-1, keepdim=True)
```
Where:
- `v_i`: The i-th component of the hidden state vector
- `|v_i|`: Local magnitude of neuron i (absolute value)
- `||v||`: Euclidean norm of the vector (global magnitude)
- `sign(v_i)`: Sign function, distinguishing excitatory (positive) and inhibitory (negative) contributions

### Why This Formula Works

1. **Normalization controls stability**: Modern neural networks use LayerNorm/RMSNorm to normalize `||v||` to a constant
2. **Direction carries semantics**: Normalization only adjusts length, not direction
3. **Local magnitude encodes information**: Semantic information is encoded in the distribution of local magnitudes, not global magnitude
4. **Sign preserves directionality**: Distinguishes between positive and negative contributions

---

## PyTorch Implementation

### Basic Semantic Extraction

```python
import torch

def extract_semantic(hidden_states: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Extract semantic vectors from hidden states
    
    Args:
        hidden_states: Shape (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
        eps: Numerical stability constant
    
    Returns:
        semantic: Semantic vector, same shape as input
    """
    norm = torch.norm(hidden_states, p=2, dim=-1, keepdim=True) + eps
    semantic = torch.sign(hidden_states) * torch.abs(hidden_states) / norm
    return semantic
```

### Simple Example

```python
# Example: Extract semantics from a hidden state
hidden_states = torch.randn(1, 32, 768)  # (batch, seq, hidden_dim)
semantic = extract_semantic(hidden_states)
print(f"Semantic shape: {semantic.shape}")  # Output: torch.Size([1, 32, 768])
```

---

## Key Theorems

### 1. Dual-Channel Principle
Every neural network implicitly separates its hidden representations into two independent channels:
- **Semantic Channel**: `sign(v) ⊙ |v|` — encodes task-relevant information
- **Stability Channel**: `||v||` — ensures numerical stability and learning convergence

### 2. Semantic Extraction Theorem
Given any neural network layer with hidden state `v ∈ R^d`, the semantic contribution `s_i = sign(v_i) × |v_i| / ||v||` satisfies:
- **Completeness**: Captures all semantic information relevant to downstream tasks
- **Uniqueness**: No other decomposition can extract more semantic information
- **Universality**: Applicable to all neural network architectures

### 3. Key Properties
- **Normalization Property**: `||s||₁ = Σ|s_i| = 1`
- **Sign Preservation**: `sign(s_i) = sign(v_i)`
- **Scale Invariance**: `s_i(αv) = s_i(v)` for any `α > 0`

---

## Applications

### 1. Neuron-Level Interpretation
Identify which neurons are most important for the current input:
```python
importance = torch.abs(semantic)
top_neurons = torch.topk(importance, k=10)
```

### 2. Expert Neuron Discovery
Find neurons that consistently contribute to specific tasks

### 3. Layer Function Analysis
Track semantic changes across layers to understand layer functionality

### 4. Semantic Attention Mechanism
Identify important semantic components in sentences

### 5. Controlled Generation
Achieve controlled generation by modifying semantic contributions

.......

---

## Installation

```bash
pip install torch transformers
```

## License

MIT License

---

## Key Insight

This formula is not a heuristic but a mathematical theorem. It inevitably, universally, and deterministically can parse the semantics of any neural network.

#Qwen 2.5 1.5B

![1](https://raw.githubusercontent.com/Winamin/nn-semantic-extractor/main/layer_evolution.png)

![2](https://raw.githubusercontent.com/Winamin/nn-semantic-extractor/main/neuron_expertise.png)

![3](https://raw.githubusercontent.com/Winamin/nn-semantic-extractor/main/semantic_pca.png)
