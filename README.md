// Copyright (c) 2025 tickyChan. All rights reserved.

// Usage restrictions

// This project is intended for **read-only** educational or informational purposes. The code is not licensed for use, copying, modification, distribution or any other use. All rights reserved.

// **Warning**: Any unauthorized use of the code violates copyright.

# Quark
Quark is a Transformer model implemented from scratch in C++, created for natural language processing (NLP) with a focus on Russian. The project does not use large machine learning libraries (like PyTorch or TensorFlow), relying only on minimal dependencies such as OpenBLAS for matrix operations.

(! Training was not carried out due to poor hardware, as a base there is only a file of embeddings. Development is also suspended!)

## Main Features
- **Complete Transformer Implementation**: An autoregressive Transformer model including Multi-Head Attention with causal masking, FeedForward, LayerNorm, Positional Encoding, and KV cache for efficient text generation.
- **Built from Scratch in C++**: All components (tensors, gradients, Adam optimizer) are implemented manually without relying on large machine learning frameworks.
- **Russian Language Support**: Integrated with a tokenizer optimized for Russian text, supporting special characters and mixed Russian-English input.
- **Data Storage**: The model is saved in JSON files. Support for other formats (e.g., RocksDB) is planned.


## Architecture
Quark implements the classic autoregressive Transformer architecture with the following components:
- Tensor: A class for working with tensors, supporting matrix operations (via OpenBLAS) and automatic differentiation.
- MultiHeadAttention: A multi-headed attention mechanism with a causal mask and a KV cache.
- FeedForward: A fully connected layer with GeLU activation.
- LayerNorm: Normalization of layers with trainable parameters (gamma, beta).
- PositionalEncoding: Sinusoidal positional encodings.
- AdamOptimizer: An Adam optimizer with gradient clipping.
- TransformerModel: The main class that combines components for training and text generation.
- TrainingSystem: A system for loading data and training a model with support for validation and early stopping.
