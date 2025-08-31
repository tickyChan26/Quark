// Copyright (c) 2025 tickyChan. All rights reserved.

## Usage restrictions
This project is intended for **read-only** educational or informational purposes. The code is not licensed for use, copying, modification, distribution or any other use. All rights reserved.
**Warning**: Any unauthorized use of the code violates copyright.

# Quark
Quark is a Transformer model implemented from scratch in C++, created for natural language processing (NLP) with a focus on Russian. The project does not use large machine learning libraries (like PyTorch or TensorFlow), relying only on minimal dependencies such as OpenBLAS for matrix operations.

(! Training was not carried out due to poor hardware, as a base there is only a file of embeddings. Development is also suspended!)

## Main features
- Full implementation of Transformer: Includes Multi-Head Attention, FeedForward, LayerNorm, Positional Encoding and KV-cache for efficient generation.
- From scratch in C++: All components (tensors, gradients, Adam optimizer) are implemented manually without using large frameworks.
- Language: Integration with a tokenizer for processing Russian-language texts. The ability to learn Russian with various special characters and English letters is available.
- Data storage: The model is saved in JSON files for convenience. Support for other formats (for example, RocksDB) is planned in the future.

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
