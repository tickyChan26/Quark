// Copyright (c) 2025 tickyChan. All rights reserved.

// Usage restrictions

// This project is intended for **read-only** educational or informational purposes. The code is not licensed for use, copying, modification, distribution or any other use. All rights reserved.

// **Warning**: Any unauthorized use of the code violates copyright.

# Quark
Quark is a Transformer model implemented from scratch in C++, created for natural language processing (NLP) with a focus on Russian. The project does not use large machine learning libraries (like PyTorch or TensorFlow), relying only on minimal dependencies such as OpenBLAS for matrix operations.

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

## Project Status
Development is currently paused and training has not been conducted due to limited hardware resources. The project is a skeleton. The code may contain inaccuracies since it is still in development.

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


## Main Features
- **Complete Transformer Implementation**: An autoregressive Transformer model including Multi-Head Attention with causal masking, FeedForward, LayerNorm, Positional Encoding, and KV cache for efficient text generation.
- **Built from Scratch in C++**: All components (tensors, gradients, Adam optimizer) are implemented manually without relying on large machine learning frameworks.
- **Russian Language Support**: Integrated with a tokenizer optimized for Russian text, supporting special characters and mixed Russian-English input.
- **Data Storage**: The model is saved in JSON files. Support for other formats (e.g., RocksDB) is planned.


## Architecture
Quark implements an autoregressive Transformer architecture with the following components:
- **Tensor**: A class for tensor operations, supporting matrix computations (via OpenBLAS) and automatic differentiation for gradient-based training.
- **MultiHeadAttention**: Multi-Head Attention mechanism with causal masking and KV cache for efficient generation.
- **FeedForward**: Fully connected layer with GeLU activation.
- **LayerNorm**: Layer normalization with trainable parameters (gamma, beta).
- **PositionalEncoding**: Sinusoidal positional encodings for sequence order.
- **AdamOptimizer**: Adam optimizer with gradient clipping.
- **TransformerModel**: Main class integrating components for training and text generation.
- **TrainingSystem**: System for data loading, training, validation, and early stopping.
