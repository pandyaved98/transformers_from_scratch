# Transformer from Scratch

## Description
This project implements a Transformer model from scratch, following the architecture described in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762). The implementation includes enhancements such as Pre-Layer Normalization, relative positional embeddings, and mixed precision training. It is designed to be modular, easy to understand, and extensible for various natural language processing (NLP) tasks.

# Implemented Features in Transformer Project

## Core Features

### Full Transformer Architecture
- **Encoder-Decoder Structure:**
  - Implements the standard Transformer architecture with encoder-decoder structure.
  - Includes multi-head self-attention and position-wise feed-forward networks.

### Embeddings
- **Token Embeddings:**
  - Maps tokens to dense vector representations.
- **Absolute Positional Embeddings:**
  - Adds positional information to token embeddings.

### Attention Mechanism
- **Scaled Dot-Product Attention:**
  - Computes attention scores using scaled dot-product mechanism.
- **Multi-Head Attention:**
  - Enables multiple parallel attention heads.

### Training
- **Loss Function:**
  - Cross-entropy loss with label smoothing for better generalization.
- **Optimizer:**
  - Adam optimizer with a warm-up learning rate scheduler.

### Evaluation
- **BLEU Score:**
  - Calculates BLEU scores for evaluating text generation tasks.
- **Greedy Decoding:**
  - Implements basic greedy decoding for sequence generation.

### Beam Search
- **Basic Implementation:**
  - Supports sequence generation using beam search.

### Data Handling
- **Data Preprocessing:**
  - Tokenization and preparation of datasets.
- **Batching:**
  - Handles variable-length sequences with padding for efficient batching.

### Utilities
- **Model Saving and Loading:**
  - Save and load models during training or for inference.
- **Logging:**
  - Logs training progress and key metrics.

---

## Enhancements

### Pre-Layer Normalization (Pre-LN)
- **Normalization:**
  - Applies layer normalization before attention and feed-forward layers to improve training stability.

### Mixed Precision Training
- **FP16 Training:**
  - Uses `torch.cuda.amp` for faster training and reduced memory usage.

### Relative Positional Embeddings
- **Sequence Modeling:**
  - Provides optional support for relative positional embeddings for improved performance on sequence tasks.

### Gradient Checkpointing
- **Memory Efficiency:**
  - Reduces memory usage by recomputing intermediate activations during backpropagation.

### Attention Visualization
- **Interpretability:**
  - Visualizes attention weights using matplotlib and seaborn.

---

## File-Specific Features

### `config.py`
- Centralized configuration file for defining hyperparameters and settings.

### `embeddings.py`
- Implements token and positional embeddings.

### `attention.py`
- Multi-head self-attention with scaled dot-product attention.

### `feed_forward.py`
- Defines position-wise feed-forward networks.

### `encoder_layer.py`
- Single encoder layer with multi-head attention and feed-forward network.

### `decoder_layer.py`
- Single decoder layer with:
  - Masked multi-head attention.
  - Encoder-decoder attention.
  - Feed-forward network.

### `encoder.py`
- Full encoder stack with multiple encoder layers.

### `decoder.py`
- Full decoder stack with multiple decoder layers.

### `transformer.py`
- Combines encoder and decoder into the full Transformer model.

### `optimizer.py`
- Adam optimizer with a warm-up learning rate scheduler.

### `loss.py`
- Implements cross-entropy loss with label smoothing.

### `data_loader.py`
- Handles data preprocessing, tokenization, and batching.

### `utils.py`
- Utility functions for saving/loading models and logging.

### `train.py`
- Training loop with:
  - Mixed precision training.
  - Gradient checkpointing.

### `evaluate.py`
- Evaluation logic with:
  - BLEU score calculation.
  - Greedy decoding.

### `beam_search.py`
- Implements beam search for sequence generation.

### `metrics.py`
- BLEU score calculation using NLTK.

### `visualization.py`
- Visualizes attention weights for better interpretability.

### `pretrain.py`
- Contains routines for pretraining tasks, such as masked language modeling.

### `main.py`
- Entry point for running training and evaluation tasks.

# Planned Enhancements

## Model Improvements

### Attention Mechanisms
- Implement **sparse attention** (e.g., Longformer, BigBird) to reduce memory usage for long sequences.
- Add **linear attention** (e.g., Performer, Linformer) to reduce the quadratic complexity of self-attention.
- Implement **grouped query attention (GQA)** or **multi-query attention (MQA)** for faster inference.

### Positional Embeddings
- Add support for **rotary positional embeddings (RoPE)**.
- Implement **learnable positional embeddings** instead of fixed ones.

### Architecture Variants
- Add support for **Transformer-XL** to handle longer sequences.
- Implement **Reformer** for memory-efficient attention.

### Layer Improvements
- Add **residual connections with scaling** for better gradient flow.
- Implement **gradient checkpointing** for memory-efficient training of deeper models.

---

## Training Optimizations

### Mixed Precision Training
- Fully integrate **FP16 training** with gradient scaling for faster training and reduced memory usage.

### Dynamic Batching
- Implement **dynamic batching** to handle variable-length sequences efficiently.

### Learning Rate Schedulers
- Add more **learning rate schedulers** (e.g., cosine annealing, reduce on plateau).

### Regularization
- Add **dropout** and **weight decay** for better generalization.
- Implement **gradient clipping** to prevent exploding gradients.

### Data Augmentation
- Add **back-translation** for sequence-to-sequence tasks.
- Implement **token masking and shuffling** for pretraining.

---

## Evaluation Enhancements

### Metrics
- Add **ROUGE** and **METEOR** scores for better evaluation of text generation tasks.
- Implement **perplexity** as a metric for language modeling tasks.

### Beam Search Improvements
- Add **length normalization** and **repetition penalty** for better sequence generation.
- Implement **diverse beam search** for more diverse outputs.

### Attention Visualization
- Add functionality to **visualize attention weights** for interpretability.

### Error Analysis
- Add tools for analyzing model errors (e.g., **misclassified examples, attention patterns**).

---

## Usability Features

### Command-Line Interface (CLI)
- Add a **CLI** for training, evaluation, and inference using `argparse` or `Click`.

### Configuration Management
- Replace `config.py` with a configuration management library like **Hydra** or **YAML** for easier hyperparameter tuning.

### Logging and Monitoring
- Integrate **TensorBoard** or **Weights & Biases (W&B)** for logging training metrics and visualizing attention weights.

### Pretraining Support
- Add support for **masked language modeling (MLM)** and **causal language modeling (CLM)**.

### Deployment
- Add support for exporting the model to **ONNX** or **TorchScript** for deployment in production environments.

---

## Quick Wins (Easy to Implement)
- **Gradient Clipping:** Add gradient clipping to prevent exploding gradients.
- **Top-k and Top-p Sampling:** Implement top-k sampling and top-p (nucleus) sampling for diverse text generation.
- **TensorBoard Logging:** Integrate TensorBoard to log training loss, validation loss, and attention weights.
- **Dynamic Batching:** Modify the data loader to group sequences of similar lengths for efficient batching.
- **Docstrings:** Add docstrings to all functions and classes for better code readability.

---

## Prioritization

### Quick Wins
- Gradient clipping.
- Top-k sampling.
- TensorBoard logging.

### Model Improvements
- Sparse attention.
- Rotary positional embeddings.

### Training Optimizations
- Mixed precision training.
- Dynamic batching.

### Evaluation Enhancements
- ROUGE and METEOR scores.
- Beam search improvements.
