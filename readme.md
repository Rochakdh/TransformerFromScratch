# Transformer-based English-Italian Translation Model

This project implements a Transformer architecture from scratch using PyTorch for bilingual machine translation. It supports training and inference on sentence pairs from the `opus_books` dataset (English → Italian) with custom tokenizers and configurable training parameters. The implementation was guided by the youtube tutorial at **Coding a Transformer from scratch on PyTorch, with full explanation, training and inference.**

## Project Structure

- `model.py` – Implements the Transformer model, including:
  - Multi-Head Attention
  - Positional Encoding
  - Layer Normalization
  - Encoder & Decoder Blocks
  - Final Projection Layer
- `dataset.py` – Custom `BilingualDataset` class with:
  - Tokenizer-based sentence encoding
  - Padding and masking logic
  - Causal masking for decoder
- `config.py` – Centralized configuration for hyperparameters and file paths.
- `train.py` (assumed) – Training loop that loads data, builds the model, and trains it using PyTorch.

## Configuration

All settings are defined in `config.py` via the `get_config()` function:

```python
{
  "batch_size": 8,
  "num_epochs": 20,
  "lr": 1e-4,
  "seq_len": 350,
  "d_model": 512,
  "datasource": "opus_books",
  "lang_src": "en",
  "lang_tgt": "it",
  "model_folder": "weights",
  "model_basename": "tmodel_",
  "preload": "latest",
  "tokenizer_file": "tokenizer_{lang}.json",
  "experiment_name": "runs/tmodel"
}
```

## Model Highlights

- Implements the original Transformer architecture (Vaswani et al., 2017)
- Modular design with separate blocks for encoder, decoder, attention, and embeddings
- Causal masking for autoregressive decoding
- Xavier weight initialization for improved convergence

## Requirements

Minimal dependencies (see `requirements.txt`):

- `torch`
- `datasets`
- `tokenizers`
- `tqdm`
- `numpy`

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset**

   ```python
   from datasets import load_dataset
   dataset = load_dataset("opus_books", "en-it")
   ```

3. **Train model**Customize and run your training script (`train.py`) using:

   ```python
   config = get_config()
   model = build_transformer(...)
   ```

## Outputs

- **Model checkpoints**: Saved in `./opus_books_weights/`
- **Tokenizer files**: `tokenizer_en.json`, `tokenizer_it.json`
- **TensorBoard logs**: `runs/tmodel/`

## Evaluation

The model outputs predictions in token ID format, which can be decoded using the trained tokenizer. Evaluate performance using BLEU or custom accuracy metrics.

## References

- Vaswani et al., "Attention is All You Need", 2017
- HuggingFace Datasets: opus_books
- HuggingFace Tokenizers Library
- **Coding a Transformer from scratch on PyTorch, with full explanation, training and inference.**