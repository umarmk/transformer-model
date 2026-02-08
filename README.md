# Transformer Model from Scratch (PyTorch)

An implementation of the Transformer architecture (specifically a Decoder-only GPT-style model) built from scratch in PyTorch. Trained on the Shakespeare dataset from Kaggle to generate Shakespearean-like text.

## Project Overview

This project implements the core components of the "Attention Is All You Need" paper to help understand the inner workings of Transformers:

- **Architecture**: Decoder-only Transformer (suited for text generation).
- **Components**: Multi-Head Attention, Sinusoidal Positional Encoding, Feed-Forward Networks, Layer Normalization.
- **Dataset**: Character-level tokenization of the Shakespeare dataset.
- **Hardware**: Optimized for CUDA-enabled GPUs 

## Structure

- `src/model.py`: The Transformer architecture implementation.
- `src/train.py`: Training loop, loss calculation, and text generation.
- `src/dataset.py`: Data downloading, processing, and Tokenizer.
- `src/utils.py`: Masking utilities (causal masks).
- `tests/`: Unit tests for model components.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Training Options

### 1. Local Training (Accelerated)

The training script now supports **Mixed Precision Training (AMP)** automatically on CUDA devices. This reduces memory usage and speeds up training.

```bash
python -m src.train
```

### 2. Cloud Training (Google Colab)

Running on a laptop without a strong GPU? Use Google Colab's free T4 GPUs.

1. Upload the `src` folder to Colab files.
2. Open and run `notebooks/colab_training.ipynb`.
   _Or simply upload `notebooks/colab_training.ipynb` to Colab and follow the instructions therein._

### 3. Model Export & Usage

When training finishes (locally or on Colab), checkpoints are saved as `.pt` files in the `checkpoints/` folder.

- **On Colab**: The notebook includes a step to zip and download these files.
- **Why .pt?**: PyTorch saves model weights (state dicts) in `.pt` or `.pth` format. This is the standard, not `.pkl`.

#### Export to ONNX

You can convert the trained PyTorch model to **ONNX** for deployment (e.g., to run in browser or C++).

```bash
python -m src.export --checkpoint checkpoints/model_epoch_5.pt --output transformer.onnx
```

**To load a saved model locally:**

```python
model = TransformerDecoder(...)
model.load_state_dict(torch.load("checkpoints/model_epoch_5.pt"))
model.eval()
```

## Usage

### Train the Model

Start training on the Shakespeare dataset:

```bash
python -m src.train
```

This will download the data, train for 5 epochs, and output generated text samples.

### Run Tests

Verify the implementation with unit tests:

```bash
python -m unittest discover tests
```

## Credits

Based on the "Attention Is All You Need" paper architecture.
Dataset provided by `adarshpathak/shakespeare-text` via Kaggle.
