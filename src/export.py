import torch
import torch.nn as nn
import os
import argparse
from src.model import TransformerDecoder
from src.utils import create_causal_mask
from src.dataset import Tokenizer, download_data

class TransformerOnnxWrapper(nn.Module):
    """
    Wrapper to handle mask generation inside the ONNX model.
    Input: x [batch, seq_len]
    Output: logits [batch, seq_len, vocab_size]
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        seq_len = x.size(1)
        # Create mask dynamically on the same device as input
        mask = create_causal_mask(seq_len).to(x.device)
        return self.model(x, mask)

def export_onnx(checkpoint_path, output_path="transformer.onnx"):
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Needs to match training config
    # Ideally these should be saved in the checkpoint or a config file
    # For now, we hardcode to match train.py defaults
    batch_size = 1
    seq_length = 256
    d_model = 384
    n_head = 6
    num_layers = 6
    d_ff = 4 * d_model
    dropout = 0.0 # No dropout for inference!
    
    # We need to know vocab size. 
    # Option 1: Load tokenizer (requires downloading data again if not present)
    # Option 2: Hardcode if known
    # Let's load tokenizer to be safe
    file_path = download_data()
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = Tokenizer(text)
    vocab_size = tokenizer.vocab_size

    device = torch.device("cpu") # Export usually done on CPU

    model = TransformerDecoder(vocab_size, d_model, n_head, d_ff, num_layers, dropout)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return

    model.eval()
    
    # Wrap model to simplify inputs (handle mask internally)
    onnx_model = TransformerOnnxWrapper(model)
    
    # Dummy input
    dummy_input = torch.randint(0, vocab_size, (1, seq_length), dtype=torch.long)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        onnx_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14, # 14 supports tril/triu better
        do_constant_folding=True,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_length'},
            'logits': {0: 'batch_size', 1: 'seq_length'}
        }
    )
    print("Export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--output", type=str, default="transformer.onnx", help="Output ONNX file path")
    args = parser.parse_args()
    
    export_onnx(args.checkpoint, args.output)
