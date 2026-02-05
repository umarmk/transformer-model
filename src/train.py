import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from src.dataset import get_dataloader, Tokenizer
from src.model import TransformerDecoder
from src.utils import create_causal_mask

def train():
    # Hardware check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Hyperparameters
    batch_size = 64 # Increased batch size for GPU
    seq_length = 256
    d_model = 384
    n_head = 6
    num_layers = 6
    d_ff = 4 * d_model
    dropout = 0.2
    learning_rate = 3e-4
    epochs = 5 # Small number for demonstration
    
    # Data
    print("Loading data...")
    dataloader, tokenizer = get_dataloader(batch_size, seq_length)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Model
    model = TransformerDecoder(vocab_size, d_model, n_head, d_ff, num_layers, dropout).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            # Create mask
            # x shape: [batch_size, seq_len]
            seq_len = x.size(1)
            mask = create_causal_mask(seq_len).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                output = model(x, mask) # [batch_size, seq_len, vocab_size]
                
                # Loss calculation
                # View: [batch_size * seq_len, vocab_size] vs [batch_size * seq_len]
                loss = criterion(output.view(-1, vocab_size), y.view(-1))
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Generation sample
        generate_sample(model, tokenizer, device, start_text="\n", max_new_tokens=100)
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pt")

def generate_sample(model, tokenizer, device, start_text=" ", max_new_tokens=100):
    model.eval()
    context = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if it becomes too long (to fit in positional encoding/block size)
            # In our implementation, positional encoding goes up to 5000, so we are safe for a bit,
            # but usually we train with seq_length, so we should probably crop to that.
            # However, for simple generation, we can just feed the last seq_length tokens.
            cond_ctx = context[:, -256:] # assuming max context length we trained on (256)
            
            seq_len = cond_ctx.size(1)
            mask = create_causal_mask(seq_len).to(device)
            
            logits = model(cond_ctx, mask) 
            # Focus only on the last time step
            logits = logits[:, -1, :] # [batch_size, vocab_size]
            
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            context = torch.cat((context, next_token), dim=1)
            
    decoded_text = tokenizer.decode(context[0].tolist())
    print("\n--- Generated Sample ---")
    print(decoded_text)
    print("------------------------\n")
    model.train()

if __name__ == "__main__":
    train()
