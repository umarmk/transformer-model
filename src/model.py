import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create a matrix of [max_len, d_model] representing the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as a buffer (not a learnable parameter, but part of state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # output shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.d_model = d_model
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        # Shape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_head, d_head]
        # Then transpose to [batch_size, n_head, seq_len, d_head]
        query = self.w_q(q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # scores shape: [batch_size, n_head, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            # Mask should be [seq_len, seq_len] or broadcastable
            # Apply mask (mask positions with True are filled with -inf)
            scores = scores.masked_fill(mask == 1, float('-inf'))
            
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Weighted sum of values
        # output shape: [batch_size, n_head, seq_len, d_head]
        output = torch.matmul(attention, value)
        
        # Concatenate and final linear projection
        # Shape: [batch_size, seq_len, n_head * d_head] = [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Sublayer 1: Masked Multi-Head Attention
        # Residual connection + LayerNorm
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Sublayer 2: Feed Forward
        # Residual connection + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        # Weight tying 
        # self.projection.weight = self.embedding.embed.weight

    def forward(self, x, mask):
        # x shape: [batch_size, seq_len]
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        output = self.projection(x)
        return output
