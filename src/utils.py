import torch

def create_causal_mask(size):
    """
    Creates a causal mask for self-attention.
    Returns: a boolean mask where True indicates positions to be masked (future positions).
    Shape: [size, size]
    """
    # Create a lower triangular matrix of ones (allowed positions)
    # The mask should be True where we want to BLOCK attention.
    # So we want upper triangular (diagonal=1) to be True.
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)
    return mask

