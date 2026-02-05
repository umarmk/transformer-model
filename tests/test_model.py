import unittest
import torch
from src.model import InputEmbeddings, PositionalEncoding, MultiHeadAttention, TransformerDecoder
from src.utils import create_causal_mask

class TestTransformerComponents(unittest.TestCase):
    def test_embeddings(self):
        vocab_size = 100
        d_model = 512
        emb = InputEmbeddings(d_model, vocab_size)
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = emb(x)
        self.assertEqual(out.shape, (2, 3, 512))

    def test_positional_encoding(self):
        d_model = 512
        pe = PositionalEncoding(d_model)
        x = torch.zeros(1, 10, d_model)
        out = pe(x)
        self.assertEqual(out.shape, (1, 10, 512))

    def test_multi_head_attention(self):
        d_model = 512
        n_head = 8
        mha = MultiHeadAttention(d_model, n_head)
        x = torch.randn(2, 10, d_model)
        mask = create_causal_mask(10)
        out = mha(x, x, x, mask)
        self.assertEqual(out.shape, (2, 10, 512))

    def test_transformer_decoder(self):
        vocab_size = 100
        d_model = 512
        n_head = 8
        d_ff = 2048
        num_layers = 2
        model = TransformerDecoder(vocab_size, d_model, n_head, d_ff, num_layers)
        
        x = torch.randint(0, vocab_size, (2, 10))
        mask = create_causal_mask(10)
        out = model(x, mask)
        self.assertEqual(out.shape, (2, 10, 100)) # (batch, seq, vocab)


if __name__ == '__main__':
    unittest.main()
