import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        '''
        Learned positional encoding using nn.Embedding.
        Args:
            d_model: Dimension of the embeddings.
            max_len: Maximum length of input sequences.
        '''
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):  # x: (batch, seq_len, d_model)
        '''
        Add positional encoding to the input embeddings.
        Args:
            x: Input embeddings (batch_size, seq_len, d_model).
        Returns:
            x + positional embeddings.
        '''
        pos = torch.arange(x.size(1), device=x.device).view(1, x.size(1))  # Ensure pos is on the same device as x
        embedding = self.pos_embedding(pos)  # (1, seq_len, d_model)
        return x + embedding
