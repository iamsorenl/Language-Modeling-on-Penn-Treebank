import torch
import torch.nn as nn
from encoding import LearnedPositionalEncoding  # Import your positional encoding module

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)  # Embed tokens
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_len=max_seq_len)  # Embed positions

        # Define Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # Final prediction layer to map to vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):  # x: (batch, seq_len)
        # Token and positional embeddings
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)  # Add positional encoding

        # Generate square subsequent mask
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)  # (seq_len, seq_len)

        # Transformer encoder
        encoded = self.transformer_encoder(x, mask=mask)  # (batch, seq_len, d_model)

        # Predict next token probabilities
        logits = self.fc_out(encoded)  # (batch, seq_len, vocab_size)
        return logits
