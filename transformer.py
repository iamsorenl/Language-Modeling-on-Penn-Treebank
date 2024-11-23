import torch
import torch.nn as nn
from encoding import LearnedPositionalEncoding 

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_seq_len):
        super().__init__()
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Positional encoding
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_len=max_seq_len)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        # Final output projection to vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, pad_mask):  
        """
        Args:
            x: Input sequence tensor of shape (batch_size, seq_len)
            pad_mask: Padding mask of shape (batch_size, seq_len)
        Returns:
            logits: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Token and positional embeddings
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)  # Add positional encoding

        # Generate causal mask (upper triangular mask for auto-regressive decoding)
        seq_len = x.size(1)
        device = x.device  # Ensure all tensors are created on the same device as `x`
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Pass through Transformer Encoder
        encoded = self.transformer_encoder(
            x,
            mask=causal_mask,  # Causal mask for auto-regression
            src_key_padding_mask=~pad_mask.bool()  # Invert pad_mask for compatibility with PyTorch
        )

        # Project encoder output to vocabulary logits
        logits = self.fc_out(encoded)  # (batch_size, seq_len, vocab_size)
        return logits
